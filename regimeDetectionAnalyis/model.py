"""
model.py
========
Supervised regime classification engine.

Pipeline for a single "experiment":
  1. Build features (features.py) and regime labels (labeling.py).
  2. Shift labels to create a T+N-ahead prediction target (predict the
     regime that will be in effect N days from now, using only
     information available today) -> this is what makes the task
     "actionable" for tactical allocation decisions made today.
  3. Tune RandomForest / XGBoost hyperparameters with Optuna, scoring each
     trial via `PurgedGroupTimeSeriesSplit` (cv.py) so the tuner itself
     cannot overfit to leaked folds.
  4. Refit the best model on a final walk-forward split, report
     precision/recall/F1/confusion matrix, and compute SHAP-based feature
     importance.

Leakage discipline
-------------------
`StandardScaler` is fit ONLY on each fold's training slice inside
`_fit_predict_fold`, and applied (transform-only) to that fold's test
slice. No scaler is ever fit on data that includes test-period samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score,
)
from xgboost import XGBClassifier

from features import build_feature_matrix, FEATURE_COLUMNS
from labeling import label_regimes_hmm
from cv import PurgedGroupTimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

REGIME_ORDER = ["Bear_HighVol", "Sideways", "Bull_LowVol"]
REGIME_TO_INT = {name: i for i, name in enumerate(REGIME_ORDER)}
INT_TO_REGIME = {i: name for name, i in REGIME_TO_INT.items()}


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series          # int-encoded regime, target is regime at t+horizon
    regime_at_t: pd.Series  # regime AS OF t (not shifted) -- useful for backtest
    dates: pd.DatetimeIndex


def build_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
    labeling_fn=label_regimes_hmm,
) -> Dataset:
    """
    Assemble the modeling dataset:
      - features at time t (causal, using only data up to t)
      - target = regime label at time t + horizon (int-encoded)
      - also retains the *current* (unshifted) regime label, which the
        backtest module uses to simulate "regime becomes known and acted
        upon" style strategies as a complement to the pure forecasting task.
    """
    feats = build_feature_matrix(df)[FEATURE_COLUMNS]
    regime = labeling_fn(df)

    target = regime.shift(-horizon)  # T+N ahead label, aligned to t

    combined = feats.join(target.rename("target")).join(regime.rename("current_regime"))
    combined = combined.dropna(subset=FEATURE_COLUMNS + ["target", "current_regime"])

    y = combined["target"].map(REGIME_TO_INT).astype(int)
    X = combined[FEATURE_COLUMNS]
    regime_at_t = combined["current_regime"]

    return Dataset(X=X, y=y, regime_at_t=regime_at_t, dates=combined.index)


# --------------------------------------------------------------------------- #
# Fold-level fit/predict with leakage-safe scaling
# --------------------------------------------------------------------------- #
def _fit_predict_fold(model_ctor, X, y, train_idx, test_idx):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])   # fit ONLY on train fold
    X_test = scaler.transform(X.iloc[test_idx])          # transform-only on test fold

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = model_ctor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return y_test.values, preds, model, scaler


def _cv_macro_f1(model_ctor, X, y, cv: PurgedGroupTimeSeriesSplit) -> float:
    scores = []
    for train_idx, test_idx in cv.split(X):
        if len(np.unique(y.iloc[train_idx])) < 2:
            continue
        y_true, y_pred, _, _ = _fit_predict_fold(model_ctor, X, y, train_idx, test_idx)
        scores.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------- #
# Optuna tuning
# --------------------------------------------------------------------------- #
def tune_random_forest(X, y, cv, n_trials: int = 25, seed: int = 42) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        ctor = lambda: RandomForestClassifier(**params)
        return _cv_macro_f1(ctor, X, y, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(X, y, cv, n_trials: int = 25, seed: int = 42) -> dict:
    n_classes = y.nunique()

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
        ctor = lambda: XGBClassifier(**params)
        return _cv_macro_f1(ctor, X, y, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# --------------------------------------------------------------------------- #
# Final evaluation on a held-out walk-forward split
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    model: object
    scaler: StandardScaler
    report: str
    conf_matrix: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    test_dates: pd.DatetimeIndex


def evaluate_final_model(model_ctor, X, y, dates, cv: PurgedGroupTimeSeriesSplit) -> EvalResult:
    """
    Use the LAST fold of the purged CV splitter as the final held-out test
    set (the most recent, most realistic out-of-sample period), fit on
    everything before it (respecting the purge gap), and report metrics.
    """
    splits = list(cv.split(X))
    train_idx, test_idx = splits[-1]

    y_true, y_pred, model, scaler = _fit_predict_fold(model_ctor, X, y, train_idx, test_idx)

    labels_present = sorted(set(y_true) | set(y_pred))
    target_names = [INT_TO_REGIME[i] for i in labels_present]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    report = classification_report(
        y_true, y_pred, labels=labels_present, target_names=target_names, zero_division=0
    )

    return EvalResult(
        y_true=y_true,
        y_pred=y_pred,
        model=model,
        scaler=scaler,
        report=report,
        conf_matrix=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        test_dates=dates[test_idx],
    )


def shap_feature_importance(model, X_sample: pd.DataFrame, scaler: StandardScaler, max_samples: int = 500):
    """
    Compute mean |SHAP value| per feature (averaged across classes for
    multi-class models) as a model-agnostic importance ranking.
    """
    import shap

    X_scaled = pd.DataFrame(
        scaler.transform(X_sample), columns=X_sample.columns, index=X_sample.index
    )
    if len(X_scaled) > max_samples:
        X_scaled = X_scaled.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # shap_values can be a list (per-class) or a 3D array depending on
    # model/version; normalize to mean absolute importance per feature.
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
        mean_abs = stacked.mean(axis=(0, 1))
    elif shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    importance = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)
    return importance


if __name__ == "__main__":
    from data_loader import load_market_data

    df, _ = load_market_data(synthetic_kwargs={"n_days": 3000})
    ds = build_dataset(df, horizon=5)
    print("Dataset:", ds.X.shape, "target distribution:\n", ds.y.value_counts())

    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window=25, embargo_window=10)

    rf_params = tune_random_forest(ds.X, ds.y, cv, n_trials=8)
    print("Best RF params:", rf_params)

    rf_ctor = lambda: RandomForestClassifier(**rf_params, class_weight="balanced", random_state=42, n_jobs=-1)
    result = evaluate_final_model(rf_ctor, ds.X, ds.y, ds.dates, cv)
    print(result.report)

    importance = shap_feature_importance(result.model, ds.X.iloc[-300:], result.scaler)
    print(importance.head(10))
