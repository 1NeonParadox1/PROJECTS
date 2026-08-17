"""
pipeline.py
===========
End-to-end orchestration script tying together data_loader -> labeling ->
features -> model -> backtest. This is what the accompanying Jupyter
notebook calls into; it can also be run standalone:

    python pipeline.py

Produces a dict of artifacts (datasets, tuned models, eval results,
backtest results) that the notebook uses for its charts, plus prints a
console summary.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_market_data
from labeling import label_regimes_hmm, label_regimes_gmm, label_regimes_rule_based
from model import (
    build_dataset,
    tune_random_forest,
    tune_xgboost,
    evaluate_final_model,
    shap_feature_importance,
    INT_TO_REGIME,
)
from cv import PurgedGroupTimeSeriesSplit
from backtest import run_regime_backtest, run_buy_and_hold, summarize_results


@dataclass
class PipelineArtifacts:
    df: pd.DataFrame
    true_regime: pd.Series | None
    regime_hmm: pd.Series
    regime_gmm: pd.Series
    regime_rule: pd.Series
    dataset: object
    cv: PurgedGroupTimeSeriesSplit
    rf_params: dict
    xgb_params: dict
    rf_result: object
    xgb_result: object
    rf_importance: pd.Series
    xgb_importance: pd.Series
    backtest_summary: pd.DataFrame
    backtests: dict


def run_pipeline(
    ticker: str = "SPY",
    start: str = "2012-01-01",
    horizon: int = 5,
    n_trials: int = 20,
    n_splits: int = 6,
    purge_window: int = 25,
    embargo_window: int = 10,
    synthetic_n_days: int = 3000,
    seed: int = 42,
) -> PipelineArtifacts:
    print(f"[1/6] Loading market data for {ticker}...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df, true_regime = load_market_data(
            ticker=ticker, start=start,
            synthetic_kwargs={"n_days": synthetic_n_days, "seed": seed},
        )
        for w in caught:
            print(f"    NOTE: {w.message}")
    print(f"    -> {len(df)} bars from {df.index.min().date()} to {df.index.max().date()}")

    print("[2/6] Labeling historical regimes (HMM primary; GMM & rule-based for comparison)...")
    regime_hmm = label_regimes_hmm(df, random_state=seed)
    regime_gmm = label_regimes_gmm(df, random_state=seed)
    regime_rule = label_regimes_rule_based(df)
    print("    HMM regime distribution:\n", regime_hmm.value_counts().to_string())

    print(f"[3/6] Building feature matrix + T+{horizon} target...")
    dataset = build_dataset(df, horizon=horizon, labeling_fn=lambda d: regime_hmm)
    print(f"    -> {dataset.X.shape[0]} samples x {dataset.X.shape[1]} features")

    cv = PurgedGroupTimeSeriesSplit(
        n_splits=n_splits, purge_window=purge_window, embargo_window=embargo_window
    )

    print(f"[4/6] Tuning RandomForest ({n_trials} Optuna trials, purged CV)...")
    rf_params = tune_random_forest(dataset.X, dataset.y, cv, n_trials=n_trials, seed=seed)
    print("    Best RF params:", rf_params)

    print(f"[4/6] Tuning XGBoost ({n_trials} Optuna trials, purged CV)...")
    xgb_params = tune_xgboost(dataset.X, dataset.y, cv, n_trials=n_trials, seed=seed)
    print("    Best XGB params:", xgb_params)

    print("[5/6] Evaluating both models on final held-out walk-forward fold...")
    rf_ctor = lambda: RandomForestClassifier(
        **rf_params, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    n_classes = dataset.y.nunique()
    xgb_ctor = lambda: XGBClassifier(
        **xgb_params, objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", random_state=seed, n_jobs=-1,
    )

    rf_result = evaluate_final_model(rf_ctor, dataset.X, dataset.y, dataset.dates, cv)
    xgb_result = evaluate_final_model(xgb_ctor, dataset.X, dataset.y, dataset.dates, cv)

    print("\n--- RandomForest classification report (held-out fold) ---")
    print(rf_result.report)
    print("--- XGBoost classification report (held-out fold) ---")
    print(xgb_result.report)

    rf_importance = shap_feature_importance(
        rf_result.model, dataset.X.loc[rf_result.test_dates], rf_result.scaler
    )
    xgb_importance = shap_feature_importance(
        xgb_result.model, dataset.X.loc[xgb_result.test_dates], xgb_result.scaler
    )

    print("[6/6] Backtesting regime-switching allocation vs buy & hold...")
    # (a) "perfect info" backtest using the current (unshifted) HMM regime
    #     label -- upper bound on strategy potential if regimes were known
    #     instantly and perfectly.
    perfect_info_bt = run_regime_backtest(df["Close"], regime_hmm)

    # (b) "model-driven" backtest using the RandomForest's out-of-sample
    #     T+horizon-ahead predictions on the held-out fold, i.e. a genuinely
    #     forward-looking, non-leaked signal.
    test_dates = rf_result.test_dates
    pred_regime = pd.Series(
        [INT_TO_REGIME[i] for i in rf_result.y_pred], index=test_dates, name="pred_regime"
    )
    model_bt_close = df["Close"].loc[test_dates.min():]
    model_driven_bt = run_regime_backtest(model_bt_close, pred_regime)

    bh_bt = run_buy_and_hold(model_bt_close)
    bh_bt_full = run_buy_and_hold(df["Close"])

    backtests = {
        "perfect_info": perfect_info_bt,
        "model_driven": model_driven_bt,
        "buy_and_hold_test_period": bh_bt,
        "buy_and_hold_full_history": bh_bt_full,
    }

    backtest_summary = summarize_results(
        {
            "Regime-Switching (Perfect Info, full hist.)": perfect_info_bt,
            "Regime-Switching (RF Model, held-out fold)": model_driven_bt,
            "Buy & Hold (held-out fold, same period)": bh_bt,
            "Buy & Hold (full history)": bh_bt_full,
        }
    )
    print(backtest_summary)

    return PipelineArtifacts(
        df=df,
        true_regime=true_regime,
        regime_hmm=regime_hmm,
        regime_gmm=regime_gmm,
        regime_rule=regime_rule,
        dataset=dataset,
        cv=cv,
        rf_params=rf_params,
        xgb_params=xgb_params,
        rf_result=rf_result,
        xgb_result=xgb_result,
        rf_importance=rf_importance,
        xgb_importance=xgb_importance,
        backtest_summary=backtest_summary,
        backtests=backtests,
    )


if __name__ == "__main__":
    artifacts = run_pipeline(n_trials=15, synthetic_n_days=3000)
