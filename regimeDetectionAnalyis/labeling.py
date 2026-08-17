"""
labeling.py
============
Generates ground-truth-proxy regime labels for historical bars, which the
supervised classifiers (RandomForest / XGBoost) are then trained to predict
ahead of time.

Two labeling mechanisms are provided:

1. `label_regimes_gmm()` — unsupervised Gaussian Mixture Model clustering on
   [trailing return, trailing volatility]. This is the "let the data speak"
   approach: it finds statistical clusters without any hard thresholds.

2. `label_regimes_hmm()` — Hidden Markov Model over the same feature space.
   Unlike GMM, an HMM models the *transition dynamics* between regimes
   (regimes are "sticky" in reality — you don't teleport between bull and
   bear day to day), which tends to produce more temporally coherent labels
   (less flickering) than i.i.d. clustering like GMM.

3. `label_regimes_rule_based()` — a transparent, auditable alternative using
   rolling return-direction and volatility-percentile thresholds. Useful as
   a sanity check / baseline against the unsupervised methods, and as a
   fallback when you want fully deterministic, explainable labels.

All methods return a categorical Series aligned to the input index with
values drawn from {"Bull_LowVol", "Sideways", "Bear_HighVol"}. Cluster/state
identities from GMM/HMM are arbitrary integers by default — we canonicalize
them by sorting on mean return so label semantics are consistent and
interpretable regardless of which method produced them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

REGIME_NAMES_ASC = ["Bear_HighVol", "Sideways", "Bull_LowVol"]  # sorted by mean return


def _labeling_features(close: pd.Series, ret_window: int = 20, vol_window: int = 20) -> pd.DataFrame:
    """Trailing return + trailing volatility used as the clustering input.
    Deliberately a *low-dimensional* summary (not the full feature set from
    features.py) — regime labeling should reflect broad market character,
    not the fine-grained signals we later ask the classifier to learn."""
    log_ret = np.log(close / close.shift(1))
    trailing_return = log_ret.rolling(ret_window).mean() * 252  # annualized
    trailing_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    out = pd.DataFrame({"trailing_return": trailing_return, "trailing_vol": trailing_vol})
    return out


def _canonicalize_labels(raw_labels: np.ndarray, cluster_means: np.ndarray) -> np.ndarray:
    """
    Map arbitrary cluster ids -> {Bear_HighVol, Sideways, Bull_LowVol} by
    ranking clusters on mean trailing return (ascending). Assumes exactly 3
    clusters, matching the project's 3-regime spec.
    """
    order = np.argsort(cluster_means)  # ascending return: worst -> best
    id_to_name = {cluster_id: REGIME_NAMES_ASC[rank] for rank, cluster_id in enumerate(order)}
    return np.array([id_to_name[c] for c in raw_labels])


def label_regimes_gmm(
    df: pd.DataFrame,
    n_regimes: int = 3,
    ret_window: int = 20,
    vol_window: int = 20,
    random_state: int = 42,
) -> pd.Series:
    """Unsupervised regime labels via Gaussian Mixture Model clustering."""
    feats = _labeling_features(df["Close"], ret_window, vol_window)
    valid = feats.dropna()

    gmm = GaussianMixture(
        n_components=n_regimes, covariance_type="full", random_state=random_state, n_init=5
    )
    raw = gmm.fit_predict(valid.values)

    # cluster mean return = column 0 of gmm.means_
    cluster_mean_return = gmm.means_[:, 0]
    named = _canonicalize_labels(raw, cluster_mean_return)

    labels = pd.Series(index=df.index, dtype=object, name="regime")
    labels.loc[valid.index] = named
    return labels


def label_regimes_hmm(
    df: pd.DataFrame,
    n_regimes: int = 3,
    ret_window: int = 20,
    vol_window: int = 20,
    random_state: int = 42,
    n_iter: int = 200,
) -> pd.Series:
    """
    Unsupervised regime labels via a Gaussian Hidden Markov Model.
    Captures transition persistence (sticky regimes), typically yielding
    smoother / less-flickering labels than plain GMM clustering.
    """
    from hmmlearn.hmm import GaussianHMM

    feats = _labeling_features(df["Close"], ret_window, vol_window)
    valid = feats.dropna()

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(valid.values)
    raw = model.predict(valid.values)

    cluster_mean_return = model.means_[:, 0]
    named = _canonicalize_labels(raw, cluster_mean_return)

    labels = pd.Series(index=df.index, dtype=object, name="regime")
    labels.loc[valid.index] = named
    return labels


def label_regimes_rule_based(
    df: pd.DataFrame,
    ret_window: int = 20,
    vol_window: int = 20,
    vol_percentile_high: float = 0.67,
    vol_percentile_low: float = 0.33,
) -> pd.Series:
    """
    Transparent rule-based labeling using rolling return direction and
    volatility percentile thresholds (expanding-window percentiles, so no
    look-ahead: each day's percentile rank is computed only against history
    up to that day).

    Rule:
      - High vol (top tercile, expanding) & negative trailing return -> Bear_HighVol
      - Low/mid vol & positive trailing return -> Bull_LowVol
      - everything else -> Sideways
    """
    feats = _labeling_features(df["Close"], ret_window, vol_window)

    # Expanding-window percentile rank avoids look-ahead: percentile of
    # vol[t] is computed using only vol[0..t].
    vol_rank = feats["trailing_vol"].expanding(min_periods=vol_window * 2).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else np.nan,
        raw=False,
    )

    labels = pd.Series(index=df.index, dtype=object, name="regime")
    ret = feats["trailing_return"]

    is_high_vol = vol_rank >= vol_percentile_high
    is_low_vol = vol_rank <= vol_percentile_low
    is_pos_ret = ret > 0
    is_neg_ret = ret < 0

    labels[is_high_vol & is_neg_ret] = "Bear_HighVol"
    labels[~is_high_vol & is_pos_ret & ~is_neg_ret.isna()] = "Bull_LowVol"
    remaining = labels.isna() & vol_rank.notna() & ret.notna()
    labels[remaining] = "Sideways"

    return labels


if __name__ == "__main__":
    from data_loader import load_market_data

    df, truth = load_market_data(synthetic_kwargs={"n_days": 2500})

    gmm_labels = label_regimes_gmm(df)
    hmm_labels = label_regimes_hmm(df)
    rule_labels = label_regimes_rule_based(df)

    print("GMM distribution:\n", gmm_labels.value_counts())
    print("\nHMM distribution:\n", hmm_labels.value_counts())
    print("\nRule-based distribution:\n", rule_labels.value_counts())

    if truth is not None:
        comp = pd.DataFrame({"true": truth, "gmm": gmm_labels, "hmm": hmm_labels}).dropna()
        print("\nGMM vs true agreement: %.1f%%" % (100 * (comp["true"] == comp["gmm"]).mean()))
        print("HMM vs true agreement: %.1f%%" % (100 * (comp["true"] == comp["hmm"]).mean()))
