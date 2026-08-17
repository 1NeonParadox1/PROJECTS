"""
features.py
============
Feature engineering for market-regime detection.

Design principles
------------------
1. **Non-stationarity**: we work with log returns rather than raw price
   levels for anything drift/volatility related. Raw price levels are never
   fed to the model directly.
2. **No look-ahead bias**: every rolling/EWM computation at time t uses only
   information available up to and including t (pandas `.rolling` /
   `.ewm` are causal by construction — we never use `center=True` and never
   shift features backward). Where an indicator conventionally needs a
   "current bar close" (e.g., RSI/MACD on close[t]), that is still
   information available at the close of bar t, i.e. valid for a decision
   made at t to be acted on at t+1's open (see `model.py` for the label
   shift that encodes this).
3. **Scaling** is deliberately NOT done in this module. Standardization is
   fitted per training fold inside the CV loop (see `model.py` /
   `cv.py`) to avoid leaking test-fold statistics into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Returns & basic transforms
# --------------------------------------------------------------------------- #
def log_returns(close: pd.Series) -> pd.Series:
    """Log return r_t = ln(C_t / C_{t-1}). First value is NaN."""
    return np.log(close / close.shift(1))


# --------------------------------------------------------------------------- #
# Volatility features
# --------------------------------------------------------------------------- #
def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualized realized volatility from log returns (causal)."""
    return returns.rolling(window).std() * np.sqrt(252)


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Wilder's Average True Range, normalized by close (ATR%) for
    scale-invariance across the sample period."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return atr / close  # ATR as a % of price -> stationary-ish, comparable across time


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 10) -> pd.Series:
    """
    Parkinson (1980) high-low range volatility estimator, annualized.
    More efficient than close-to-close vol since it uses the day's range.
    """
    hl_ratio_sq = (np.log(high / low)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    daily_var = factor * hl_ratio_sq
    rolling_var = daily_var.rolling(window).mean()
    return np.sqrt(rolling_var * 252)


# --------------------------------------------------------------------------- #
# Trend / momentum features
# --------------------------------------------------------------------------- #
def ema_spread(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """(EMA_fast - EMA_slow) / EMA_slow -- normalized trend strength."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow) / ema_slow


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram, normalized by price."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / close
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


# --------------------------------------------------------------------------- #
# Statistical / distributional features
# --------------------------------------------------------------------------- #
def rolling_skewness(returns: pd.Series, window: int = 30) -> pd.Series:
    return returns.rolling(window).skew()


def rolling_kurtosis(returns: pd.Series, window: int = 30) -> pd.Series:
    return returns.rolling(window).kurt()


# --------------------------------------------------------------------------- #
# Master feature-matrix builder
# --------------------------------------------------------------------------- #
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw OHLCV data, construct the full feature set described in the
    project spec. Returns a DataFrame aligned to `df.index`, with the
    warm-up NaN rows (from the longest rolling window) still present —
    callers should `dropna()` after joining with labels.

    All features are causal (computed only from data at/before time t).
    """
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    feats = pd.DataFrame(index=df.index)

    # --- returns (also the base for labeling) ---
    r = log_returns(c)
    feats["log_return"] = r
    feats["log_return_5d"] = np.log(c / c.shift(5))

    # --- volatility features ---
    feats["realized_vol_10d"] = realized_volatility(r, 10)
    feats["realized_vol_30d"] = realized_volatility(r, 30)
    feats["realized_vol_60d"] = realized_volatility(r, 60)
    feats["atr_pct_14d"] = average_true_range(h, l, c, window=14)
    feats["parkinson_vol_10d"] = parkinson_volatility(h, l, window=10)
    # vol-of-vol: how unstable is volatility itself (regime transitions
    # tend to show up here first)
    feats["vol_of_vol_30d"] = feats["realized_vol_10d"].rolling(30).std()

    # --- trend / momentum ---
    feats["ema_spread_12_26"] = ema_spread(c, 12, 26)
    feats["ema_spread_20_100"] = ema_spread(c, 20, 100)
    feats["rsi_14"] = rsi(c, 14)
    macd_df = macd(c)
    feats = feats.join(macd_df)
    # price vs long-run trend
    feats["price_vs_sma200"] = c / c.rolling(200).mean() - 1

    # --- statistical / distributional ---
    feats["rolling_skew_30d"] = rolling_skewness(r, 30)
    feats["rolling_kurt_30d"] = rolling_kurtosis(r, 30)

    # --- volume-based (secondary confirmation feature) ---
    feats["volume_zscore_20d"] = (
        (v - v.rolling(20).mean()) / v.rolling(20).std()
    )

    return feats


FEATURE_COLUMNS = [
    "log_return",
    "log_return_5d",
    "realized_vol_10d",
    "realized_vol_30d",
    "realized_vol_60d",
    "atr_pct_14d",
    "parkinson_vol_10d",
    "vol_of_vol_30d",
    "ema_spread_12_26",
    "ema_spread_20_100",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "price_vs_sma200",
    "rolling_skew_30d",
    "rolling_kurt_30d",
    "volume_zscore_20d",
]


if __name__ == "__main__":
    from data_loader import load_market_data

    df, _ = load_market_data(synthetic_kwargs={"n_days": 1500})
    feats = build_feature_matrix(df)
    print(feats.shape)
    print(feats.dropna().shape)
    print(feats.describe().T)
