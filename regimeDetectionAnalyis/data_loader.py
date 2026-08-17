"""
data_loader.py
==============
Handles ingestion of daily OHLCV data for the regime-detection pipeline.

Primary path: `fetch_ohlcv()` pulls real data via `yfinance`. This requires
outbound internet access to Yahoo Finance endpoints, which is available on a
normal machine / CI runner but may be blocked in sandboxed environments.

Fallback path: `generate_synthetic_ohlcv()` produces a realistic multi-year
daily OHLCV series driven by a *known* Markov regime-switching process
(different drift/volatility per hidden state). This is useful for:
  - Unit testing the pipeline without network access
  - Demonstrating the pipeline end-to-end in restricted environments
  - Sanity-checking that the labeling/classification stages recover
    regimes that we know are "true" by construction

`load_market_data()` is the single entry point used by the rest of the
pipeline: it tries the real download first and transparently falls back to
synthetic data (with a loud warning) if that fails.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# --------------------------------------------------------------------------- #
# Real data ingestion
# --------------------------------------------------------------------------- #
def fetch_ohlcv(
    ticker: str = "SPY",
    start: str = "2012-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data via yfinance.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "SPY", "^GSPC", "QQQ".
    start, end : str
        Date range (YYYY-MM-DD). `end=None` means "up to today".
    interval : str
        Bar interval, default daily.

    Returns
    -------
    pd.DataFrame indexed by Date with columns [Open, High, Low, Close, Volume]

    Raises
    ------
    RuntimeError if the download fails or returns no data (e.g. no network,
    invalid ticker, rate limiting).
    """
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for ticker='{ticker}'. "
            "Check network access / ticker symbol / date range."
        )

    # yfinance sometimes returns MultiIndex columns (ticker, field) even for
    # a single ticker depending on version — flatten defensively.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[REQUIRED_COLUMNS].copy()
    df.index.name = "Date"
    df = df.sort_index()
    df = df.dropna(how="any")
    return df


# --------------------------------------------------------------------------- #
# Synthetic fallback: Markov regime-switching price simulator
# --------------------------------------------------------------------------- #
@dataclass
class RegimeParams:
    name: str
    mu: float       # daily drift (log-return mean)
    sigma: float     # daily volatility (log-return std)


DEFAULT_REGIMES = [
    RegimeParams("Bull_LowVol", mu=0.00055, sigma=0.006),
    RegimeParams("Sideways", mu=0.00000, sigma=0.009),
    RegimeParams("Bear_HighVol", mu=-0.00090, sigma=0.021),
]

# Transition matrix: regimes are "sticky" (persist for weeks/months), matching
# real market behaviour where regime switches are infrequent.
DEFAULT_TRANSITION = np.array([
    [0.985, 0.012, 0.003],
    [0.020, 0.960, 0.020],
    [0.004, 0.016, 0.980],
])


def generate_synthetic_ohlcv(
    n_days: int = 3000,
    start_date: str = "2012-01-02",
    regimes: list[RegimeParams] | None = None,
    transition: np.ndarray | None = None,
    start_price: float = 100.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate a daily OHLCV series driven by a hidden Markov regime process.

    Returns
    -------
    df : pd.DataFrame  -- OHLCV data, business-day indexed
    true_regime : pd.Series -- the ground-truth regime label used to
        generate each day's return (str). Useful ONLY for validating that
        unsupervised labeling recovers something sensible; the supervised
        pipeline should not be given this directly.
    """
    regimes = regimes or DEFAULT_REGIMES
    transition = transition if transition is not None else DEFAULT_TRANSITION
    rng = np.random.default_rng(seed)
    n_regimes = len(regimes)

    # Simulate hidden state path
    state = rng.integers(0, n_regimes)
    states = np.empty(n_days, dtype=int)
    for t in range(n_days):
        states[t] = state
        state = rng.choice(n_regimes, p=transition[state])

    # Simulate log returns conditional on state, with fat tails (Student-t)
    # to make the series more realistic than pure Gaussian.
    dof = 5
    t_draws = rng.standard_t(dof, size=n_days) / np.sqrt(dof / (dof - 2))
    mu = np.array([regimes[s].mu for s in states])
    sigma = np.array([regimes[s].sigma for s in states])
    log_returns = mu + sigma * t_draws

    close = start_price * np.exp(np.cumsum(log_returns))

    # Build plausible OHLC around the close path using intraday noise scaled
    # by the regime's volatility (higher vol regimes -> wider daily ranges).
    intraday_scale = sigma * 0.6
    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, intraday_scale[1:] * 0.15, n_days - 1))

    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, intraday_scale, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, intraday_scale, n_days)))
    low = np.minimum(low, np.minimum(open_, close) * 0.999)  # ensure low <= min(o,c)
    high = np.maximum(high, np.maximum(open_, close) * 1.001)  # ensure high >= max(o,c)

    # Volume: baseline + spikes during high-vol regimes
    base_vol = 5e7
    vol_multiplier = 1 + 3 * (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-9)
    volume = (base_vol * vol_multiplier * (1 + rng.normal(0, 0.15, n_days))).clip(min=1e6)

    idx = pd.bdate_range(start=start_date, periods=n_days, name="Date")
    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume.astype(np.int64),
        },
        index=idx,
    )

    regime_names = np.array([regimes[s].name for s in states])
    true_regime = pd.Series(regime_names, index=idx, name="true_regime")

    return df, true_regime


# --------------------------------------------------------------------------- #
# Unified entry point
# --------------------------------------------------------------------------- #
def load_market_data(
    ticker: str = "SPY",
    start: str = "2012-01-01",
    end: str | None = None,
    use_synthetic_fallback: bool = True,
    synthetic_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Try to fetch real OHLCV data; fall back to synthetic data if the
    download fails and `use_synthetic_fallback=True`.

    Returns
    -------
    df : pd.DataFrame OHLCV
    true_regime : pd.Series or None
        Ground-truth regime labels — ONLY populated when synthetic data is
        used (real market data has no ground truth). Useful for validating
        the unsupervised labeling stage.
    """
    try:
        df = fetch_ohlcv(ticker=ticker, start=start, end=end)
        return df, None
    except Exception as exc:  # noqa: BLE001 - broad on purpose, many failure modes
        if not use_synthetic_fallback:
            raise
        warnings.warn(
            f"Live data fetch failed ({exc!r}). Falling back to SYNTHETIC "
            "OHLCV data for demonstration/testing purposes. Results below "
            "do NOT reflect real market history.",
            stacklevel=2,
        )
        kwargs = synthetic_kwargs or {}
        df, true_regime = generate_synthetic_ohlcv(**kwargs)
        return df, true_regime


if __name__ == "__main__":
    data, truth = load_market_data(ticker="SPY", start="2012-01-01")
    print(data.head())
    print(data.shape)
    if truth is not None:
        print(truth.value_counts())
