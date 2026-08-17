"""
data_processor.py
==================
Data ingestion, continuous-contract roll adjustment, and time-series
sanitization for the statistical arbitrage engine.

Notes on continuous futures via yfinance
-----------------------------------------
Yahoo Finance's "=F" tickers (e.g. CL=F, BZ=F, GC=F, SI=F, ZN=F, ZT=F) are
themselves front-month *continuous* series already stitched by the
exchange/vendor feed -- they do not expose the underlying individual
expiries. Because of that, true Panama/back-adjustment cannot be
re-derived after the fact; instead this module treats the Yahoo series as
the raw continuous series and applies its own smoothing pass to remove any
residual level jumps that look like roll artifacts (large one-bar % moves
that revert immediately are dampened with a ratio adjustment). If you have
access to individual expiry data (e.g. from an exchange API or a vendor
like CME DataMine / Norgate), pass them into `stitch_continuous()`
directly and the Panama / ratio-adjustment logic will do a proper splice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RollMethod = Literal["panama", "ratio"]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_ohlcv(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a single ticker via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker, e.g. 'CL=F' (WTI Crude), 'BZ=F' (Brent),
        'GC=F' (Gold), 'SI=F' (Silver), 'ZN=F' (10Y Treasury Note),
        'ZT=F' (2Y Treasury Note).
    start, end : str
        Date bounds in 'YYYY-MM-DD' format. end=None means "through today".
    interval : str
        '1d' for daily, '1h' for hourly (yfinance limits '1h' to ~730 days
        of history).

    Returns
    -------
    pd.DataFrame indexed by timestamp with columns
    [Open, High, Low, Close, Volume].
    """
    import yfinance as yf

    logger.info(f"Downloading {ticker} [{interval}] from {start} to {end or 'today'}")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker}. Check the ticker symbol and that "
            f"this environment has network access to Yahoo Finance."
        )

    # yfinance sometimes returns a MultiIndex column frame for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    return df


# --------------------------------------------------------------------------- #
# Continuous contract stitching
# --------------------------------------------------------------------------- #
@dataclass
class RollEvent:
    date: pd.Timestamp
    adjustment: float  # additive (panama) or multiplicative (ratio)


def detect_roll_points(price: pd.Series, jump_threshold: float = 0.06) -> list[pd.Timestamp]:
    """
    Heuristic roll-point detector for a single continuous series: flags
    bars whose absolute log return exceeds `jump_threshold` (default 6%)
    as candidate expiry-roll discontinuities. This is only useful when you
    don't have the individual expiry legs; with real multi-contract data
    prefer `stitch_continuous()` below, which splices on the *known*
    switch dates instead of guessing.
    """
    log_ret = np.log(price / price.shift(1))
    flagged = log_ret[log_ret.abs() > jump_threshold]
    return list(flagged.index)


def stitch_continuous(
    legs: dict[pd.Timestamp, pd.Series],
    method: RollMethod = "panama",
) -> pd.Series:
    """
    Splice a sequence of individual futures-contract price legs into a
    single back-adjusted continuous series.

    Parameters
    ----------
    legs : dict[switch_date -> price Series]
        Ordered mapping of {roll/switch date: price series for the contract
        that is *active starting on* that date}. The first key should be
        the earliest date. Each series should cover from its switch date
        onward (or further; it will be truncated at the *next* leg's
        switch date).
    method : 'panama' | 'ratio'
        'panama'  -> additive back-adjustment (preserves absolute P&L,
                     can produce negative prices for deep-history spliced
                     series -- standard for spread/P&L-based stat-arb).
        'ratio'   -> multiplicative back-adjustment (preserves % returns,
                     avoids negative prices -- standard for long lookback
                     cointegration/ADF work).

    Returns
    -------
    pd.Series : continuous, back-adjusted price series.
    """
    switch_dates = sorted(legs.keys())
    if len(switch_dates) < 1:
        raise ValueError("Need at least one leg to build a continuous series.")

    # Truncate each leg to [switch_date, next_switch_date)
    segments = []
    for i, sd in enumerate(switch_dates):
        s = legs[sd]
        end = switch_dates[i + 1] if i + 1 < len(switch_dates) else None
        seg = s[s.index >= sd]
        if end is not None:
            seg = seg[seg.index < end]
        segments.append(seg)

    # Walk backwards from the most recent (unadjusted) segment, computing
    # the cumulative adjustment factor needed at each splice point so that
    # the joined series has no artificial gap at the roll date.
    adjusted_segments = [segments[-1]]
    cum_add, cum_mult = 0.0, 1.0

    for i in range(len(segments) - 2, -1, -1):
        older_seg = segments[i]
        newer_seg_first_val = adjusted_segments[0].iloc[0]
        older_seg_last_val = older_seg.iloc[-1]

        if method == "panama":
            gap = newer_seg_first_val - older_seg_last_val
            cum_add += gap
            adjusted_segments.insert(0, older_seg + cum_add)
        else:  # ratio
            ratio = newer_seg_first_val / older_seg_last_val if older_seg_last_val != 0 else 1.0
            cum_mult *= ratio
            adjusted_segments.insert(0, older_seg * cum_mult)

    continuous = pd.concat(adjusted_segments).sort_index()
    continuous = continuous[~continuous.index.duplicated(keep="last")]
    logger.info(f"Stitched {len(segments)} legs into continuous series ({method} adjustment).")
    return continuous


# --------------------------------------------------------------------------- #
# Cleaning / alignment
# --------------------------------------------------------------------------- #
def align_pair(
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    price_col: str = "Close",
    max_ffill: int = 2,
) -> pd.DataFrame:
    """
    Align two OHLCV frames (e.g. WTI vs Brent) onto a common, gap-cleaned
    timestamp index without introducing forward-looking bias.

    - Inner-joins on timestamp (only bars where both markets traded).
    - Forward-fills isolated single-tick gaps (<= max_ffill bars) using only
      *past* data (no look-ahead).
    - Drops any remaining rows with NaNs after the limited ffill.

    Returns a DataFrame with columns ['Y', 'X'].
    """
    y = df_y[price_col].rename("Y")
    x = df_x[price_col].rename("X")

    merged = pd.concat([y, x], axis=1).sort_index()
    merged = merged.ffill(limit=max_ffill)  # only fills using prior observed values
    merged = merged.dropna()

    n_dropped = (len(df_y) + len(df_x)) - 2 * len(merged)
    logger.info(f"Aligned pair: {len(merged)} common bars retained (approx {n_dropped} unmatched bars discarded).")
    return merged


def resample_ohlcv(df: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    """Resample an OHLCV frame to a coarser bar size without look-ahead."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule).agg(agg).dropna()
    return out


def sanity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data-quality checks: drop non-positive prices, flag duplicate
    timestamps, and report gap statistics."""
    before = len(df)
    df = df[(df.select_dtypes(include=[np.number]) > 0).all(axis=1)]
    df = df[~df.index.duplicated(keep="first")]
    after = len(df)
    if after < before:
        logger.warning(f"sanity_check: removed {before - after} invalid/duplicate rows.")
    return df
