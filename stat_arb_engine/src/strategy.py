"""
strategy.py
===========
Z-score based mean-reversion signal generation and volatility-adjusted
position sizing for the cointegrated spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class Signal(IntEnum):
    FLAT = 0
    LONG_SPREAD = 1   # long Y, short beta*X
    SHORT_SPREAD = -1  # short Y, long beta*X


@dataclass
class StrategyParams:
    entry_z: float = 2.0
    exit_z: float = 0.2
    stop_z: float = 3.5
    z_window: int = 60          # rolling window (bars) for mean/std of the spread
    atr_window: int = 20        # rolling window for ATR-based sizing
    use_kelly: bool = False
    kelly_fraction: float = 0.5  # fractional Kelly cap
    kelly_win_rate: float = 0.55
    kelly_win_loss_ratio: float = 1.2
    max_position_notional: float = 1.0  # cap as a fraction of equity


def compute_zscore(spread: pd.Series, window: int) -> pd.DataFrame:
    """Rolling z-score of the spread (uses only trailing/past data)."""
    roll_mean = spread.rolling(window, min_periods=window).mean()
    roll_std = spread.rolling(window, min_periods=window).std()
    z = (spread - roll_mean) / roll_std
    return pd.DataFrame({"spread": spread, "roll_mean": roll_mean, "roll_std": roll_std, "zscore": z})


def generate_signals(zscore: pd.Series, params: StrategyParams) -> pd.Series:
    """
    Stateful signal generator implementing:
      - Long spread entry:  z <= -entry_z
      - Short spread entry: z >= +entry_z
      - Mean exit:          |z| <= exit_z  OR z crosses zero while in a position
      - Stop-loss / regime break: |z| >= stop_z  -> force flat

    Returns a Series of Signal enum values aligned to zscore's index.
    """
    signals = pd.Series(Signal.FLAT, index=zscore.index, dtype=int)
    position = Signal.FLAT
    prev_z = np.nan

    for t, z in zscore.items():
        if np.isnan(z):
            signals.loc[t] = position
            prev_z = z
            continue

        if position == Signal.FLAT:
            if z <= -params.entry_z:
                position = Signal.LONG_SPREAD
            elif z >= params.entry_z:
                position = Signal.SHORT_SPREAD

        elif position == Signal.LONG_SPREAD:
            crossed_zero = (not np.isnan(prev_z)) and (prev_z < 0 <= z)
            if abs(z) <= params.exit_z or crossed_zero or z >= params.stop_z:
                position = Signal.FLAT
            elif z <= -params.stop_z:
                # regime break beyond stop on the same side -> force liquidate
                position = Signal.FLAT

        elif position == Signal.SHORT_SPREAD:
            crossed_zero = (not np.isnan(prev_z)) and (prev_z > 0 >= z)
            if abs(z) <= params.exit_z or crossed_zero or z <= -params.stop_z:
                position = Signal.FLAT
            elif z >= params.stop_z:
                position = Signal.FLAT

        signals.loc[t] = position
        prev_z = z

    return signals


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range (Wilder), computed causally."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def kelly_position_fraction(params: StrategyParams) -> float:
    """
    Fractional Kelly criterion sizing:
        f* = W - (1-W)/R
    where W = win rate, R = win/loss ratio. Capped by kelly_fraction and by
    max_position_notional.
    """
    w, r = params.kelly_win_rate, params.kelly_win_loss_ratio
    f = w - (1 - w) / r if r > 0 else 0.0
    f = max(f, 0.0) * params.kelly_fraction
    return min(f, params.max_position_notional)


def volatility_adjusted_size(
    equity: float,
    price_y: float,
    atr_y: float,
    params: StrategyParams,
    risk_per_trade: float = 0.01,
) -> float:
    """
    Inverse-ATR volatility-adjusted position sizing: risk a fixed fraction
    of equity per trade, scaled by the leg's ATR. Returns the notional
    fraction of equity to allocate to the Y leg (X leg is beta * this,
    handled by the backtester).
    """
    if atr_y <= 0 or np.isnan(atr_y):
        return 0.0
    dollar_risk = equity * risk_per_trade
    contracts_equiv = dollar_risk / atr_y
    notional = contracts_equiv * price_y
    frac = notional / equity if equity > 0 else 0.0

    if params.use_kelly:
        frac = min(frac, kelly_position_fraction(params))

    return float(np.clip(frac, 0.0, params.max_position_notional))


def build_strategy(
    df_prices: pd.DataFrame,  # columns Y, X (and optionally Y_high/Y_low for ATR)
    kalman_out: pd.DataFrame,  # columns beta, alpha, spread (from KalmanHedgeRatio)
    params: StrategyParams,
) -> pd.DataFrame:
    """
    Full pipeline: dynamic spread -> rolling z-score -> signals -> sizing
    scaffold. Returns a single DataFrame ready for the backtester.
    """
    z_df = compute_zscore(kalman_out["spread"], params.z_window)
    signals = generate_signals(z_df["zscore"], params)

    out = pd.concat(
        [df_prices[["Y", "X"]], kalman_out[["beta", "alpha", "spread"]], z_df[["roll_mean", "roll_std", "zscore"]]],
        axis=1,
    )
    out["signal"] = signals
    return out.dropna(subset=["zscore"])
