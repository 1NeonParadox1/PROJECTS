"""
backtest.py
===========
Financial strategy validation layer: turns predicted (or actual) regime
labels into a tactical asset-allocation strategy and benchmarks it against
passive buy-and-hold using standard risk-adjusted return metrics.

Strategy (regime-switching allocation)
---------------------------------------
  - Bull_LowVol   -> 100% equity
  - Sideways      -> 50% equity / 50% cash
  - Bear_HighVol  -> 0% equity / 100% cash (hedge proxy)

Decisions made on day t (using a regime signal known at the close of t)
are applied to the return realized from t to t+1 (`shift(1)` on weights)
to avoid look-ahead: you cannot trade on today's close using information
only available at today's close and have it affect today's return.

Metrics
-------
CAGR, annualized volatility, Sharpe ratio, Sortino ratio, and maximum
drawdown, computed from the daily strategy-return series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIME_WEIGHTS = {
    "Bull_LowVol": 1.00,
    "Sideways": 0.50,
    "Bear_HighVol": 0.00,
}

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # cumulative growth of $1
    daily_returns: pd.Series
    weights: pd.Series
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float


def _perf_metrics(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> dict:
    daily_returns = daily_returns.dropna()
    n = len(daily_returns)
    if n == 0:
        return dict(cagr=np.nan, ann_vol=np.nan, sharpe=np.nan, sortino=np.nan,
                    max_drawdown=np.nan, calmar=np.nan)

    growth = (1 + daily_returns).cumprod()
    total_return = growth.iloc[-1] - 1
    years = n / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)

    rf_daily = risk_free_annual / TRADING_DAYS
    excess = daily_returns - rf_daily
    sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan

    downside = excess[excess < 0]
    downside_std = downside.std()
    sortino = (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS) if downside_std and downside_std > 0 else np.nan

    running_max = growth.cummax()
    drawdown = growth / running_max - 1
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return dict(
        cagr=cagr, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
        max_drawdown=max_drawdown, calmar=calmar,
    )


def run_regime_backtest(
    close: pd.Series,
    regime_signal: pd.Series,
    regime_weights: dict = REGIME_WEIGHTS,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """
    Simulate the regime-switching allocation strategy.

    Parameters
    ----------
    close : pd.Series
        Close prices, full index (used to compute asset daily returns).
    regime_signal : pd.Series
        Regime label KNOWN as of the close of day t (e.g., the model's
        T+N-ahead forecast made N days ago and now realized, or the
        current-regime label if simulating a "perfect information" /
        rule-based strategy baseline). Must be aligned to (a subset of)
        close's index.
    regime_weights : dict
        Regime name -> target equity weight.
    transaction_cost_bps : float
        Round-trip-ish cost applied whenever the weight changes, in basis
        points of notional traded (e.g. 5 bps = 0.05%).

    Returns
    -------
    BacktestResult
    """
    asset_returns = close.pct_change()

    aligned_regime = regime_signal.reindex(close.index).ffill()
    target_weight = aligned_regime.map(regime_weights)

    # Decision made using info available at close of t is executed starting
    # the NEXT bar -> shift(1) prevents look-ahead.
    applied_weight = target_weight.shift(1)

    strategy_gross_returns = applied_weight * asset_returns

    # Transaction costs: charged when applied_weight changes vs the prior day
    weight_changes = applied_weight.diff().abs().fillna(0)
    costs = weight_changes * (transaction_cost_bps / 1e4)

    strategy_returns = (strategy_gross_returns - costs).dropna()

    equity_curve = (1 + strategy_returns).cumprod()
    metrics = _perf_metrics(strategy_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=strategy_returns,
        weights=applied_weight,
        **metrics,
    )


def run_buy_and_hold(close: pd.Series) -> BacktestResult:
    asset_returns = close.pct_change().dropna()
    equity_curve = (1 + asset_returns).cumprod()
    metrics = _perf_metrics(asset_returns)
    weights = pd.Series(1.0, index=asset_returns.index)
    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=asset_returns,
        weights=weights,
        **metrics,
    )


def summarize_results(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """results: {'Strategy Name': BacktestResult, ...} -> comparison table."""
    rows = {}
    for name, r in results.items():
        rows[name] = {
            "CAGR": r.cagr,
            "Ann. Vol": r.ann_vol,
            "Sharpe": r.sharpe,
            "Sortino": r.sortino,
            "Max Drawdown": r.max_drawdown,
            "Calmar": r.calmar,
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from data_loader import load_market_data
    from labeling import label_regimes_hmm

    df, _ = load_market_data(synthetic_kwargs={"n_days": 3000})
    regime = label_regimes_hmm(df)

    strat = run_regime_backtest(df["Close"], regime)
    bh = run_buy_and_hold(df["Close"])

    summary = summarize_results({"Regime-Switching (perfect-info)": strat, "Buy & Hold": bh})
    print(summary)
