"""
backtester.py
=============
Event-driven backtesting engine for the cointegrated-pair spread strategy.
Tracks cash, equity, margin, realized/unrealized PnL bar-by-bar, applies
transaction frictions (commission, slippage, financing), and reports
institutional-style performance & risk metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import Signal, StrategyParams, volatility_adjusted_size, atr


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission_per_contract: float = 2.50       # fixed $ fee per contract, per side
    slippage_ticks: float = 1.0                 # bid-ask/impact slippage, in ticks
    tick_size_y: float = 0.01
    tick_size_x: float = 0.01
    contract_multiplier_y: float = 1000.0        # $ per 1.0 price move, per contract
    contract_multiplier_x: float = 1000.0
    initial_margin_pct: float = 0.10             # % of notional held as initial margin
    maintenance_margin_pct: float = 0.075
    annual_financing_rate: float = 0.04          # cost of carry / margin financing
    risk_per_trade: float = 0.01                 # used by volatility_adjusted_size
    bars_per_year: int = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    cash_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _perf_metrics(equity: pd.Series, bars_per_year: int, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    n_years = len(equity) / bars_per_year if bars_per_year else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1.0 if n_years > 0 else np.nan

    ann_vol = rets.std() * np.sqrt(bars_per_year)
    sharpe = (rets.mean() * bars_per_year) / ann_vol if ann_vol > 0 else np.nan

    downside = rets[rets < 0]
    downside_vol = downside.std() * np.sqrt(bars_per_year) if len(downside) > 0 else np.nan
    sortino = (rets.mean() * bars_per_year) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    # Drawdown duration (peak-to-trough, longest underwater streak)
    underwater = drawdown < 0
    max_dd_duration = 0
    cur = 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        max_dd_duration = max(max_dd_duration, cur)

    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    win_rate = np.nan
    max_consec_losses = 0
    if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
        wins = trades["pnl"] > 0
        win_rate = wins.mean()
        cur_losses = 0
        for is_win in wins:
            if not is_win:
                cur_losses += 1
                max_consec_losses = max(max_consec_losses, cur_losses)
            else:
                cur_losses = 0

    return {
        "Total Return %": total_return * 100,
        "CAGR %": cagr * 100 if not np.isnan(cagr) else np.nan,
        "Annualized Vol %": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown %": max_dd * 100,
        "Max DD Duration (bars)": max_dd_duration,
        "Win Rate %": win_rate * 100 if not np.isnan(win_rate) else np.nan,
        "Max Consecutive Losses": max_consec_losses,
        "Number of Trades": len(trades) if trades is not None else 0,
    }


def run_backtest(strategy_df: pd.DataFrame, config: BacktestConfig, params: StrategyParams) -> BacktestResult:
    """
    Event-driven walk-forward loop over `strategy_df`, which must contain
    columns: Y, X, beta, spread, zscore, signal (see strategy.build_strategy).

    Position sizing: number of Y contracts is volatility/ATR-scaled off
    equity; the X leg is sized at beta * (Y contracts) to remain hedge-
    neutral per the dynamic Kalman beta at entry.
    """
    df = strategy_df.copy()
    df["atr_y"] = atr(df["Y"], df["Y"], df["Y"], params.atr_window)  # proxy ATR from close-only data
    # If real OHLC for Y is available under Y_high/Y_low, prefer that:
    if "Y_high" in df.columns and "Y_low" in df.columns:
        df["atr_y"] = atr(df["Y_high"], df["Y_low"], df["Y"], params.atr_window)

    cash = config.initial_capital
    equity_curve = []
    cash_curve = []
    position_records = []
    trades = []

    pos_signal = Signal.FLAT
    contracts_y = 0.0
    contracts_x = 0.0
    entry_price_y = entry_price_x = None
    entry_equity = None
    realized_pnl = 0.0

    daily_financing_rate = config.annual_financing_rate / config.bars_per_year

    prev_row = None
    for t, row in df.iterrows():
        y_price, x_price, beta, sig = row["Y"], row["X"], row["beta"], Signal(row["signal"])

        # --- mark-to-market unrealized PnL on existing position ---
        unrealized = 0.0
        if pos_signal != Signal.FLAT and entry_price_y is not None:
            unrealized = (
                contracts_y * (y_price - entry_price_y) * config.contract_multiplier_y
                + contracts_x * (x_price - entry_price_x) * config.contract_multiplier_x
            )
            # financing cost on margin held
            notional = abs(contracts_y) * y_price * config.contract_multiplier_y + abs(
                contracts_x
            ) * x_price * config.contract_multiplier_x
            financing_cost = notional * config.initial_margin_pct * daily_financing_rate
            cash -= financing_cost

        equity = cash + realized_pnl * 0 + unrealized + (0 if pos_signal == Signal.FLAT else 0)
        # NOTE: realized pnl already folded into cash at close; equity = cash + open unrealized
        equity = cash + unrealized

        # --- handle signal transitions ---
        if sig != pos_signal:
            # Close existing position (if any) at this bar's price with slippage+fees
            if pos_signal != Signal.FLAT:
                exit_slip_y = config.slippage_ticks * config.tick_size_y
                exit_slip_x = config.slippage_ticks * config.tick_size_x
                exit_price_y = y_price - np.sign(contracts_y) * exit_slip_y
                exit_price_x = x_price - np.sign(contracts_x) * exit_slip_x

                pnl = (
                    contracts_y * (exit_price_y - entry_price_y) * config.contract_multiplier_y
                    + contracts_x * (exit_price_x - entry_price_x) * config.contract_multiplier_x
                )
                fees = (abs(contracts_y) + abs(contracts_x)) * config.commission_per_contract
                pnl -= fees
                cash += pnl
                realized_pnl += pnl

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": t,
                        "side": "LONG_SPREAD" if pos_signal == Signal.LONG_SPREAD else "SHORT_SPREAD",
                        "contracts_y": contracts_y,
                        "contracts_x": contracts_x,
                        "entry_price_y": entry_price_y,
                        "entry_price_x": entry_price_x,
                        "exit_price_y": exit_price_y,
                        "exit_price_x": exit_price_x,
                        "pnl": pnl,
                        "fees": fees,
                    }
                )
                contracts_y = contracts_x = 0.0
                entry_price_y = entry_price_x = None

            # Open new position (if the new signal isn't flat)
            if sig != Signal.FLAT:
                atr_y = row["atr_y"] if not np.isnan(row["atr_y"]) else y_price * 0.01
                notional_frac = volatility_adjusted_size(equity, y_price, atr_y, params, config.risk_per_trade)
                notional = notional_frac * equity
                raw_contracts_y = notional / (y_price * config.contract_multiplier_y) if y_price > 0 else 0.0

                direction = 1.0 if sig == Signal.LONG_SPREAD else -1.0
                contracts_y = direction * raw_contracts_y
                contracts_x = -direction * raw_contracts_y * beta  # hedge leg, opposite side, beta-scaled

                entry_slip_y = config.slippage_ticks * config.tick_size_y
                entry_slip_x = config.slippage_ticks * config.tick_size_x
                entry_price_y = y_price + np.sign(contracts_y) * entry_slip_y
                entry_price_x = x_price + np.sign(contracts_x) * entry_slip_x

                entry_fees = (abs(contracts_y) + abs(contracts_x)) * config.commission_per_contract
                cash -= entry_fees
                entry_time = t
                entry_equity = equity

            pos_signal = sig

        equity = cash + (
            0.0
            if pos_signal == Signal.FLAT
            else contracts_y * (y_price - entry_price_y) * config.contract_multiplier_y
            + contracts_x * (x_price - entry_price_x) * config.contract_multiplier_x
        )

        equity_curve.append(equity)
        cash_curve.append(cash)
        position_records.append(
            {
                "timestamp": t,
                "signal": int(pos_signal),
                "contracts_y": contracts_y,
                "contracts_x": contracts_x,
                "equity": equity,
            }
        )
        prev_row = row

    equity_series = pd.Series(equity_curve, index=df.index, name="equity")
    cash_series = pd.Series(cash_curve, index=df.index, name="cash")
    positions_df = pd.DataFrame(position_records).set_index("timestamp")
    trades_df = pd.DataFrame(trades)

    metrics = _perf_metrics(equity_series, config.bars_per_year, trades_df)

    return BacktestResult(
        equity_curve=equity_series,
        cash_curve=cash_series,
        positions=positions_df,
        trades=trades_df,
        metrics=metrics,
    )


def benchmark_buy_and_hold(price: pd.Series, initial_capital: float) -> pd.Series:
    """Passive buy-and-hold benchmark equity curve on a single price series."""
    rets = price.pct_change().fillna(0.0)
    equity = initial_capital * (1 + rets).cumprod()
    equity.name = "benchmark_equity"
    return equity
