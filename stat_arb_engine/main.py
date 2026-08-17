"""
main.py
=======
Command-line entry point: run the full cointegration + Kalman + backtest
pipeline on a chosen futures pair and print a risk/performance summary.

Usage
-----
    python main.py --y CL=F --x BZ=F --start 2018-01-01
    python main.py --y GC=F --x SI=F --start 2015-01-01 --interval 1d
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data_processor import download_ohlcv, align_pair, sanity_check
from src.cointegration import (
    confirm_i1,
    engle_granger_test,
    johansen_test,
    ols_hedge_ratio,
    KalmanHedgeRatio,
    hurst_exponent,
    ou_half_life,
)
from src.strategy import StrategyParams, build_strategy
from src.backtester import BacktestConfig, run_backtest, benchmark_buy_and_hold


def parse_args():
    p = argparse.ArgumentParser(description="Statistical Arbitrage & Cointegration Engine")
    p.add_argument("--y", type=str, default="CL=F", help="Yahoo ticker for Y leg (e.g. CL=F WTI Crude)")
    p.add_argument("--x", type=str, default="BZ=F", help="Yahoo ticker for X leg (e.g. BZ=F Brent Crude)")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.2)
    p.add_argument("--stop-z", type=float, default=3.5)
    p.add_argument("--z-window", type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n=== Downloading {args.y} & {args.x} ({args.interval}) from {args.start} ===")
    df_y = sanity_check(download_ohlcv(args.y, start=args.start, end=args.end, interval=args.interval))
    df_x = sanity_check(download_ohlcv(args.x, start=args.start, end=args.end, interval=args.interval))
    pair = align_pair(df_y, df_x)

    print("\n=== Unit Root Tests (confirming I(1)) ===")
    ur = confirm_i1(pair["Y"], pair["X"])
    for r in ur.values():
        print(r.summary())

    print("\n=== Engle-Granger Cointegration Test ===")
    eg = engle_granger_test(pair["Y"], pair["X"])
    print(f"beta={eg.beta:.4f}  alpha={eg.alpha:.4f}  t-stat={eg.coint_t_stat:.4f}  "
          f"p-value={eg.coint_pvalue:.4f}  cointegrated@5%={eg.is_cointegrated_5pct}")

    print("\n=== Johansen Cointegration Test ===")
    jt = johansen_test(pair[["Y", "X"]])
    print(jt.summary())

    print("\n=== Mean-Reversion Diagnostics (static OLS spread) ===")
    h = hurst_exponent(eg.residuals)
    ou = ou_half_life(eg.residuals)
    print(f"Hurst exponent: {h:.4f} ({'mean-reverting' if h < 0.5 else 'trending/random-walk'})")
    print(f"OU half-life: {ou.half_life_bars:.2f} bars  (theta={ou.theta:.5f}, mu={ou.mu:.4f})")

    print("\n=== Kalman Filter Dynamic Hedge Ratio ===")
    kf = KalmanHedgeRatio(delta=1e-4, obs_var=1e-3)
    kf_out = kf.filter(pair["Y"], pair["X"])
    print(f"Final beta: {kf_out['beta'].iloc[-1]:.4f}  (started at {kf_out['beta'].iloc[0]:.4f})")

    print("\n=== Building Strategy Signals ===")
    params = StrategyParams(entry_z=args.entry_z, exit_z=args.exit_z, stop_z=args.stop_z, z_window=args.z_window)
    strat_df = build_strategy(pair, kf_out, params)
    n_long = (strat_df["signal"] == 1).sum()
    n_short = (strat_df["signal"] == -1).sum()
    print(f"Bars in LONG_SPREAD: {n_long}  |  Bars in SHORT_SPREAD: {n_short}")

    print("\n=== Running Backtest ===")
    config = BacktestConfig(initial_capital=args.capital)
    result = run_backtest(strat_df, config, params)

    print("\n=== Performance & Risk Summary ===")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:,.3f}")
        else:
            print(f"  {k:28s}: {v}")

    bench = benchmark_buy_and_hold(pair["Y"], args.capital)
    bench_total_return = (bench.iloc[-1] / bench.iloc[0] - 1) * 100
    print(f"\n  Benchmark (buy&hold {args.y}) Total Return %: {bench_total_return:,.3f}")

    out_path = "backtest_equity_curve.csv"
    result.equity_curve.to_frame("strategy_equity").join(bench.rename("benchmark_equity")).to_csv(out_path)
    print(f"\nSaved equity curve to {out_path}")

    if len(result.trades) > 0:
        trades_path = "backtest_trades.csv"
        result.trades.to_csv(trades_path, index=False)
        print(f"Saved trade log to {trades_path}")


if __name__ == "__main__":
    sys.exit(main())
