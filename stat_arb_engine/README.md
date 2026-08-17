# Multi-Asset Statistical Arbitrage & Cointegration Engine

A modular Python engine that detects cointegrated futures pairs, models a
dynamic (Kalman-filtered) hedge ratio, generates mean-reversion z-score
signals, and backtests the strategy under realistic trading frictions.

## Project layout

```
stat_arb_engine/
├── data/                  # (optional) local OHLCV cache
├── src/
│   ├── data_processor.py  # download, continuous-contract stitching, cleaning
│   ├── cointegration.py   # ADF/KPSS, Engle-Granger, Johansen, OLS, Kalman filter, Hurst, OU half-life
│   ├── strategy.py        # z-score engine, entry/exit/stop rules, position sizing
│   └── backtester.py      # event-driven backtest engine + performance metrics
├── app.py                 # Streamlit interactive dashboard
├── main.py                # CLI entry point (no dashboard)
├── requirements.txt
└── README.md
```

## Install

```bash
cd stat_arb_engine
pip install -r requirements.txt
```

## Run

**CLI (prints diagnostics + a performance summary, saves CSVs):**
```bash
python main.py --y CL=F --x BZ=F --start 2018-01-01
```

**Dashboard:**
```bash
streamlit run app.py
```

Both need outbound network access to `query1/query2.finance.yahoo.com`
(via `yfinance`) to pull live futures data — enable that in your
environment's network/egress settings if downloads fail with a 403.

## Preset pairs

| Pair | Y ticker | X ticker |
|---|---|---|
| WTI vs Brent Crude | `CL=F` | `BZ=F` |
| Gold vs Silver | `GC=F` | `SI=F` |
| 10Y vs 2Y Treasury futures | `ZN=F` | `ZT=F` |

Any two Yahoo Finance tickers work — pass your own via `--y/--x` (CLI) or
the "Custom" preset (dashboard).

## Methodology

1. **Data pipeline** (`data_processor.py`)
   Downloads OHLCV via `yfinance`. Yahoo's `=F` tickers are already
   vendor-stitched continuous front-month series, so this module's
   `stitch_continuous()` is provided for the case where you supply your
   own individual-expiry legs (e.g. from an exchange feed) — it does a
   proper Panama (additive) or ratio (multiplicative) back-adjustment at
   each roll date. `align_pair()` inner-joins two series on common
   timestamps and forward-fills only isolated single-bar gaps using
   strictly past data (no look-ahead).

2. **Econometrics** (`cointegration.py`)
   - ADF + KPSS on levels and first differences to confirm both legs are
     I(1).
   - Engle-Granger two-step test (OLS hedge ratio + ADF-on-residuals via
     `statsmodels.coint`, which uses the correct MacKinnon critical
     values for residual-based cointegration).
   - Johansen trace/eigenvalue test for basket (>2 asset) cointegration.
   - `KalmanHedgeRatio`: a 2-state (`beta`, `alpha`) random-walk state-
     space Kalman filter that updates the hedge ratio on every new bar,
     causally (no look-ahead — each `beta_t` only uses data through `t`).
   - Hurst exponent (rescaled variance method) and Ornstein-Uhlenbeck
     half-life (`t_½ = ln(2)/θ`, θ estimated by regressing `ΔS_t` on
     `S_{t-1}`) as mean-reversion diagnostics.

3. **Strategy** (`strategy.py`)
   - `Spread_t = Y_t - (α_t + β_t · X_t)` from the Kalman filter output.
   - Rolling z-score of the spread.
   - Rules: enter long spread at `Z ≤ -2.0`, short spread at `Z ≥ +2.0`,
     exit at `|Z| ≤ 0.2` or a zero-crossing, force-liquidate at
     `|Z| ≥ 3.5` (cointegration-breakdown stop). All thresholds are
     configurable.
   - Position sizing via inverse-ATR volatility scaling, with an optional
     fractional-Kelly cap.

4. **Backtester** (`backtester.py`)
   Event-driven, bar-by-bar loop. Applies commission per contract,
   slippage in ticks, and daily financing cost on margin held. Tracks
   cash, equity, realized/unrealized PnL, and a full trade log. Reports:
   Total Return, CAGR, annualized Sharpe & Sortino, Calmar, Max Drawdown
   (%and duration), win rate, and max consecutive losses — benchmarked
   against passive buy-and-hold on the Y leg.

5. **Dashboard** (`app.py`)
   Streamlit + Plotly: price/spread/Kalman-β/z-score panel with
   entry/exit/stop markers, equity curve vs. benchmark, drawdown chart,
   a metrics table, and an expandable full econometric diagnostics /
   trade-log panel.

## Important caveats

- **Yahoo `=F` tickers are already continuous** front-month series — true
  Panama/ratio back-adjustment across individual expiries requires raw
  per-contract data, which Yahoo does not expose. `stitch_continuous()`
  is ready for that data if you source it elsewhere (CME DataMine,
  Norgate, a broker API, etc.).
- **Backtest assumptions are simplified**: fixed per-contract commission,
  constant slippage in ticks, and a flat annualized financing rate on
  margin. Contract multipliers, tick sizes, and margin percentages in
  `BacktestConfig` are illustrative defaults — set them to the actual
  contract specs for the instruments you're trading before trusting the
  P&L in dollar terms.
- **Not investment advice** — this is a research/engineering scaffold,
  not a production trading system. Cointegration relationships can and
  do break down; the `stop_z` regime-break rule is a partial safeguard,
  not a guarantee.
