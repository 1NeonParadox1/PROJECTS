"""
app.py
======
Interactive Streamlit dashboard for the Multi-Asset Statistical Arbitrage
& Cointegration Engine.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_processor import download_ohlcv, align_pair, sanity_check
from src.cointegration import (
    confirm_i1,
    engle_granger_test,
    johansen_test,
    KalmanHedgeRatio,
    hurst_exponent,
    ou_half_life,
)
from src.strategy import StrategyParams, build_strategy
from src.backtester import BacktestConfig, run_backtest, benchmark_buy_and_hold

st.set_page_config(page_title="Stat-Arb & Cointegration Engine", layout="wide")

PRESET_PAIRS = {
    "WTI vs Brent Crude (CL=F / BZ=F)": ("CL=F", "BZ=F"),
    "Gold vs Silver (GC=F / SI=F)": ("GC=F", "SI=F"),
    "10Y vs 2Y Treasury Futures (ZN=F / ZT=F)": ("ZN=F", "ZT=F"),
    "Custom": None,
}

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Engine Controls")

preset = st.sidebar.selectbox("Preset pair", list(PRESET_PAIRS.keys()))
if PRESET_PAIRS[preset] is None:
    ticker_y = st.sidebar.text_input("Y ticker (Yahoo Finance)", "CL=F")
    ticker_x = st.sidebar.text_input("X ticker (Yahoo Finance)", "BZ=F")
else:
    ticker_y, ticker_x = PRESET_PAIRS[preset]

col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Start date", pd.to_datetime("2018-01-01"))
interval = col_b.selectbox("Interval", ["1d", "1h"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Signal thresholds")
entry_z = st.sidebar.slider("Entry |Z|", 1.0, 3.5, 2.0, 0.1)
exit_z = st.sidebar.slider("Exit |Z|", 0.0, 1.0, 0.2, 0.05)
stop_z = st.sidebar.slider("Stop-loss |Z|", 2.5, 5.0, 3.5, 0.1)
z_window = st.sidebar.slider("Rolling Z-score window (bars)", 20, 250, 60, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Kalman filter")
kf_delta = st.sidebar.select_slider(
    "Beta adaptation speed (delta)", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4
)

st.sidebar.markdown("---")
st.sidebar.subheader("Account & frictions")
capital = st.sidebar.number_input("Initial capital ($)", value=1_000_000, step=100_000)
commission = st.sidebar.number_input("Commission per contract ($)", value=2.50, step=0.5)
slippage_ticks = st.sidebar.number_input("Slippage (ticks)", value=1.0, step=0.5)
risk_per_trade = st.sidebar.slider("Risk per trade (% of equity)", 0.1, 5.0, 1.0, 0.1) / 100.0

run_btn = st.sidebar.button("🚀 Run Engine", type="primary", use_container_width=True)

st.title("📊 Multi-Asset Statistical Arbitrage & Cointegration Engine")
st.caption(
    "Continuous futures cointegration detection • Kalman-filtered dynamic hedge ratios • "
    "Z-score mean-reversion signals • Event-driven backtest with realistic frictions"
)


# --------------------------------------------------------------------------- #
# Cached pipeline
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_pair(y_ticker, x_ticker, start, interval):
    df_y = sanity_check(download_ohlcv(y_ticker, start=str(start), interval=interval))
    df_x = sanity_check(download_ohlcv(x_ticker, start=str(start), interval=interval))
    return align_pair(df_y, df_x)


def run_pipeline(pair, params, kf_delta, config):
    ur = confirm_i1(pair["Y"], pair["X"])
    eg = engle_granger_test(pair["Y"], pair["X"])
    jt = johansen_test(pair[["Y", "X"]])
    h = hurst_exponent(eg.residuals)
    ou = ou_half_life(eg.residuals)

    kf = KalmanHedgeRatio(delta=kf_delta, obs_var=1e-3)
    kf_out = kf.filter(pair["Y"], pair["X"])

    strat_df = build_strategy(pair, kf_out, params)
    result = run_backtest(strat_df, config, params)
    bench = benchmark_buy_and_hold(pair["Y"], config.initial_capital)

    return dict(ur=ur, eg=eg, jt=jt, hurst=h, ou=ou, kf_out=kf_out, strat_df=strat_df, result=result, bench=bench)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if run_btn or "last_result" in st.session_state:
    if run_btn:
        with st.spinner(f"Downloading {ticker_y} & {ticker_x} and running full pipeline..."):
            try:
                pair = load_pair(ticker_y, ticker_x, start_date, interval)
                params = StrategyParams(entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, z_window=z_window)
                config = BacktestConfig(
                    initial_capital=float(capital),
                    commission_per_contract=float(commission),
                    slippage_ticks=float(slippage_ticks),
                    risk_per_trade=float(risk_per_trade),
                )
                st.session_state["last_result"] = run_pipeline(pair, params, kf_delta, config)
                st.session_state["last_pair"] = pair
                st.session_state["last_tickers"] = (ticker_y, ticker_x)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

    data = st.session_state["last_result"]
    pair = st.session_state["last_pair"]
    ty, tx = st.session_state["last_tickers"]
    eg, jt, kf_out, strat_df, result, bench = (
        data["eg"], data["jt"], data["kf_out"], data["strat_df"], data["result"], data["bench"]
    )

    # ---- Top-level stat cards ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cointegrated (Engle-Granger, 5%)", "Yes ✅" if eg.is_cointegrated_5pct else "No ❌",
               f"p={eg.coint_pvalue:.4f}")
    c2.metric("Johansen rank @5%", jt.rank_5pct)
    c3.metric("Hurst exponent", f"{data['hurst']:.3f}",
               "mean-reverting" if data["hurst"] < 0.5 else "trending")
    c4.metric("OU half-life (bars)", f"{data['ou'].half_life_bars:.1f}")
    c5.metric("Static OLS beta", f"{eg.beta:.3f}")

    st.markdown("---")

    # ---- Price / spread chart with entry markers ----
    st.subheader("Spread & Trade Signals")
    fig1 = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=(f"{ty} vs {tx} (Close)", "Dynamic Spread & Kalman Beta", "Rolling Z-Score"),
    )

    fig1.add_trace(go.Scatter(x=pair.index, y=pair["Y"], name=ty, line=dict(color="#2563eb")), row=1, col=1)
    fig1.add_trace(go.Scatter(x=pair.index, y=pair["X"], name=tx, line=dict(color="#f97316"), yaxis="y2"), row=1, col=1)

    fig1.add_trace(go.Scatter(x=strat_df.index, y=strat_df["spread"], name="Spread",
                                line=dict(color="#7c3aed")), row=2, col=1)
    fig1.add_trace(go.Scatter(x=strat_df.index, y=strat_df["beta"], name="Kalman β (hedge ratio)",
                                line=dict(color="#059669", dash="dot"), yaxis="y3"), row=2, col=1)

    fig1.add_trace(go.Scatter(x=strat_df.index, y=strat_df["zscore"], name="Z-Score",
                                line=dict(color="#0891b2")), row=3, col=1)
    for level, color, label in [(entry_z, "red", "Entry"), (-entry_z, "red", None),
                                  (stop_z, "black", "Stop"), (-stop_z, "black", None),
                                  (exit_z, "gray", "Exit"), (-exit_z, "gray", None)]:
        fig1.add_hline(y=level, line=dict(color=color, dash="dash", width=1), row=3, col=1)

    long_entries = strat_df[(strat_df["signal"] == 1) & (strat_df["signal"].shift(1) != 1)]
    short_entries = strat_df[(strat_df["signal"] == -1) & (strat_df["signal"].shift(1) != -1)]
    exits = strat_df[(strat_df["signal"] == 0) & (strat_df["signal"].shift(1) != 0)]

    fig1.add_trace(go.Scatter(x=long_entries.index, y=long_entries["zscore"], mode="markers",
                                name="Long Spread Entry", marker=dict(color="green", size=9, symbol="triangle-up")),
                    row=3, col=1)
    fig1.add_trace(go.Scatter(x=short_entries.index, y=short_entries["zscore"], mode="markers",
                                name="Short Spread Entry", marker=dict(color="red", size=9, symbol="triangle-down")),
                    row=3, col=1)
    fig1.add_trace(go.Scatter(x=exits.index, y=exits["zscore"], mode="markers",
                                name="Exit", marker=dict(color="gray", size=7, symbol="x")),
                    row=3, col=1)

    fig1.update_layout(height=850, hovermode="x unified", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ---- Equity curve ----
    st.subheader("Equity Curve: Strategy vs Benchmark")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve, name="Strategy Equity",
                                line=dict(color="#2563eb", width=2)))
    fig2.add_trace(go.Scatter(x=bench.index, y=bench, name=f"Buy & Hold {ty}",
                                line=dict(color="#9ca3af", width=1.5, dash="dot")))
    fig2.update_layout(height=420, hovermode="x unified", yaxis_title="Equity ($)")
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Drawdown ----
    dd = result.equity_curve / result.equity_curve.cummax() - 1.0
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", name="Drawdown %",
                                line=dict(color="#dc2626")))
    fig3.update_layout(height=250, yaxis_title="Drawdown %", hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ---- Risk & performance table ----
    st.subheader("Risk & Performance Summary")
    metrics_df = pd.DataFrame(result.metrics.items(), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df["Value"].apply(
        lambda v: f"{v:,.3f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v)
    )
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # ---- Cointegration diagnostics ----
    with st.expander("📐 Full Econometric Diagnostics"):
        st.markdown("**Unit Root Tests**")
        for r in data["ur"].values():
            st.text(r.summary())
        st.markdown("**Engle-Granger**")
        st.text(f"beta={eg.beta:.4f}  alpha={eg.alpha:.4f}  t-stat={eg.coint_t_stat:.4f}  "
                 f"p-value={eg.coint_pvalue:.4f}  cointegrated@5%={eg.is_cointegrated_5pct}")
        st.markdown("**Johansen**")
        st.text(jt.summary())
        st.markdown("**Mean Reversion**")
        st.text(f"Hurst={data['hurst']:.4f}   OU theta={data['ou'].theta:.5f}   "
                 f"OU mu={data['ou'].mu:.4f}   half-life={data['ou'].half_life_bars:.2f} bars")

    # ---- Trade log ----
    if len(result.trades) > 0:
        with st.expander(f"📋 Trade Log ({len(result.trades)} trades)"):
            st.dataframe(result.trades, use_container_width=True)
    else:
        st.info("No trades were triggered over this sample with the current thresholds.")

else:
    st.info("Configure a pair and parameters in the sidebar, then click **Run Engine**.")
    st.markdown(
        """
        **What this does:**
        1. Downloads continuous futures OHLCV for two correlated markets.
        2. Confirms both legs are I(1) (ADF + KPSS) and tests for cointegration
           (Engle-Granger + Johansen).
        3. Fits a Kalman filter to track the hedge ratio (β) and intercept (α)
           dynamically through time.
        4. Computes the rolling z-score of the spread and generates
           entry/exit/stop signals.
        5. Runs an event-driven backtest with commissions, slippage, and
           financing costs, then reports Sharpe/Sortino/Calmar, drawdown, and
           win-rate statistics against a buy-and-hold benchmark.
        """
    )
