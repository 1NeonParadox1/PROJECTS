"""
cointegration.py
=================
Econometric & statistical toolkit: unit-root tests, cointegration tests,
static (OLS) and dynamic (Kalman filter) hedge-ratio estimation, Hurst
exponent, and Ornstein-Uhlenbeck half-life estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# --------------------------------------------------------------------------- #
# Unit root tests
# --------------------------------------------------------------------------- #
@dataclass
class UnitRootResult:
    series_name: str
    adf_stat: float
    adf_pvalue: float
    adf_is_stationary_5pct: bool
    kpss_stat: float
    kpss_pvalue: float
    kpss_is_stationary_5pct: bool

    def summary(self) -> str:
        return (
            f"[{self.series_name}]  ADF: stat={self.adf_stat:.4f} p={self.adf_pvalue:.4f} "
            f"(stationary@5%={self.adf_is_stationary_5pct})   "
            f"KPSS: stat={self.kpss_stat:.4f} p={self.kpss_pvalue:.4f} "
            f"(stationary@5%={self.kpss_is_stationary_5pct})"
        )


def test_unit_root(series: pd.Series, name: str = "series") -> UnitRootResult:
    """
    Run ADF (H0: unit root / non-stationary) and KPSS (H0: stationary) on a
    price series. Using both together avoids relying on either test alone,
    since they have opposite null hypotheses.
    """
    s = series.dropna()

    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")

    # KPSS: statsmodels warns when the stat is outside its p-value table;
    # that's expected for strongly trending series and is not an error.
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")

    return UnitRootResult(
        series_name=name,
        adf_stat=adf_stat,
        adf_pvalue=adf_p,
        adf_is_stationary_5pct=adf_p < 0.05,
        kpss_stat=kpss_stat,
        kpss_pvalue=kpss_p,
        kpss_is_stationary_5pct=kpss_p > 0.05,
    )


def confirm_i1(y: pd.Series, x: pd.Series) -> dict:
    """
    Confirm both legs are I(1): non-stationary in levels, stationary in
    first differences. Returns a dict of UnitRootResult for levels & diffs.
    """
    return {
        "y_level": test_unit_root(y, "Y (level)"),
        "x_level": test_unit_root(x, "X (level)"),
        "y_diff": test_unit_root(y.diff(), "Y (diff)"),
        "x_diff": test_unit_root(x.diff(), "X (diff)"),
    }


# --------------------------------------------------------------------------- #
# Cointegration tests
# --------------------------------------------------------------------------- #
@dataclass
class EngleGrangerResult:
    beta: float
    alpha: float
    coint_t_stat: float
    coint_pvalue: float
    is_cointegrated_5pct: bool
    residuals: pd.Series


def engle_granger_test(y: pd.Series, x: pd.Series, significance: float = 0.05) -> EngleGrangerResult:
    """
    Engle-Granger two-step cointegration test:
      Step 1: OLS regress Y on X to get static hedge ratio beta, intercept alpha.
      Step 2: Test residuals (the spread) for stationarity via ADF (using
              statsmodels' `coint`, which applies the correct MacKinnon
              critical values for a residual-based cointegration test,
              rather than the plain ADF table).
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]

    X = add_constant(df["x"])
    model = OLS(df["y"], X).fit()
    alpha, beta = model.params["const"], model.params["x"]
    residuals = model.resid
    residuals.name = "spread"

    t_stat, p_value, _crit = coint(df["y"], df["x"])

    return EngleGrangerResult(
        beta=beta,
        alpha=alpha,
        coint_t_stat=t_stat,
        coint_pvalue=p_value,
        is_cointegrated_5pct=p_value < significance,
        residuals=residuals,
    )


@dataclass
class JohansenResult:
    trace_stats: np.ndarray
    trace_crit_values: np.ndarray  # columns: 90%, 95%, 99%
    eigen_stats: np.ndarray
    eigen_crit_values: np.ndarray
    cointegrating_vectors: np.ndarray
    rank_5pct: int  # number of cointegrating relationships found at 5%

    def summary(self) -> str:
        lines = ["Johansen trace test (H0: rank <= r):"]
        for r in range(len(self.trace_stats)):
            lines.append(
                f"  r<={r}: trace={self.trace_stats[r]:.3f}  "
                f"crit(90/95/99)={self.trace_crit_values[r].tolist()}"
            )
        lines.append(f"Estimated cointegration rank @5%: {self.rank_5pct}")
        return "\n".join(lines)


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """
    Johansen cointegration test on a multi-asset price DataFrame (columns =
    asset price series). Supports >2 assets (basket cointegration), unlike
    Engle-Granger.

    det_order: -1 no deterministic term, 0 constant, 1 constant + trend.
    """
    data = prices.dropna()
    result = coint_johansen(data, det_order, k_ar_diff)

    trace_stats = result.lr1
    trace_crit = result.cvt  # shape (n, 3) -> 90%, 95%, 99%

    rank = 0
    for r in range(len(trace_stats)):
        if trace_stats[r] > trace_crit[r, 1]:  # 95% column
            rank = r + 1

    return JohansenResult(
        trace_stats=trace_stats,
        trace_crit_values=trace_crit,
        eigen_stats=result.lr2,
        eigen_crit_values=result.cvm,
        cointegrating_vectors=result.evec,
        rank_5pct=rank,
    )


# --------------------------------------------------------------------------- #
# Static hedge ratio (OLS benchmark)
# --------------------------------------------------------------------------- #
def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    """Benchmark static hedge ratio via OLS. Returns (alpha, beta, spread)."""
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    X = add_constant(df["x"])
    model = OLS(df["y"], X).fit()
    alpha, beta = model.params["const"], model.params["x"]
    spread = df["y"] - (alpha + beta * df["x"])
    return alpha, beta, spread


# --------------------------------------------------------------------------- #
# Kalman Filter dynamic hedge ratio
# --------------------------------------------------------------------------- #
class KalmanHedgeRatio:
    """
    State-space Kalman filter for a dynamically time-varying hedge ratio,
    following the standard "rolling linear regression" formulation used in
    pairs-trading (Ernie Chan's approach):

        Observation:  y_t = [x_t, 1] * theta_t + eps_t,      eps_t ~ N(0, R)
        State:        theta_t = theta_{t-1} + w_t,           w_t   ~ N(0, Q)
        theta_t = [beta_t, alpha_t]'

    beta_t is the dynamic hedge ratio, alpha_t the dynamic intercept. Q
    controls how fast the hedge ratio is allowed to drift; R is the
    observation noise variance.
    """

    def __init__(self, delta: float = 1e-4, obs_var: float = 1e-3):
        """
        delta   : transition covariance scaling (higher = beta adapts faster,
                  but noisier). Typical range 1e-5 - 1e-3.
        obs_var : observation noise variance R (typical range 1e-4 - 1e-1,
                  scale-dependent on the price series).
        """
        self.delta = delta
        self.obs_var = obs_var
        self.n_dim_state = 2  # [beta, alpha]

        self.theta = np.zeros(self.n_dim_state)  # state mean
        self.P = np.ones((self.n_dim_state, self.n_dim_state))  # state covariance
        self.Q = (delta / (1 - delta)) * np.eye(self.n_dim_state)  # process noise
        self._initialized = False

    def filter(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        Run the Kalman filter forward through time (no look-ahead: at each
        step t, theta_t is estimated using only data up to and including t).

        Returns a DataFrame indexed like y/x with columns
        ['beta', 'alpha', 'spread', 'spread_var'].
        """
        df = pd.concat([y, x], axis=1).dropna()
        df.columns = ["y", "x"]

        n = len(df)
        betas = np.zeros(n)
        alphas = np.zeros(n)
        spreads = np.zeros(n)
        spread_vars = np.zeros(n)

        theta = self.theta.copy()
        P = self.P.copy()
        Q = self.Q
        R = self.obs_var

        for i in range(n):
            H = np.array([df["x"].iloc[i], 1.0])  # observation matrix row

            # --- Predict ---
            P = P + Q  # theta_t|t-1 = theta_{t-1}; random-walk state model

            # --- Update ---
            y_pred = H @ theta
            innovation = df["y"].iloc[i] - y_pred
            S = H @ P @ H.T + R  # innovation variance
            K = (P @ H) / S  # Kalman gain

            theta = theta + K * innovation
            P = P - np.outer(K, H) @ P

            betas[i] = theta[0]
            alphas[i] = theta[1]
            spreads[i] = innovation  # = y - (beta*x + alpha), i.e. the spread
            spread_vars[i] = S

        self.theta, self.P = theta, P
        self._initialized = True

        out = pd.DataFrame(
            {"beta": betas, "alpha": alphas, "spread": spreads, "spread_var": spread_vars},
            index=df.index,
        )
        return out


# --------------------------------------------------------------------------- #
# Mean-reversion diagnostics
# --------------------------------------------------------------------------- #
def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent via the rescaled-range / variance-of-lagged-
    differences method. H < 0.5 => mean-reverting, H = 0.5 => random walk,
    H > 0.5 => trending/persistent.
    """
    s = series.dropna().values
    max_lag = min(max_lag, len(s) // 2)
    lags = range(2, max_lag)

    tau = [np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags]
    tau = np.array(tau)
    tau[tau == 0] = 1e-10  # avoid log(0)

    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0] * 2.0  # slope * 2 = Hurst exponent


@dataclass
class OUHalfLifeResult:
    theta: float          # mean-reversion speed
    mu: float             # long-run mean
    half_life_bars: float # half-life in units of the input series' bar frequency


def ou_half_life(spread: pd.Series) -> OUHalfLifeResult:
    """
    Estimate Ornstein-Uhlenbeck mean-reversion speed theta and half-life via
    the discretized OU process, regressed as:

        dS_t = theta * (mu - S_{t-1}) * dt + sigma * dW_t
             ~=  S_t - S_{t-1} = a + b * S_{t-1} + eps_t     (dt = 1 bar)

    where b = -theta  =>  theta = -b,  mu = -a/b, half_life = ln(2)/theta.
    """
    s = spread.dropna()
    s_lag = s.shift(1).dropna()
    ds = (s - s.shift(1)).dropna()

    common_idx = s_lag.index.intersection(ds.index)
    s_lag = s_lag.loc[common_idx]
    ds = ds.loc[common_idx]

    X = add_constant(s_lag)
    model = OLS(ds, X).fit()
    a, b = model.params.iloc[0], model.params.iloc[1]

    theta = -b
    mu = -a / b if b != 0 else np.nan
    half_life = np.log(2) / theta if theta > 0 else np.inf

    return OUHalfLifeResult(theta=theta, mu=mu, half_life_bars=half_life)
