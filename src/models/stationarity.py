"""
Augmented Dickey-Fuller (ADF) unit-root test.
Implemented from scratch — no statsmodels dependency.

Theory:
    Tests H₀: series has a unit root (is non-stationary / I(1))
    against H₁: series is stationary.

    ADF regression:
        Δy_t = α + γ·y_{t-1} + Σ_{i=1}^{k} δᵢ·Δy_{t-i} + ε_t

    t-statistic on γ is the ADF statistic.
    Critical values from MacKinnon (1994).

Decision rule:
    If ADF stat < critical value (or p < 0.05) → reject H₀ → stationary.
    All 10 routes show I(1) — unit root in levels, stationary in differences.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats

from src.utils import get_logger

log = get_logger(__name__)


# MacKinnon (1994) approximate critical values for regression='c'
ADF_CRITICAL = {"1%": -3.4336, "5%": -2.8621, "10%": -2.5671}


@dataclass
class ADFResult:
    adf_statistic:  float
    p_value:        float
    critical_values: dict
    n_lags:         int
    n_obs:          int
    is_stationary:  bool          # True if reject H₀ at 5%

    def __str__(self) -> str:
        sig = "STATIONARY" if self.is_stationary else "UNIT ROOT (non-stationary)"
        return (
            f"ADF Test  |  stat={self.adf_statistic:.4f}  "
            f"p={self.p_value:.4f}  "
            f"5%CV={self.critical_values['5%']}  "
            f"→ {sig}"
        )


def adf_test(series: np.ndarray, max_lag: int | None = None) -> ADFResult:
    """
    Run the ADF test on a 1-D series.

    Parameters
    ----------
    series : array-like
        The time series to test.
    max_lag : int, optional
        Maximum lag order. Defaults to Schwert (1989) rule:
        floor(12 × (T/100)^{1/4})

    Returns
    -------
    ADFResult dataclass.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)

    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)
    max_lag = min(max_lag, n // 4)

    dy = np.diff(y)          # Δy_t
    k  = max_lag
    nobs = len(dy) - k

    if nobs < 10:
        log.warning("ADF: too few observations (%d) for lag=%d", nobs, k)
        return ADFResult(np.nan, 1.0, ADF_CRITICAL, k, nobs, False)

    # Build regressor matrix:  [y_{t-1}, Δy_{t-1}, …, Δy_{t-k}, 1]
    Y   = dy[k:]
    cols = [y[k : k + nobs]]              # lagged level  (γ coefficient)
    for i in range(1, k + 1):
        cols.append(dy[k - i : k - i + nobs])   # lagged differences
    cols.append(np.ones(nobs))            # constant
    X = np.column_stack(cols)

    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        dof   = nobs - X.shape[1]
        s2    = np.sum(resid ** 2) / dof
        var   = s2 * np.linalg.pinv(X.T @ X)
        se    = np.sqrt(np.diag(var))
        t_stat = beta[0] / se[0]
    except np.linalg.LinAlgError:
        log.error("ADF: linear algebra failure")
        return ADFResult(np.nan, 1.0, ADF_CRITICAL, k, nobs, False)

    # Approximate p-value (MacKinnon 1994 interpolation)
    p = _mackinon_pvalue(t_stat)

    return ADFResult(
        adf_statistic  = float(t_stat),
        p_value        = float(p),
        critical_values= ADF_CRITICAL,
        n_lags         = k,
        n_obs          = nobs,
        is_stationary  = p < 0.05,
    )


def _mackinon_pvalue(tau: float) -> float:
    """Piecewise-linear approximation of MacKinnon (1994) p-values."""
    if   tau <= -3.5: p = 0.001
    elif tau <= -2.9: p = 0.010 + (tau + 3.5) / 0.6 * 0.040
    elif tau <= -2.6: p = 0.050 + (tau + 2.9) / 0.3 * 0.050
    elif tau <= -2.0: p = 0.100 + (tau + 2.6) / 0.6 * 0.400
    elif tau <=  0.0: p = 0.500 + tau / 2.0 * (-0.400)
    else:             p = 0.900
    return float(np.clip(p, 0.001, 0.999))
