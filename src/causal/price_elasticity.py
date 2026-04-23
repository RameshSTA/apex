"""
Price elasticity of demand: OLS and 2SLS IV estimation.

Model
-----
Log-log demand equation:
    ln(Q_t) = α + β·ln(P_t) + γ·X_t + ε_t

    β = price elasticity of demand (expected: −0.5 to −1.5)

Endogeneity problem
-------------------
Yield (price) and pax (quantity) are simultaneously determined.
An airline raises fares when demand is high, so OLS estimates are
biased towards zero (attenuation bias).

Instrumental Variable (IV) approach — 2SLS
-------------------------------------------
Instrument: ln(fuel_index)

Relevance (Stage 1): Fuel price directly drives ticket prices
  via fuel surcharges and cost-pass-through mechanisms.
  Validated by Stage-1 F-statistic (Staiger-Stock rule: F > 10).

Exclusion restriction: Fuel price has no direct effect on demand
  — passengers do not choose to fly less because fuel is expensive;
  they respond to the *fare* which is affected by fuel costs.

Procedure:
  Stage 1: ln(yield) = π₀ + π₁·ln(fuel) + π₂·X + v
  Stage 2: ln(pax)   = α  + β·ln̂(yield) + γ·X  + ε

Hausman test
-----------
H₀: yield is exogenous (OLS is consistent).
Reject → prefer IV estimate.
  χ² = (β_OLS − β_IV)² / |Var(β_IV) − Var(β_OLS)|
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger

log = get_logger(__name__)

CONTROLS = [
    "school_holiday", "event_multiplier", "covid_factor",
    "competition_index", "week_sin", "week_cos",
]


@dataclass
class ElasticityResult:
    route:               str
    n:                   int
    # OLS
    ols_elasticity:      float
    ols_se:              float
    ols_t:               float
    ols_p:               float
    ols_r2:              float
    # IV (2SLS)
    iv_elasticity:       float
    iv_se:               float
    iv_t:                float
    iv_p:                float
    stage1_fstat:        float        # instrument strength
    # Hausman endogeneity test
    hausman_stat:        float
    hausman_p:           float
    endogenous:          bool         # True → prefer IV
    # Preferred estimate
    preferred_elasticity: float

    def __str__(self) -> str:
        pref = "IV" if self.endogenous else "OLS"
        return (
            f"[{self.route}] {pref} elasticity={self.preferred_elasticity:.4f}  "
            f"OLS={self.ols_elasticity:.4f}  IV={self.iv_elasticity:.4f}  "
            f"Stage1-F={self.stage1_fstat:.1f}  Hausman-p={self.hausman_p:.3f}"
        )


def estimate_elasticity(df_route: pd.DataFrame, route: str) -> ElasticityResult:
    """
    Estimate price elasticity of demand via OLS and 2SLS IV.

    Parameters
    ----------
    df_route : pd.DataFrame
        Single-route data.  Should be post-COVID (≥2022) to avoid
        structural-break confounding of the elasticity estimate.
    route : str
        Route identifier for logging.

    Returns
    -------
    ElasticityResult dataclass.
    """
    d = df_route.copy()
    d = d[d["pax"] > 0].copy()

    # Log transforms
    d["ln_pax"]   = np.log(d["pax"])
    d["ln_yield"] = np.log(d["yield_aud"])
    d["ln_fuel"]  = np.log(d["fuel_index"])

    # Cyclical calendar if not already present
    if "week_sin" not in d.columns:
        d["week_sin"] = np.sin(2 * np.pi * d["week_of_year"] / 52)
        d["week_cos"] = np.cos(2 * np.pi * d["week_of_year"] / 52)

    ctrl = [c for c in CONTROLS if c in d.columns]
    ctrl_arr = [d[c].values for c in ctrl]
    y   = d["ln_pax"].values
    n   = len(d)

    # ── OLS ──────────────────────────────────────────────────────────
    X_ols = np.column_stack([np.ones(n), d["ln_yield"].values, *ctrl_arr])
    b_ols, se_ols, t_ols, p_ols, r2_ols = _ols_full(X_ols, y)

    # ── 2SLS IV ──────────────────────────────────────────────────────
    X_s1 = np.column_stack([np.ones(n), d["ln_fuel"].values, *ctrl_arr])
    try:
        b_s1, _, _, _ = np.linalg.lstsq(X_s1, d["ln_yield"].values, rcond=None)
        ln_yield_hat  = X_s1 @ b_s1
        # Stage-1 F-stat
        resid_s1  = d["ln_yield"].values - ln_yield_hat
        resid_r   = d["ln_yield"].values - d["ln_yield"].values.mean()
        k1        = X_s1.shape[1]
        f_stat    = ((np.sum(resid_r**2) - np.sum(resid_s1**2)) / 1) / \
                    (np.sum(resid_s1**2) / (n - k1))
        # Stage-2
        X_s2 = np.column_stack([np.ones(n), ln_yield_hat, *ctrl_arr])
        b_iv, se_iv, t_iv, p_iv, _ = _ols_full(X_s2, y)
        # Hausman test
        hausman_stat = (b_ols[1] - b_iv[1]) ** 2 / abs(se_iv[1] ** 2 - se_ols[1] ** 2 + 1e-10)
        hausman_p    = float(1 - stats.chi2.cdf(hausman_stat, df=1))
        endogenous   = hausman_p < 0.05
    except np.linalg.LinAlgError:
        b_iv = b_ols; se_iv = se_ols; t_iv = [0]*len(b_ols); p_iv = [1]*len(b_ols)
        f_stat = 0.0; hausman_stat = 0.0; hausman_p = 1.0; endogenous = False

    preferred = float(b_iv[1]) if endogenous else float(b_ols[1])

    result = ElasticityResult(
        route=route, n=n,
        ols_elasticity=round(float(b_ols[1]), 4),
        ols_se=round(float(se_ols[1]), 4),
        ols_t=round(float(t_ols[1]), 3),
        ols_p=round(float(p_ols[1]), 4),
        ols_r2=round(float(r2_ols), 4),
        iv_elasticity=round(float(b_iv[1]), 4),
        iv_se=round(float(se_iv[1]), 4),
        iv_t=round(float(t_iv[1]), 3),
        iv_p=round(float(p_iv[1]), 4),
        stage1_fstat=round(float(f_stat), 2),
        hausman_stat=round(float(hausman_stat), 4),
        hausman_p=round(float(hausman_p), 4),
        endogenous=bool(endogenous),
        preferred_elasticity=round(preferred, 4),
    )
    log.info(str(result))
    return result


# ── Helpers ──────────────────────────────────────────────────────────

def _ols_full(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2    = np.sum(resid ** 2) / (n - k)
    var   = s2 * np.linalg.pinv(X.T @ X)
    se    = np.sqrt(np.diag(var))
    t     = beta / np.maximum(se, 1e-10)
    p     = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    r2    = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return beta, se, t, p, float(r2)
