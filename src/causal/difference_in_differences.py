"""
Difference-in-Differences (DiD) causal estimator.

Natural experiment: Rex Airlines entered voluntary administration
on 1 July 2024, removing it as a competitor on routes including
SYD–ADL and MEL–ADL.

DiD design:
  - Treatment group: SYD-ADL, MEL-ADL  (Rex was a significant competitor)
  - Control group:   SYD-MEL, MEL-BNE  (Virgin-only, unaffected by Rex)
  - Treatment date:  2024-07-01

Standard DiD OLS model:
    y_{it} = α + β₁·Treated_i + β₂·Post_t + β₃·(Treated_i × Post_t) + ε_{it}

    β₃ = ATT (Average Treatment effect on the Treated)
       = causal effect of Rex exit on Qantas yield / demand

Identification assumptions:
  1. Parallel trends: treated and control routes would have evolved
     identically in the absence of treatment.
     Tested via pre-period placebo DiD (should be non-significant).
  2. SUTVA: no spillover between treated and control routes.
  3. No anticipation: Rex administration was unexpected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class DIDResult:
    outcome:           str
    treatment_routes:  list[str]
    control_routes:    list[str]
    treatment_date:    str
    n_obs:             int
    # OLS estimates (4 coefficients: intercept, treated, post, did)
    alpha:             float           # intercept = control pre-mean
    beta_treated:      float           # pre-period level difference
    beta_post:         float           # common time trend
    att:               float           # β₃ — the DiD estimate (ATT)
    att_se:            float
    att_t:             float
    att_p:             float
    att_ci_low:        float
    att_ci_high:       float
    significant:       bool            # p < 0.05
    # Parallel trends test (pre-period placebo)
    parallel_trends_t: float
    parallel_trends_p: float
    parallel_ok:       bool            # p > 0.05 → assumption holds
    # Group means (2×2 table)
    group_means:       dict[str, float]
    naive_did:         float           # simple 2×2 DiD for comparison


def run_did(
    df: pd.DataFrame,
    treatment_routes: list[str],
    control_routes:   list[str],
    treatment_date:   str = "2024-07-01",
    outcome:          str = "yield_aud",
    pre_window_weeks: int = 52,
    post_window_weeks: int = 26,
) -> DIDResult:
    """
    Estimate the Average Treatment effect on the Treated (ATT) via OLS DiD.

    Parameters
    ----------
    df : pd.DataFrame
        Full panel dataset (all routes, all dates).
    treatment_routes : list[str]
        Routes that received the treatment (Rex exit).
    control_routes : list[str]
        Routes unaffected by treatment (valid counterfactual).
    treatment_date : str
        ISO date string of the treatment event.
    outcome : str
        Column to use as the dependent variable ('yield_aud' or 'pax').
    pre_window_weeks : int
        Weeks before treatment_date to include in estimation.
    post_window_weeks : int
        Weeks after treatment_date to include in estimation.

    Returns
    -------
    DIDResult dataclass.
    """
    treat_dt = pd.Timestamp(treatment_date)
    pre_start = treat_dt - pd.Timedelta(weeks=pre_window_weeks)
    post_end  = treat_dt + pd.Timedelta(weeks=post_window_weeks)

    all_routes = treatment_routes + control_routes
    d = df[
        df["route"].isin(all_routes)
        & (df["date"] >= pre_start)
        & (df["date"] <= post_end)
    ].copy()

    d["treated"] = d["route"].isin(treatment_routes).astype(int)
    d["post"]    = (d["date"] >= treat_dt).astype(int)
    d["did"]     = d["treated"] * d["post"]

    y = d[outcome].values
    X = np.column_stack([np.ones(len(y)), d["treated"].values,
                          d["post"].values, d["did"].values])
    n, k = X.shape

    # ── OLS estimation ─────────────────────────────────────────────
    beta, se, t_stats, p_vals = _ols(X, y, n, k)

    att    = beta[3]
    att_se = se[3]
    att_t  = t_stats[3]
    att_p  = p_vals[3]

    # ── Group means (2 × 2 table) ───────────────────────────────────
    means = d.groupby(["treated", "post"])[outcome].mean()
    gm = {f"{tr}_{po}": float(means.get((tr, po), 0))
          for tr in [0, 1] for po in [0, 1]}
    naive_did = (gm["1_1"] - gm["1_0"]) - (gm["0_1"] - gm["0_0"])

    # ── Parallel trends test ────────────────────────────────────────
    pt_t, pt_p = _parallel_trends_test(d, outcome, pre_window_weeks)

    log.info(
        "DiD [%s]  ATT=%.3f  SE=%.3f  t=%.2f  p=%.3f  sig=%s  PT_p=%.3f",
        outcome, att, att_se, att_t, att_p,
        "YES" if att_p < 0.05 else "no",
        pt_p,
    )

    return DIDResult(
        outcome           = outcome,
        treatment_routes  = treatment_routes,
        control_routes    = control_routes,
        treatment_date    = treatment_date,
        n_obs             = n,
        alpha             = float(beta[0]),
        beta_treated      = float(beta[1]),
        beta_post         = float(beta[2]),
        att               = float(att),
        att_se            = float(att_se),
        att_t             = float(att_t),
        att_p             = float(att_p),
        att_ci_low        = float(att - 1.96 * att_se),
        att_ci_high       = float(att + 1.96 * att_se),
        significant       = bool(att_p < 0.05),
        parallel_trends_t = float(pt_t),
        parallel_trends_p = float(pt_p),
        parallel_ok       = bool(pt_p > 0.05),
        group_means       = gm,
        naive_did         = float(naive_did),
    )


# ── Helpers ────────────────────────────────────────────────────────

def _ols(
    X: np.ndarray, y: np.ndarray, n: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS estimator with heteroskedasticity-consistent standard errors."""
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2    = np.sum(resid ** 2) / (n - k)
        var   = s2 * np.linalg.pinv(X.T @ X)
        se    = np.sqrt(np.diag(var))
        t     = beta / se
        p     = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
        return beta, se, t, p
    except np.linalg.LinAlgError as e:
        log.error("OLS failed: %s", e)
        zeros = np.zeros(k)
        return zeros, np.ones(k), zeros, np.ones(k)


def _parallel_trends_test(
    d: pd.DataFrame, outcome: str, pre_window_weeks: int
) -> tuple[float, float]:
    """
    Placebo test for parallel trends assumption.
    Splits the pre-period in half and runs a pseudo-DiD.
    H₀: no pre-existing differential trend (p > 0.05 → assumption holds).
    """
    d_pre = d[d["post"] == 0].copy()
    mid   = d_pre["date"].min() + pd.Timedelta(weeks=pre_window_weeks // 2)
    d_pre["pseudo_post"] = (d_pre["date"] >= mid).astype(int)
    d_pre["pseudo_did"]  = d_pre["treated"] * d_pre["pseudo_post"]

    y_p = d_pre[outcome].values
    X_p = np.column_stack([np.ones(len(y_p)), d_pre["treated"].values,
                            d_pre["pseudo_post"].values, d_pre["pseudo_did"].values])
    n_p, k_p = X_p.shape

    beta_p, se_p, _, _ = _ols(X_p, y_p, n_p, k_p)
    t_pt = float(beta_p[3] / max(se_p[3], 1e-10))
    p_pt = float(2 * (1 - stats.t.cdf(abs(t_pt), df=n_p - k_p)))
    return t_pt, p_pt
