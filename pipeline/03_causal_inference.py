"""
Step 3 — Causal inference analysis.
  (a) Difference-in-Differences: Rex administration effect
  (b) Price elasticity: OLS + 2SLS IV per route

Usage:
    python pipeline/03_causal_inference.py

Output:
    results/tables/did_results.json
    results/tables/elasticity.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import PATHS, DID_CONFIG
from src.causal.difference_in_differences import run_did
from src.causal.price_elasticity import estimate_elasticity
from src.utils import get_logger, load_csv, save_csv, save_json, ensure_dirs

log = get_logger(__name__)


def main():
    """Run DiD (Rex administration) and IV price-elasticity analyses for all routes."""
    log.info("=" * 60)
    log.info("  Step 3 — Causal Inference")
    log.info("=" * 60)

    ensure_dirs(PATHS["tables"])
    df = load_csv(PATHS["raw_data"] / "australian_aviation.csv", parse_dates=["date"])

    # ── (a) DiD: Rex administration ──────────────────────────────────
    log.info("Running DiD analysis — Rex administration (Jul 2024)")
    did_summary = {}

    for outcome in DID_CONFIG["outcomes"]:
        result = run_did(
            df,
            treatment_routes  = DID_CONFIG["treatment_routes"],
            control_routes    = DID_CONFIG["control_routes"],
            treatment_date    = DID_CONFIG["treatment_date"],
            outcome           = outcome,
            pre_window_weeks  = DID_CONFIG["pre_window_weeks"],
            post_window_weeks = DID_CONFIG["post_window_weeks"],
        )
        did_summary[outcome] = {
            "att":               result.att,
            "att_se":            result.att_se,
            "att_t":             result.att_t,
            "att_p":             result.att_p,
            "att_ci_low":        result.att_ci_low,
            "att_ci_high":       result.att_ci_high,
            "significant":       result.significant,
            "parallel_trends_p": result.parallel_trends_p,
            "parallel_ok":       result.parallel_ok,
            "n_obs":             result.n_obs,
            "naive_did":         result.naive_did,
            "group_means":       result.group_means,
        }

    save_json(did_summary, PATHS["tables"] / "did_results.json")
    log.info(
        "DiD yield ATT = %.2f (p=%.3f) | parallel trends OK: %s",
        did_summary["yield_aud"]["att"],
        did_summary["yield_aud"]["att_p"],
        did_summary["yield_aud"]["parallel_ok"],
    )

    # ── (b) Price elasticity per route ───────────────────────────────
    log.info("Running price elasticity (OLS + 2SLS IV) per route")
    elas_rows = []

    for route in sorted(df["route"].unique()):
        # Use post-COVID data only to avoid structural break confounding
        df_r = df[(df["route"] == route) & (df["date"] >= "2022-01-01")].copy()
        if len(df_r) < 60:
            log.warning("  %s: too few post-COVID obs (%d), skipping", route, len(df_r))
            continue

        result = estimate_elasticity(df_r, route)
        elas_rows.append({
            "route":                result.route,
            "n":                    result.n,
            "ols_elasticity":       result.ols_elasticity,
            "ols_se":               result.ols_se,
            "ols_t":                result.ols_t,
            "ols_p":                result.ols_p,
            "ols_r2":               result.ols_r2,
            "iv_elasticity":        result.iv_elasticity,
            "iv_se":                result.iv_se,
            "iv_t":                 result.iv_t,
            "iv_p":                 result.iv_p,
            "stage1_fstat":         result.stage1_fstat,
            "hausman_stat":         result.hausman_stat,
            "hausman_p":            result.hausman_p,
            "endogenous":           result.endogenous,
            "preferred_elasticity": result.preferred_elasticity,
        })

    save_csv(pd.DataFrame(elas_rows), PATHS["tables"] / "elasticity.csv")

    log.info("\n%-12s %12s %12s %10s %10s",
             "Route", "OLS elas.", "IV elas.", "Stage1-F", "Preferred")
    log.info("-" * 58)
    for row in elas_rows:
        log.info("%-12s %12.4f %12.4f %10.1f %10.4f",
                 row["route"], row["ols_elasticity"],
                 row["iv_elasticity"], row["stage1_fstat"],
                 row["preferred_elasticity"])


if __name__ == "__main__":
    main()
