"""
Step 2 — Fit hybrid forecaster on all 10 routes.
Saves metrics, forecasts, feature importance to results/tables/.

Usage:
    python pipeline/02_run_forecasts.py

Output:
    results/tables/model_metrics.csv
    results/tables/forecasts.csv
    results/tables/feature_importance.csv
    results/tables/cv_results.csv
    results/tables/decompositions.json
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import PATHS, ROUTES
from src.data.features import FeatureEngineer
from src.models.hybrid_forecaster import HybridForecaster
from src.models.stationarity import adf_test
from src.utils import get_logger, load_csv, save_csv, save_json, ensure_dirs

log = get_logger(__name__)


def main():
    """Fit the hybrid Holt-Winters + GBT forecaster on all routes and save artefacts."""
    log.info("=" * 60)
    log.info("  Step 2 — Hybrid Demand Forecasting (all routes)")
    log.info("=" * 60)

    ensure_dirs(PATHS["tables"])

    raw = PATHS["raw_data"] / "australian_aviation.csv"
    df = load_csv(raw, parse_dates=["date"])

    fe = FeatureEngineer()
    all_metrics, all_forecasts, all_shap, all_cv = [], [], [], []
    decompositions = {}

    for route in sorted(df["route"].unique()):
        log.info("-" * 50)
        log.info("Route: %s", route)

        df_r = df[df["route"] == route].sort_values("date").reset_index(drop=True)
        df_feat = fe.transform(df_r)

        # ── ADF test ────────────────────────────────────────────────
        adf = adf_test(df_r["pax"].values)
        log.info("ADF: stat=%.4f  p=%.4f  stationary=%s",
                 adf.adf_statistic, adf.p_value, adf.is_stationary)

        # ── Hybrid forecaster ────────────────────────────────────────
        model = HybridForecaster()
        model.fit(df_feat, route=route)
        r = model.result_

        # Metrics row
        m = model.to_metrics_dict()
        m["adf_t_stat"]    = round(adf.adf_statistic, 4)
        m["adf_p"]         = round(adf.p_value, 4)
        m["is_stationary"] = adf.is_stationary
        all_metrics.append(m)

        # Forecasts (test set)
        for i, dt_str in enumerate(r.test_dates):
            all_forecasts.append({
                "route":         route,
                "date":          dt_str,
                "period":        "test",
                "actual":        int(r.y_test[i]),
                "hybrid_fc":     int(r.hybrid_forecast[i]),
                "hw_fc":         int(r.hw_forecast[i]),
                "ci_low":        int(max(0, r.ci_low[i])),
                "ci_high":       int(r.ci_high[i]),
            })
        # Last 52 weeks of training fitted
        n_tr = len(r.train_dates)
        for i in range(max(0, n_tr - 52), n_tr):
            all_forecasts.append({
                "route":     route,
                "date":      r.train_dates[i],
                "period":    "train",
                "actual":    int(model._hw.result_.y_train[i]),
                "hybrid_fc": int(r.train_fitted[i]),
                "hw_fc":     int(model._hw.result_.fitted[i]),
                "ci_low":    None, "ci_high": None,
            })

        # SHAP / permutation importance
        for _, row in r.feature_importance.iterrows():
            all_shap.append({
                "route":          route,
                "rank":           _ + 1,
                "feature":        row["feature"],
                "feature_code":   row["feature_code"],
                "importance":     float(row["importance"]),
                "importance_std": float(row["importance_std"]),
            })

        # CV results
        for cv in r.cv_folds:
            all_cv.append({"route": route, **cv})

        # Decomposition (for plotting)
        hw_comps = model._hw.get_components()
        n_train = len(r.train_dates)
        decompositions[route] = {
            "dates":    r.train_dates[-104:],
            "pax":      model._hw.result_.y_train[-104:].tolist(),
            "level":    hw_comps["level"][-104:].tolist(),
            "trend":    hw_comps["trend"][-104:].tolist(),
            "seasonal": hw_comps["seasonal"][-104:].tolist(),
            "residual": hw_comps["residual"][-104:].tolist(),
        }

    # ── Save all results ─────────────────────────────────────────────
    save_csv(pd.DataFrame(all_metrics),   PATHS["tables"] / "model_metrics.csv")
    save_csv(pd.DataFrame(all_forecasts), PATHS["tables"] / "forecasts.csv")
    save_csv(pd.DataFrame(all_shap),      PATHS["tables"] / "feature_importance.csv")
    save_csv(pd.DataFrame(all_cv),        PATHS["tables"] / "cv_results.csv")
    save_json(decompositions,             PATHS["tables"] / "decompositions.json")

    # ── Print summary ─────────────────────────────────────────────────
    mdf = pd.DataFrame(all_metrics)
    log.info("=" * 60)
    log.info("  RESULTS — ALL ROUTES")
    log.info("%-12s %8s %12s %10s", "Route", "HW MAPE", "Hybrid MAPE", "Improve")
    log.info("-" * 50)
    for _, row in mdf.iterrows():
        log.info("%-12s %8.2f%% %11.2f%% %+9.1f%%",
                 row["route"],
                 row["hw_mape"] * 100,
                 row["hybrid_mape"] * 100,
                 row["mape_improvement_pct"])
    log.info("-" * 50)
    log.info("%-12s %8.2f%% %11.2f%% %+9.1f%%", "AVERAGE",
             mdf["hw_mape"].mean() * 100,
             mdf["hybrid_mape"].mean() * 100,
             mdf["mape_improvement_pct"].mean())


if __name__ == "__main__":
    main()
