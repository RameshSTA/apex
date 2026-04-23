"""
Step 4 — Run LP seat optimiser for all routes.

Reads the demand forecasts from Step 2, solves the LP for each route,
and writes a results table + sensitivity analysis.

Usage:
    python pipeline/04_seat_optimiser.py

Output:
    results/tables/lp_results.csv
    results/tables/lp_sensitivity.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import PATHS, LP_DEFAULTS
from src.optimiser.seat_optimiser import SeatOptimiser
from src.utils import get_logger, load_csv, save_csv

log = get_logger(__name__)


def main() -> None:
    """Solve the LP seat-inventory optimisation problem for every route."""
    log.info("=" * 55)
    log.info("  Step 4 — LP Seat Optimiser")
    log.info("=" * 55)

    # ── Load forecast data ─────────────────────────────────────────
    fc_path = PATHS["tables"] / "forecasts.csv"
    if not fc_path.exists():
        log.error("forecasts.csv not found — run pipeline/02 first")
        sys.exit(1)

    forecasts = load_csv(fc_path)
    test_fc   = forecasts[forecasts["period"] == "test"].copy()

    # ── Estimated Qantas daily departures per route ───────────────
    # Used to convert weekly pax → per-flight demand for the LP.
    # Qantas market share ~58% of route total (BITRE/ACCC data).
    QANTAS_SHARE = 0.58
    DAILY_FLIGHTS = {
        "SYD-MEL": 14, "SYD-BNE": 9, "MEL-PER": 4, "SYD-ADL": 5,
        "BNE-PER": 3,  "SYD-CBR": 3, "MEL-ADL": 4, "SYD-PER": 3,
        "MEL-BNE": 6,  "ADL-PER": 2,
    }

    # ── Load route-specific yields from dataset ───────────────────
    raw_df = load_csv(PATHS["raw_data"] / "australian_aviation.csv")
    route_avg_yield = raw_df.groupby("route")["yield_aud"].mean().to_dict()
    # Scale class-specific yields proportionally to route average yield
    # Base yields are calibrated for SYD-MEL (avg A$189)
    BASE_YIELD = 189.0

    # ── Run LP per route ───────────────────────────────────────────
    lp_rows = []
    sens_rows = []

    for route in sorted(test_fc["route"].unique()):
        route_fc      = test_fc[test_fc["route"] == route]
        weekly_pax    = float(route_fc["hybrid_fc"].mean())
        flights_week  = DAILY_FLIGHTS.get(route, 5) * 7
        # Per-flight Qantas pax: weekly market total × Qantas share ÷ flights
        per_flight_pax = weekly_pax * QANTAS_SHARE / flights_week

        log.info("  [%s]  weekly_pax=%.0f  qantas_pax/flt=%.0f",
                 route, weekly_pax, per_flight_pax)

        # Scale yields by route average yield (longer/higher-yield routes → higher fares)
        route_yield_scalar = route_avg_yield.get(route, BASE_YIELD) / BASE_YIELD
        scaled_yields = {
            k: round(v * route_yield_scalar)
            for k, v in LP_DEFAULTS["yields"].items()
        }
        opt = SeatOptimiser(config=LP_DEFAULTS)
        result = opt.optimise(
            forecast_pax  = per_flight_pax,
            capacity      = LP_DEFAULTS["capacity"],
            lf_target     = LP_DEFAULTS["lf_target"],
            ob_rate       = LP_DEFAULTS["ob_rate"],
            min_eco_pct   = LP_DEFAULTS["min_eco_pct"],
            min_first_pct = LP_DEFAULTS["min_first_pct"],
            yields        = scaled_yields,
        )

        row = {
            "route":              route,
            "forecast_pax":       round(per_flight_pax),
            "first_seats":        result.first_seats,
            "biz_seats":          result.biz_seats,
            "prem_seats":         result.prem_seats,
            "eco_seats":          result.eco_seats,
            "first_pct":          result.first_pct,
            "biz_pct":            result.biz_pct,
            "prem_pct":           result.prem_pct,
            "eco_pct":            result.eco_pct,
            "expected_revenue":   result.expected_revenue,
            "flat_baseline_rev":  result.flat_baseline_rev,
            "revenue_uplift_pct": result.revenue_uplift_pct,
            "bid_price_eco":      result.bid_price_eco,
            "lp_success":         result.lp_success,
        }
        lp_rows.append(row)

        # Sensitivity: sweep eco_pct allocation
        # Sensitivity: sweep economy allocation
        cap = LP_DEFAULTS["capacity"]
        eco_range = list(range(int(cap * 0.40), int(cap * 0.76), 5))
        sens = opt.sensitivity(
            per_flight_pax,
            eco_range=eco_range,
            capacity=cap,
            lf_target=LP_DEFAULTS["lf_target"],
            ob_rate=LP_DEFAULTS["ob_rate"],
            yields=LP_DEFAULTS["yields"],
        )
        for s in sens:
            s["route"] = route
        sens_rows.extend(sens)

    # ── Save results ───────────────────────────────────────────────
    lp_df   = pd.DataFrame(lp_rows)
    sens_df = pd.DataFrame(sens_rows)

    save_csv(lp_df,   PATHS["tables"] / "lp_results.csv")
    save_csv(sens_df, PATHS["tables"] / "lp_sensitivity.csv")

    # Print summary
    log.info("\n  LP Results Summary:")
    log.info("  %-10s  %8s  %8s  %10s", "Route", "Uplift %", "BidPrice", "Rev(A$)")
    for _, r in lp_df.iterrows():
        log.info("  %-10s  %7.1f%%  A$%6.0f  A$%9.0f",
                 r["route"], r["revenue_uplift_pct"],
                 r["bid_price_eco"], r["expected_revenue"])

    log.info("\n  Avg revenue uplift: +%.2f%%",
             lp_df["revenue_uplift_pct"].mean())
    log.info("  Step 4 complete.")


if __name__ == "__main__":
    main()
