"""
Single source of truth for all project configuration.
Change values here — never hard-code them in model files.
"""

from pathlib import Path

# ── Project root (resolves relative to this file) ──────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Directory layout ───────────────────────────────────────────────
PATHS = {
    "raw_data":   ROOT / "data" / "raw",
    "processed":  ROOT / "data" / "processed",
    "results":    ROOT / "results",
    "tables":     ROOT / "results" / "tables",
    "plots":      ROOT / "results" / "plots",
    "reports":    ROOT / "results" / "reports",
    "outputs":    ROOT / "outputs",
}

# ── Routes ─────────────────────────────────────────────────────────
ROUTES = {
    "SYD-MEL": {
        "desc": "Sydney–Melbourne",
        "base_pax_weekly": 95000,
        "avg_yield": 185,
        "distance_km": 713,
        "competitors": ["Virgin", "Rex"],
        "comp_capacity_share": 0.42,
        "business_mix": 0.22,
        "avg_load_factor": 0.845,
        "trend_pa": 0.031,
    },
    "SYD-BNE": {
        "desc": "Sydney–Brisbane",
        "base_pax_weekly": 48000,
        "avg_yield": 195,
        "distance_km": 924,
        "competitors": ["Virgin"],
        "comp_capacity_share": 0.38,
        "business_mix": 0.19,
        "avg_load_factor": 0.828,
        "trend_pa": 0.028,
    },
    "MEL-PER": {
        "desc": "Melbourne–Perth",
        "base_pax_weekly": 26000,
        "avg_yield": 310,
        "distance_km": 2699,
        "competitors": ["Virgin"],
        "comp_capacity_share": 0.35,
        "business_mix": 0.28,
        "avg_load_factor": 0.821,
        "trend_pa": 0.024,
    },
    "SYD-ADL": {
        "desc": "Sydney–Adelaide",
        "base_pax_weekly": 22000,
        "avg_yield": 215,
        "distance_km": 1165,
        "competitors": ["Virgin", "Rex"],
        "comp_capacity_share": 0.39,
        "business_mix": 0.18,
        "avg_load_factor": 0.812,
        "trend_pa": 0.021,
    },
    "BNE-PER": {
        "desc": "Brisbane–Perth",
        "base_pax_weekly": 14000,
        "avg_yield": 340,
        "distance_km": 3606,
        "competitors": ["Virgin"],
        "comp_capacity_share": 0.31,
        "business_mix": 0.31,
        "avg_load_factor": 0.808,
        "trend_pa": 0.019,
    },
    "SYD-CBR": {
        "desc": "Sydney–Canberra",
        "base_pax_weekly": 11000,
        "avg_yield": 165,
        "distance_km": 248,
        "competitors": ["Rex"],
        "comp_capacity_share": 0.28,
        "business_mix": 0.41,
        "avg_load_factor": 0.798,
        "trend_pa": 0.018,
    },
    "MEL-ADL": {
        "desc": "Melbourne–Adelaide",
        "base_pax_weekly": 17000,
        "avg_yield": 175,
        "distance_km": 653,
        "competitors": ["Virgin", "Rex"],
        "comp_capacity_share": 0.41,
        "business_mix": 0.16,
        "avg_load_factor": 0.805,
        "trend_pa": 0.022,
    },
    "SYD-PER": {
        "desc": "Sydney–Perth",
        "base_pax_weekly": 18000,
        "avg_yield": 295,
        "distance_km": 3278,
        "competitors": ["Virgin"],
        "comp_capacity_share": 0.36,
        "business_mix": 0.26,
        "avg_load_factor": 0.830,
        "trend_pa": 0.025,
    },
    "MEL-BNE": {
        "desc": "Melbourne–Brisbane",
        "base_pax_weekly": 35000,
        "avg_yield": 205,
        "distance_km": 1748,
        "competitors": ["Virgin"],
        "comp_capacity_share": 0.39,
        "business_mix": 0.21,
        "avg_load_factor": 0.835,
        "trend_pa": 0.027,
    },
    "ADL-PER": {
        "desc": "Adelaide–Perth",
        "base_pax_weekly": 8500,
        "avg_yield": 270,
        "distance_km": 2130,
        "competitors": ["Rex"],
        "comp_capacity_share": 0.25,
        "business_mix": 0.24,
        "avg_load_factor": 0.792,
        "trend_pa": 0.016,
    },
}

# ── Macro time-series (real public data) ──────────────────────────
RBA_CASH_RATE = {2019: 0.015, 2020: 0.0025, 2021: 0.001,
                  2022: 0.030,  2023: 0.043,  2024: 0.043}

CPI_ANNUAL = {2019: 0.017, 2020: 0.009, 2021: 0.032,
               2022: 0.072, 2023: 0.055, 2024: 0.038}

FUEL_INDEX = {2019: 100, 2020: 62, 2021: 78,
               2022: 142, 2023: 118, 2024: 108}

# ── COVID shock parameters (calibrated to BITRE data) ─────────────
import pandas as pd
COVID = {
    "shock_start":          pd.Timestamp("2020-03-23"),
    "shock_bottom":         pd.Timestamp("2020-05-01"),
    "recovery_start":       pd.Timestamp("2020-10-01"),
    "state_borders_reopen": pd.Timestamp("2021-11-01"),
    "full_recovery":        pd.Timestamp("2022-09-01"),
    "bottom_factor":        0.05,
}

# ── Dataset parameters ─────────────────────────────────────────────
DATA_START = "2019-01-07"
DATA_END   = "2024-12-30"
FREQ       = "W-MON"          # Weekly, starting Monday
SEASON_M   = 52               # Seasonal period (weeks)

# ── Model hyperparameters ──────────────────────────────────────────
GB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "min_samples_leaf": 5,
    "random_state":    42,
}

CV_FOLDS     = 3
CV_TEST_SIZE = 8      # weeks per fold
TEST_WEEKS   = 26     # held-out test set size
N_BOOTSTRAP  = 300    # bootstrap CI samples

# ── LP optimiser defaults ──────────────────────────────────────────
LP_DEFAULTS = {
    "capacity":    189,
    "lf_target":   0.85,
    "ob_rate":     0.05,
    "min_eco_pct": 0.45,
    "min_first_pct": 0.05,
    "yields": {"first": 4200, "biz": 1850, "prem": 680, "eco": 310},
}

# ── DiD natural experiment ─────────────────────────────────────────
DID_CONFIG = {
    "treatment_date":  "2024-07-01",   # Rex administration
    "treatment_routes": ["SYD-ADL", "MEL-ADL"],
    "control_routes":   ["SYD-MEL", "MEL-BNE"],
    "pre_window_weeks":  52,
    "post_window_weeks": 26,
    "outcomes": ["yield_aud", "pax"],
}

# ── Feature list ───────────────────────────────────────────────────
FEATURE_COLS = [
    "week_sin", "week_cos", "month_sin", "month_cos",
    "is_summer", "is_winter", "year_idx",
    "school_holiday", "public_holiday", "event_multiplier",
    "covid_factor", "competition_index", "rba_cash_rate",
    "fuel_index", "cpi_index", "business_mix",
    "pax_lag4", "pax_lag8", "pax_lag13", "pax_lag52",
    "pax_yoy_growth", "rolling_mean_4w", "rolling_std_4w",
    "rolling_mean_13w", "pax_vs_rolling", "yield_lag4",
    "comp_x_biz", "rate_x_biz", "fuel_x_yield", "covid_x_school",
]

FEATURE_NAMES = [
    "Week (sin)", "Week (cos)", "Month (sin)", "Month (cos)",
    "Summer flag", "Winter flag", "Year trend",
    "School holiday", "Public holiday", "Event multiplier",
    "COVID factor", "Competition index", "RBA cash rate",
    "Fuel index", "CPI index", "Business mix",
    "Pax lag 4w", "Pax lag 8w", "Pax lag 13w", "Pax lag 52w (YoY)",
    "YoY growth rate", "Rolling mean 4w", "Rolling std 4w",
    "Rolling mean 13w", "Pax vs rolling avg", "Yield lag 4w",
    "Competition×Biz mix", "Rate×Biz mix", "Fuel×Yield", "COVID×School",
]
