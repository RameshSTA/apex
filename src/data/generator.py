"""
Generates the BITRE-grounded Australian domestic aviation dataset.

All route parameters are calibrated to published BITRE statistics.
COVID shock parameters are fitted to BITRE's documented demand collapse.
Macro variables (RBA rate, CPI, fuel) use real published data series.

Usage:
    from src.data.generator import DatasetGenerator
    gen = DatasetGenerator()
    df = gen.generate()
    gen.save(df)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    ROUTES, COVID, RBA_CASH_RATE, CPI_ANNUAL, FUEL_INDEX,
    DATA_START, DATA_END, FREQ, PATHS,
)
from src.data.calendar import (
    is_school_holiday, is_public_holiday_week, get_event_multiplier,
)
from src.utils import get_logger, save_csv

log = get_logger(__name__)


class DatasetGenerator:
    """
    Generates weekly route-level aviation demand data.

    The data-generating process (DGP) decomposes demand into:
        pax_t = base × trend(t) × seasonal(t) × school_mult(t)
                × pubhol_mult(t) × event_mult(t) × covid_factor(t)
                × rate_effect(t) × fuel_effect(t) × (1 + ε_AR1_t)

    where ε_AR1_t = 0.65 × ε_{t-1} + N(0, 0.02) ensures realistic
    autocorrelation in the booking time-series.
    """

    AR1_RHO        = 0.65
    AR1_SIGMA      = 0.020
    NOISE_SEASONAL = 0.025

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    # ── Public API ──────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate and return the full multi-route dataset."""
        log.info("Generating Australian domestic aviation dataset")
        log.info("Period: %s → %s | %d routes", DATA_START, DATA_END, len(ROUTES))

        frames = []
        for route_key, params in ROUTES.items():
            log.info("  Route %s (%s)", route_key, params["desc"])
            frames.append(self._generate_route(route_key, params))

        df = pd.concat(frames, ignore_index=True).sort_values(["route", "date"])
        df = df.reset_index(drop=True)

        log.info("Dataset complete: %d rows × %d cols", *df.shape)
        log.info("Total pax: %s  |  Revenue: A$%.2fB",
                 f"{df['pax'].sum():,.0f}", df["revenue"].sum() / 1e9)
        return df

    def save(self, df: pd.DataFrame) -> Path:
        """Save to data/raw/australian_aviation.csv."""
        out = PATHS["raw_data"] / "australian_aviation.csv"
        PATHS["raw_data"].mkdir(parents=True, exist_ok=True)
        save_csv(df, out)
        return out

    # ── Private ─────────────────────────────────────────────────────

    def _generate_route(self, route_key: str, params: dict) -> pd.DataFrame:
        dates = pd.date_range(DATA_START, DATA_END, freq=FREQ)
        start_dt = dates[0]
        records = []
        noise_ar = 0.0          # initialise AR(1) state

        cumulative_cpi = 1.0

        for dt in dates:
            year = dt.year
            week = int(dt.isocalendar()[1])

            # ── Trend
            years_elapsed = (dt - start_dt).days / 365.25
            trend = (1 + params["trend_pa"]) ** years_elapsed

            # ── Seasonality (two Fourier harmonics)
            s1 =  0.08 * np.sin(2 * np.pi * week / 52)
            s2 =  0.12 * np.cos(2 * np.pi * week / 52)
            s3 =  0.04 * np.sin(4 * np.pi * week / 52)
            s4 =  0.03 * np.cos(4 * np.pi * week / 52)
            seasonal = 1.0 + s1 + s2 + s3 + s4

            # ── Calendar effects
            school_hol = is_school_holiday(dt)
            pub_hol    = is_public_holiday_week(dt)
            event_mult = get_event_multiplier(dt, route_key)
            leisure_share = 1 - params["business_mix"]
            school_mult   = 1 + school_hol * 0.14 * leisure_share
            pub_hol_mult  = 1 + pub_hol * 0.06

            # ── COVID shock
            covid_factor = self._covid_factor(dt)

            # ── Macro effects
            rba_rate = RBA_CASH_RATE.get(year, 0.035)
            rate_eff = 1 - (rba_rate - 0.015) * params["business_mix"] * 0.8
            fuel_idx = FUEL_INDEX.get(year, 100)
            fuel_eff = 1 - max(0, (fuel_idx - 100) / 100) * 0.08

            # ── Competition index (time-varying: Rex admin Jul 2024)
            comp = params["comp_capacity_share"]
            if year == 2024 and "Rex" in params["competitors"]:
                comp *= 0.65       # Rex entered administration
            if year == 2023 and leisure_share > 0.7:
                comp *= 1.08       # Bonza entry (Feb 2023)

            # ── CPI compounding
            cumulative_cpi = 1.0
            for y in range(2019, year + 1):
                cumulative_cpi *= 1 + CPI_ANNUAL.get(y, 0.025)

            # ── Compose demand
            base_signal = (params["base_pax_weekly"] * trend * seasonal
                           * school_mult * pub_hol_mult * event_mult
                           * covid_factor * rate_eff * fuel_eff)

            # ── AR(1) noise
            noise_ar = self.AR1_RHO * noise_ar + np.random.normal(0, self.AR1_SIGMA)
            pax = max(50, round(base_signal * (1 + noise_ar)))

            # ── Yield
            base_yield  = params["avg_yield"] * cumulative_cpi
            load_sens   = pax / (params["base_pax_weekly"] * trend * seasonal + 1) - 1
            yield_val   = base_yield * (1 + load_sens * 0.35)
            yield_val  *= 1 + max(0, (fuel_idx - 100) / 100) * 0.12
            yield_val  *= 1 - comp * 0.08
            yield_val   = max(80.0, round(yield_val, 2))

            # ── Load factor
            lf = np.clip(params["avg_load_factor"] + noise_ar * 0.15, 0.30, 0.99)
            lf = np.clip(lf * covid_factor, 0.05, 0.99)

            records.append({
                "date":              dt,
                "route":             route_key,
                "route_desc":        params["desc"],
                "pax":               pax,
                "revenue":           round(pax * yield_val),
                "yield_aud":         yield_val,
                "load_factor":       round(float(lf), 4),
                "school_holiday":    school_hol,
                "public_holiday":    pub_hol,
                "event_multiplier":  round(event_mult, 4),
                "covid_factor":      round(float(covid_factor), 4),
                "competition_index": round(comp, 4),
                "rba_cash_rate":     rba_rate,
                "fuel_index":        fuel_idx,
                "cpi_index":         round(cumulative_cpi, 4),
                "business_mix":      params["business_mix"],
                "week_of_year":      week,
                "year":              year,
                "month":             dt.month,
                "quarter":           (dt.month - 1) // 3 + 1,
            })

        return pd.DataFrame(records)

    @staticmethod
    def _covid_factor(dt: pd.Timestamp) -> float:
        """Return demand multiplier for COVID impact (calibrated to BITRE)."""
        c = COVID
        if dt < c["shock_start"]:
            return 1.0
        if dt < c["shock_bottom"]:
            d = (dt - c["shock_start"]).days
            T = (c["shock_bottom"] - c["shock_start"]).days
            return 1.0 - (1.0 - c["bottom_factor"]) * (d / T)
        if dt < c["recovery_start"]:
            return c["bottom_factor"]
        if dt < c["state_borders_reopen"]:
            d = (dt - c["recovery_start"]).days
            T = (c["state_borders_reopen"] - c["recovery_start"]).days
            return c["bottom_factor"] + (0.45 - c["bottom_factor"]) * (d / T)
        if dt < c["full_recovery"]:
            d = (dt - c["state_borders_reopen"]).days
            T = (c["full_recovery"] - c["state_borders_reopen"]).days
            return 0.45 + 0.55 * (d / T)
        return 1.0


# ── CLI entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    gen = DatasetGenerator()
    df  = gen.generate()
    out = gen.save(df)
    print(f"\nSaved → {out}")
    print(df.groupby("route")[["pax", "revenue"]].sum().to_string())
