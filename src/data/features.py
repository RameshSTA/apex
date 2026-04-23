"""
Feature engineering pipeline.

All transformations are centralised here. Model files never
construct features — they call FeatureEngineer.transform().

Feature groups:
  1. Cyclical calendar encodings  (sin/cos — preserves circular structure)
  2. Season binary flags
  3. Trend index
  4. Lag features                 (4w, 8w, 13w, 52w YoY)
  5. Rolling statistics           (mean, std, 4w and 13w windows)
  6. Booking pace proxy
  7. Macro interaction terms      (competition×biz, rate×biz, fuel×yield)
  8. COVID×holiday interaction
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FEATURE_COLS
from src.utils import get_logger

log = get_logger(__name__)


class FeatureEngineer:
    """
    Stateless feature engineering transformer.
    Input: a single-route DataFrame with raw columns.
    Output: same DataFrame with additional feature columns appended.
    """

    def transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Parameters
        ----------
        df : pd.DataFrame
            Raw single-route data, sorted by date.
        drop_na : bool
            Whether to drop rows with NaN lag features (first 52 rows).

        Returns
        -------
        pd.DataFrame with additional feature columns.
        """
        d = df.copy().reset_index(drop=True)

        d = self._cyclical_calendar(d)
        d = self._season_flags(d)
        d = self._trend(d)
        d = self._lag_features(d)
        d = self._rolling_features(d)
        d = self._booking_pace(d)
        d = self._interaction_terms(d)

        if drop_na:
            before = len(d)
            d = d.dropna(subset=["pax_lag52"]).reset_index(drop=True)
            log.debug("Dropped %d NaN-lag rows (lag-52 warmup)", before - len(d))

        missing = [c for c in FEATURE_COLS if c not in d.columns]
        if missing:
            log.warning("Missing expected feature cols: %s", missing)

        return d

    # ── Private steps ───────────────────────────────────────────────

    @staticmethod
    def _cyclical_calendar(d: pd.DataFrame) -> pd.DataFrame:
        """
        Sine/cosine encoding for week-of-year and month.
        Preserves the circular (wrap-around) structure of time:
        week 52 is adjacent to week 1 in feature space.
        """
        d["week_sin"] = np.sin(2 * np.pi * d["week_of_year"] / 52)
        d["week_cos"] = np.cos(2 * np.pi * d["week_of_year"] / 52)
        d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        return d

    @staticmethod
    def _season_flags(d: pd.DataFrame) -> pd.DataFrame:
        d["is_summer"] = ((d["month"] == 12) | (d["month"] <= 2)).astype(int)
        d["is_winter"] = ((d["month"] >= 6) & (d["month"] <= 8)).astype(int)
        return d

    @staticmethod
    def _trend(d: pd.DataFrame) -> pd.DataFrame:
        """Linear year index as trend proxy (2019=0, 2020=1, …)."""
        d["year_idx"] = d["year"] - 2019
        return d

    @staticmethod
    def _lag_features(d: pd.DataFrame) -> pd.DataFrame:
        """
        Autoregressive lags:
          lag4  = same time 4 weeks ago  (short memory)
          lag8  = 8 weeks ago
          lag13 = 13 weeks ago  (quarterly)
          lag52 = 52 weeks ago  (year-over-year — primary seasonal reference)
        """
        d["pax_lag4"]  = d["pax"].shift(4)
        d["pax_lag8"]  = d["pax"].shift(8)
        d["pax_lag13"] = d["pax"].shift(13)
        d["pax_lag52"] = d["pax"].shift(52)
        d["pax_yoy_growth"] = (d["pax"] / (d["pax_lag52"] + 1) - 1).clip(-1, 5)
        d["yield_lag4"] = d["yield_aud"].shift(4)
        return d

    @staticmethod
    def _rolling_features(d: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling window statistics:
          mean_4w  = trailing 4-week average  (recent booking trend)
          std_4w   = trailing 4-week std dev  (volatility signal)
          mean_13w = trailing 13-week average (seasonal baseline)
        """
        d["rolling_mean_4w"]  = d["pax"].rolling(4,  min_periods=1).mean()
        d["rolling_std_4w"]   = d["pax"].rolling(4,  min_periods=1).std().fillna(0)
        d["rolling_mean_13w"] = d["pax"].rolling(13, min_periods=1).mean()
        return d

    @staticmethod
    def _booking_pace(d: pd.DataFrame) -> pd.DataFrame:
        """
        Deviation of current pax from the 4-week rolling mean.
        Values > 0 → demand above recent trend (positive booking pace).
        Values < 0 → demand below recent trend (negative pace).
        """
        d["pax_vs_rolling"] = d["pax"] / (d["rolling_mean_4w"] + 1) - 1
        return d

    @staticmethod
    def _interaction_terms(d: pd.DataFrame) -> pd.DataFrame:
        """
        Domain-knowledge interaction terms:
          comp_x_biz   = competition × business_mix
                         (competitive pressure hits business routes harder)
          rate_x_biz   = RBA rate × business_mix
                         (rate rises suppress business travel)
          fuel_x_yield = fuel_index/100 × yield
                         (fuel passthrough into fare levels)
          covid_x_school = covid_factor × school_holiday
                           (holiday demand recovery interacts with COVID)
        """
        d["comp_x_biz"]     = d["competition_index"] * d["business_mix"]
        d["rate_x_biz"]     = d["rba_cash_rate"]     * d["business_mix"]
        d["fuel_x_yield"]   = d["fuel_index"] / 100  * d["yield_aud"]
        d["covid_x_school"] = d["covid_factor"]       * d["school_holiday"]
        return d
