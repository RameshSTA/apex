"""
Feature engineering unit tests.
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.data.features import FeatureEngineer
from config import FEATURE_COLS

def _make_df(n=200, seed=0):
    np.random.seed(seed)
    dates = pd.date_range("2019-01-07", periods=n, freq="W-MON")
    return pd.DataFrame({
        "date": dates, "route": "TEST",
        "pax": np.random.randint(5000,15000,n),
        "yield_aud": np.random.uniform(150,350,n),
        "load_factor": np.random.uniform(0.7,0.95,n),
        "school_holiday": np.random.randint(0,2,n),
        "public_holiday": np.random.randint(0,2,n),
        "event_multiplier": np.random.uniform(1.0,1.3,n),
        "covid_factor": np.ones(n), "competition_index": np.full(n,0.4),
        "rba_cash_rate": np.full(n,0.03), "fuel_index": np.full(n,100.0),
        "cpi_index": np.ones(n), "business_mix": np.full(n,0.2),
        "week_of_year": dates.isocalendar().week.values.astype(int),
        "year": dates.year.values, "month": dates.month.values,
        "quarter": (dates.month-1)//3+1,
    })

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureEngineer()
        self.df = self.fe.transform(_make_df())

    def test_all_feature_cols_present(self):
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        self.assertEqual(missing, [], f"Missing: {missing}")

    def test_no_nulls_in_feature_cols(self):
        null_cols = [c for c in FEATURE_COLS if self.df[c].isnull().any()]
        self.assertEqual(null_cols, [])

    def test_lag52_drops_correct_rows(self):
        df_out = self.fe.transform(_make_df(200), drop_na=True)
        self.assertEqual(len(df_out), 200-52)

    def test_no_drop_when_false(self):
        df_out = self.fe.transform(_make_df(200), drop_na=False)
        self.assertEqual(len(df_out), 200)

    def test_cyclical_bounded(self):
        for col in ["week_sin","week_cos","month_sin","month_cos"]:
            self.assertTrue(self.df[col].between(-1,1).all(), f"{col} out of [-1,1]")

    def test_season_flags_binary(self):
        for col in ["is_summer","is_winter"]:
            self.assertTrue(set(self.df[col].unique()).issubset({0,1}))

    def test_interactions_not_null(self):
        for col in ["comp_x_biz","rate_x_biz","fuel_x_yield","covid_x_school"]:
            self.assertTrue(self.df[col].notna().all(), f"{col} has nulls")

    def test_yoy_growth_clipped(self):
        self.assertTrue(self.df["pax_yoy_growth"].between(-1,5).all())

    def test_pax_vs_rolling_finite(self):
        self.assertTrue(np.isfinite(self.df["pax_vs_rolling"]).all())

if __name__ == "__main__": unittest.main()
