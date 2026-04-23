"""
DiD and price elasticity unit tests.
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.causal.difference_in_differences import run_did, DIDResult
from src.causal.price_elasticity import estimate_elasticity, ElasticityResult

def _did_panel(seed=10):
    np.random.seed(seed)
    dates = pd.date_range("2023-07-01", periods=400, freq="W-MON")
    routes = ["SYD-ADL","MEL-ADL","SYD-MEL","MEL-BNE"]
    rows = []
    for d in dates:
        for r in routes:
            treated = r in ["SYD-ADL","MEL-ADL"]
            post    = d >= pd.Timestamp("2024-07-01")
            base    = 220 if treated else 190
            effect  = 15 if (treated and post) else 0
            rows.append({"date":d,"route":r,
                         "yield_aud":base+effect+np.random.normal(0,8),
                         "pax":int(20000+np.random.normal(0,500))})
    return pd.DataFrame(rows)

def _elas_df(seed=11):
    np.random.seed(seed); n=150
    fuel  = np.random.uniform(80,140,n)
    price = 200 + 0.5*fuel + np.random.normal(0,10,n)
    demand = np.exp(8.0 - 0.8*np.log(price) + np.random.normal(0,0.05,n))
    dates  = pd.date_range("2022-01-01",periods=n,freq="W-MON")
    return pd.DataFrame({
        "date":dates,"route":"TEST","pax":demand.astype(int),"yield_aud":price,
        "fuel_index":fuel,"school_holiday":np.zeros(n,int),"public_holiday":np.zeros(n,int),
        "event_multiplier":np.ones(n),"covid_factor":np.ones(n),
        "competition_index":np.full(n,0.4),
        "week_sin":np.sin(2*np.pi*np.arange(n)/52),
        "week_cos":np.cos(2*np.pi*np.arange(n)/52),
        "week_of_year":(np.arange(n)%52)+1,
    })

class TestDiD(unittest.TestCase):
    def setUp(self):
        self.df = _did_panel()
        self.r  = run_did(self.df, ["SYD-ADL","MEL-ADL"], ["SYD-MEL","MEL-BNE"],
                          treatment_date="2024-07-01", outcome="yield_aud")

    def test_returns_dataclass(self):
        self.assertIsInstance(self.r, DIDResult)

    def test_att_positive_with_baked_effect(self):
        self.assertGreater(self.r.att, 5, f"ATT={self.r.att:.2f}")

    def test_att_significant(self):
        self.assertTrue(self.r.significant, f"p={self.r.att_p:.4f}")

    def test_parallel_trends_passes(self):
        self.assertTrue(self.r.parallel_ok, f"PT p={self.r.parallel_trends_p:.4f}")

    def test_ci_contains_att(self):
        self.assertLess(self.r.att_ci_low, self.r.att)
        self.assertGreater(self.r.att_ci_high, self.r.att)

    def test_required_fields(self):
        for f in ["att","att_se","att_p","att_ci_low","att_ci_high",
                  "parallel_trends_p","group_means","n_obs"]:
            self.assertTrue(hasattr(self.r, f), f"Missing: {f}")

    def test_group_means_four_cells(self):
        self.assertEqual(len(self.r.group_means), 4)
        for k in ["0_0","0_1","1_0","1_1"]:
            self.assertIn(k, self.r.group_means)

    def test_n_obs_positive(self):
        self.assertGreater(self.r.n_obs, 0)

    def test_zero_effect_not_significant(self):
        np.random.seed(12)
        dates = pd.date_range("2023-01-01",periods=300,freq="W-MON")
        rows = [{"date":d,"route":r,"yield_aud":200+np.random.normal(0,5),"pax":10000}
                for d in dates for r in ["A","B","C","D"]]
        r = run_did(pd.DataFrame(rows),["A","B"],["C","D"],treatment_date="2024-01-01")
        self.assertLess(abs(r.att), 10)

class TestElasticity(unittest.TestCase):
    def setUp(self):
        self.r = estimate_elasticity(_elas_df(), "TEST")

    def test_returns_dataclass(self):
        self.assertIsInstance(self.r, ElasticityResult)

    def test_ols_negative(self):
        self.assertLess(self.r.ols_elasticity, 0)

    def test_iv_not_more_positive_than_ols(self):
        self.assertLessEqual(self.r.iv_elasticity, self.r.ols_elasticity + 0.5)

    def test_stage1_fstat_strong(self):
        self.assertGreater(self.r.stage1_fstat, 10)

    def test_preferred_finite(self):
        self.assertTrue(np.isfinite(self.r.preferred_elasticity))

    def test_se_positive(self):
        self.assertGreater(self.r.ols_se, 0)
        self.assertGreater(self.r.iv_se, 0)

    def test_r2_bounded(self):
        self.assertGreaterEqual(self.r.ols_r2, 0)
        self.assertLessEqual(self.r.ols_r2, 1)

    def test_required_fields(self):
        for f in ["ols_elasticity","iv_elasticity","stage1_fstat",
                  "hausman_p","endogenous","preferred_elasticity"]:
            self.assertTrue(hasattr(self.r, f), f"Missing: {f}")

if __name__ == "__main__": unittest.main()
