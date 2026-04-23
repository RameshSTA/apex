"""
Holt-Winters triple exponential smoothing unit tests.

Run: python -m pytest tests/ -v
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.models.holt_winters import HoltWinters, HWResult

def _seasonal(n=104, seed=0):
    np.random.seed(seed)
    t = np.arange(n)
    return 1000 + 5*t + 200*np.sin(2*np.pi*t/52) + np.random.normal(0,20,n)

class TestHoltWinters(unittest.TestCase):
    def setUp(self):
        self.y = _seasonal()
        self.hw = HoltWinters(season_periods=52).fit(self.y, verbose=False)

    def test_fit_returns_hwresult(self):
        self.assertIsInstance(self.hw.result_, HWResult)

    def test_fitted_shape_matches_input(self):
        self.assertEqual(len(self.hw.result_.fitted), len(self.y))

    def test_params_in_valid_range(self):
        r = self.hw.result_
        self.assertGreater(r.alpha, 0); self.assertLessEqual(r.alpha, 1)
        self.assertGreater(r.beta, 0);  self.assertLessEqual(r.beta,  1)
        self.assertGreater(r.gamma, 0); self.assertLessEqual(r.gamma, 1)

    def test_forecast_correct_length(self):
        for h in [1, 13, 26, 52]:
            fc = self.hw.forecast(h)
            self.assertEqual(len(fc), h)

    def test_forecast_all_finite(self):
        fc = self.hw.forecast(26)
        self.assertTrue(np.all(np.isfinite(fc)))

    def test_residuals_smaller_std_than_original(self):
        self.assertLess(np.std(self.hw.result_.residuals), np.std(self.y))

    def test_get_components_has_keys(self):
        comps = self.hw.get_components()
        for k in ["level", "trend", "seasonal", "residual"]:
            self.assertIn(k, comps)
            self.assertEqual(len(comps[k]), len(self.y))

    def test_raises_before_fit(self):
        with self.assertRaises(RuntimeError):
            HoltWinters().forecast(5)

if __name__ == "__main__": unittest.main()
