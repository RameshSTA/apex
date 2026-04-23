"""
Augmented Dickey-Fuller unit tests.
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.models.stationarity import adf_test, ADFResult

class TestADF(unittest.TestCase):
    def test_white_noise_stationary(self):
        np.random.seed(42); y = np.random.normal(0,1,200)
        r = adf_test(y)
        self.assertTrue(r.is_stationary, f"p={r.p_value:.4f}")

    def test_random_walk_non_stationary(self):
        np.random.seed(43); y = np.cumsum(np.random.normal(0,1,200))
        r = adf_test(y)
        self.assertFalse(r.is_stationary, f"p={r.p_value:.4f}")

    def test_first_diff_stationary(self):
        np.random.seed(44); y = np.cumsum(np.random.normal(0,1,300))
        r = adf_test(np.diff(y))
        self.assertTrue(r.is_stationary, f"p={r.p_value:.4f}")

    def test_result_is_dataclass(self):
        y = np.random.normal(0,1,100)
        self.assertIsInstance(adf_test(y), ADFResult)

    def test_fields_finite(self):
        np.random.seed(45); y = np.random.normal(0,1,150)
        r = adf_test(y)
        self.assertTrue(np.isfinite(r.adf_statistic))
        self.assertTrue(0 < r.p_value < 1)

    def test_critical_values_present(self):
        r = adf_test(np.random.normal(0,1,100))
        for lvl in ["1%","5%","10%"]:
            self.assertIn(lvl, r.critical_values)

    def test_p_value_bounded(self):
        np.random.seed(46)
        for _ in range(5):
            y = np.random.normal(0,1,np.random.randint(60,300))
            r = adf_test(y)
            self.assertGreater(r.p_value, 0)
            self.assertLess(r.p_value, 1)

if __name__ == "__main__": unittest.main()
