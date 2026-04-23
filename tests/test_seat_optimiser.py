"""
LP seat optimiser unit tests.
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.optimiser.seat_optimiser import SeatOptimiser, OptimiserResult
from config import LP_DEFAULTS

BASE_KW = dict(capacity=189, lf_target=0.85, ob_rate=0.05,
               yields=LP_DEFAULTS["yields"], min_eco_pct=0.45, min_first_pct=0.05)

class TestSeatOptimiser(unittest.TestCase):
    def setUp(self):
        self.opt = SeatOptimiser(config=LP_DEFAULTS)
        self.r   = self.opt.optimise(160, **BASE_KW)

    def test_lp_success(self):
        self.assertTrue(self.r.lp_success, self.r.lp_message)

    def test_result_is_dataclass(self):
        self.assertIsInstance(self.r, OptimiserResult)

    def test_total_seats_within_cap_plus_ob(self):
        total = self.r.first_seats + self.r.biz_seats + self.r.prem_seats + self.r.eco_seats
        self.assertLessEqual(total, 189 * 1.05 + 1)
        self.assertGreaterEqual(total, 0)

    def test_min_eco_respected(self):
        self.assertGreaterEqual(self.r.eco_seats, int(189*0.45)-1)

    def test_min_first_respected(self):
        self.assertGreaterEqual(self.r.first_seats, int(189*0.05)-1)

    def test_all_allocs_non_negative(self):
        for seats in [self.r.first_seats, self.r.biz_seats,
                      self.r.prem_seats, self.r.eco_seats]:
            self.assertGreaterEqual(seats, 0)

    def test_revenue_positive(self):
        self.assertGreater(self.r.expected_revenue, 0)
        self.assertGreater(self.r.flat_baseline_rev, 0)

    def test_bid_price_positive(self):
        self.assertGreater(self.r.bid_price_eco, 0)

    def test_higher_yield_higher_revenue(self):
        low_y  = LP_DEFAULTS["yields"].copy()
        high_y = {k: v*2 for k,v in low_y.items()}
        r1 = self.opt.optimise(160, **{**BASE_KW, "yields": low_y})
        r2 = self.opt.optimise(160, **{**BASE_KW, "yields": high_y})
        self.assertGreater(r2.expected_revenue, r1.expected_revenue)

    def test_sensitivity_returns_list(self):
        sens = self.opt.sensitivity(160, eco_range=[85,95,105,115], **BASE_KW)
        self.assertEqual(len(sens), 4)
        self.assertIn("expected_revenue", sens[0])

    def test_sensitivity_decreases_with_more_eco(self):
        sens = self.opt.sensitivity(160, eco_range=list(range(85,145,10)), **BASE_KW)
        revs = [s["expected_revenue"] for s in sens]
        self.assertGreaterEqual(revs[0], revs[-1])

    def test_low_demand_feasible(self):
        r = self.opt.optimise(50, **BASE_KW)
        self.assertIsInstance(r, OptimiserResult)

    def test_high_demand_feasible(self):
        r = self.opt.optimise(500, **BASE_KW)
        self.assertIsInstance(r, OptimiserResult)

if __name__ == "__main__": unittest.main()
