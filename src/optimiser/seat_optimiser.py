"""
Linear Programme (LP) seat inventory optimiser.

Problem formulation
-------------------
Decision variables:  x = [first_seats, biz_seats, prem_seats, eco_seats]

Objective (maximise expected revenue):
    max  Σ_c  yield[c] × x[c] × demand_prob[c]

Constraints:
    (1)  Σ x[c]              ≤  capacity × (1 + ob_rate)        [total seats]
    (2)  x[first]            ≥  min_first_pct × capacity         [premium commitment]
    (3)  x[eco]              ≥  min_eco_pct × capacity           [access obligation]
    (4)  Σ x[c] × dp[c]     ≥  lf_target × capacity             [load factor floor]
    (5)  x[c]               ≥  0                                 [non-negativity]
    (6)  x[c]               ≤  capacity                          [upper bound]

Solver: scipy.optimize.linprog with HiGHS backend (open-source, fast).

Bid prices
----------
The dual variable (shadow price) of the capacity constraint (1) gives
the bid price — the marginal revenue value of adding one seat to a
fare class. This is the quantity RM systems use to set booking-class
availability controls.

scipy linprog returns dual variables via result.ineqlin.marginals.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import LP_DEFAULTS
from src.utils import get_logger

log = get_logger(__name__)

FARE_CLASSES = ["first", "biz", "prem", "eco"]
DEFAULT_MIX  = np.array([0.06, 0.19, 0.13, 0.62])  # typical Qantas cabin mix
# Class-specific fill rates: P(allocated seat is actually sold)
# First class is harder to fill; Economy easiest.
FILL_RATES   = np.array([0.72, 0.88, 0.85, 0.92])
# Upper-bound demand mix per class (market demand ceiling — can't sell 50% First)
MAX_MIX      = np.array([0.12, 0.30, 0.20, 0.80])


@dataclass
class OptimiserResult:
    # Allocation
    first_seats: int
    biz_seats:   int
    prem_seats:  int
    eco_seats:   int
    first_pct:   float
    biz_pct:     float
    prem_pct:    float
    eco_pct:     float
    # Revenue
    expected_revenue:    float
    flat_baseline_rev:   float
    revenue_uplift_pct:  float
    # Bid price (dual variable of capacity constraint)
    bid_price_eco:       float
    bid_price_biz:       float
    # Solver status
    lp_success:  bool
    lp_message:  str

    def summary(self) -> str:
        return (
            f"First {self.first_pct:.0f}% ({self.first_seats}) | "
            f"Biz {self.biz_pct:.0f}% ({self.biz_seats}) | "
            f"PremEco {self.prem_pct:.0f}% ({self.prem_seats}) | "
            f"Eco {self.eco_pct:.0f}% ({self.eco_seats})  "
            f"→ A${self.expected_revenue:,.0f}  (+{self.revenue_uplift_pct:.1f}%)"
        )


class SeatOptimiser:
    """
    LP-based cabin seat allocation optimiser.

    Usage
    -----
    opt = SeatOptimiser()
    result = opt.optimise(
        forecast_pax = 155,
        capacity     = 189,
        lf_target    = 0.85,
    )
    print(result.summary())
    """

    def __init__(self, config: dict = LP_DEFAULTS):
        self.config = config

    def optimise(
        self,
        forecast_pax: float,
        capacity:     int   = None,
        lf_target:    float = None,
        ob_rate:      float = None,
        yields:       dict  = None,
        min_eco_pct:  float = None,
        min_first_pct: float = None,
    ) -> OptimiserResult:
        """
        Solve the seat allocation LP.

        Parameters
        ----------
        forecast_pax : float
            Expected total passenger demand for the flight.
        capacity : int
            Total aircraft seat capacity.
        lf_target : float
            Minimum load factor constraint (0–1).
        ob_rate : float
            Overbooking allowance as fraction of capacity.
        yields : dict
            Fare yield per seat: {'first': AUD, 'biz': AUD, 'prem': AUD, 'eco': AUD}
        min_eco_pct : float
            Minimum fraction of capacity allocated to economy (access obligation).
        min_first_pct : float
            Minimum fraction for first class (premium product commitment).

        Returns
        -------
        OptimiserResult dataclass with allocations, revenue, and bid prices.
        """
        # ── Parameter resolution (override or use config defaults) ──
        cap       = capacity      or self.config["capacity"]
        lf        = lf_target     or self.config["lf_target"]
        ob        = ob_rate       or self.config["ob_rate"]
        yld       = yields        or self.config["yields"]
        min_eco   = min_eco_pct   or self.config["min_eco_pct"]
        min_first = min_first_pct or self.config["min_first_pct"]

        y_arr = np.array([yld["first"], yld["biz"], yld["prem"], yld["eco"]])

        # ── Demand probabilities per class ──────────────────────────
        # Use class-specific fill rates that reflect real demand constraints:
        #   First class is capacity-constrained by market demand, not just price.
        #   We scale FILL_RATES by overall load-factor signal from forecast.
        #   fill_signal = ratio of forecast demand to a full flight (cap * lf)
        fill_signal = min(1.25, forecast_pax / max(cap * lf, 1))
        dp = np.minimum(FILL_RATES * fill_signal, FILL_RATES)

        # ── LP formulation ──────────────────────────────────────────
        # scipy.linprog minimises → negate revenue objective
        c_obj = -(y_arr * dp)

        # Inequality: A_ub @ x ≤ b_ub
        A_ub = [
            [ 1,  1,  1,  1],                        # (1) total seats ≤ cap*(1+ob)
            [-1,  0,  0,  0],                        # (2) min First allocation
            [ 0,  0,  0, -1],                        # (3) min Eco allocation
            [-dp[0], -dp[1], -dp[2], -dp[3]],        # (4) load factor floor
        ]
        b_ub = [
            cap * (1 + ob),
            -min_first * cap,
            -min_eco   * cap,
            -lf * cap,
        ]
        # Per-class upper bounds: market demand ceiling prevents LP over-allocating First
        bounds = [
            (min_first * cap,   MAX_MIX[0] * cap),  # First:  5%–12%
            (0,                 MAX_MIX[1] * cap),  # Biz:    0%–30%
            (0,                 MAX_MIX[2] * cap),  # Prem:   0%–20%
            (min_eco * cap,     MAX_MIX[3] * cap),  # Eco:   45%–80%
        ]

        result = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds, method="highs",
        )

        # ── Parse solution ──────────────────────────────────────────
        if result.success:
            allocs  = np.maximum(result.x, 0)
            rev     = float(-result.fun)
            # Bid prices from dual variables of capacity constraint
            duals   = result.ineqlin.marginals if hasattr(result, "ineqlin") else np.zeros(4)
            bid_eco = abs(float(duals[0])) if len(duals) > 0 else float(y_arr[3] * dp[3])
            bid_biz = bid_eco * (y_arr[1] / y_arr[3]) if y_arr[3] > 0 else float(y_arr[1] * dp[1])
        else:
            log.warning("LP did not converge (%s), falling back to heuristic", result.message)
            allocs  = np.array(DEFAULT_MIX) * cap
            rev     = float(np.sum(allocs * y_arr * dp))
            bid_eco = float(y_arr[3] * dp[3])
            bid_biz = float(y_arr[1] * dp[1])

        # ── Flat-allocation baseline (for uplift calculation) ────────
        flat_allocs  = np.array(DEFAULT_MIX) * cap
        flat_rev     = float(np.sum(flat_allocs * y_arr * lf))
        uplift       = (rev - flat_rev) / max(flat_rev, 1) * 100

        log.info(
            "LP [cap=%d, lf=%.0f%%]: %s | uplift=+%.1f%% | success=%s",
            cap, lf * 100,
            " / ".join(f"{FARE_CLASSES[i]}={allocs[i]:.0f}" for i in range(4)),
            uplift, result.success,
        )

        return OptimiserResult(
            first_seats = int(round(allocs[0])),
            biz_seats   = int(round(allocs[1])),
            prem_seats  = int(round(allocs[2])),
            eco_seats   = int(round(allocs[3])),
            first_pct   = round(allocs[0] / cap * 100, 1),
            biz_pct     = round(allocs[1] / cap * 100, 1),
            prem_pct    = round(allocs[2] / cap * 100, 1),
            eco_pct     = round(allocs[3] / cap * 100, 1),
            expected_revenue   = round(rev),
            flat_baseline_rev  = round(flat_rev),
            revenue_uplift_pct = round(uplift, 2),
            bid_price_eco      = round(bid_eco, 2),
            bid_price_biz      = round(bid_biz, 2),
            lp_success  = bool(result.success),
            lp_message  = result.message,
        )

    def sensitivity(
        self, forecast_pax: float, eco_range: list[int] | None = None, **kwargs
    ) -> list[dict]:
        """
        Run optimiser across a range of economy seat counts.
        Returns list of dicts for sensitivity analysis / plotting.
        """
        cap = kwargs.get("capacity", self.config["capacity"])
        if eco_range is None:
            eco_range = list(range(int(cap * 0.40), int(cap * 0.80), 5))

        # Remove min_eco_pct from kwargs so we can override it per iteration
        kw = {k: v for k, v in kwargs.items() if k != "min_eco_pct"}

        rows = []
        for eco_fixed in eco_range:
            # Force a minimum eco allocation by adjusting min_eco_pct
            pct = eco_fixed / cap
            res = self.optimise(forecast_pax, min_eco_pct=pct, **kw)
            rows.append({
                "eco_seats":        eco_fixed,
                "eco_pct":          round(pct * 100, 1),
                "expected_revenue": res.expected_revenue,
                "revenue_uplift":   res.revenue_uplift_pct,
                "bid_price_eco":    res.bid_price_eco,
            })
        return rows
