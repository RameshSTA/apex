"""
Holt-Winters Triple Exponential Smoothing (Additive).

Implements the complete additive Holt-Winters model from scratch
(no statsmodels dependency).

Model equations:
    L_t = α(y_t − S_{t−m}) + (1−α)(L_{t−1} + T_{t−1})     [Level]
    T_t = β(L_t − L_{t−1})  + (1−β) T_{t−1}                [Trend]
    S_t = γ(y_t − L_t)      + (1−γ) S_{t−m}                [Seasonal]
    ŷ_{t+h} = L_t + h·T_t + S_{t+h−m(k+1)}                 [Forecast]

Parameters α, β, γ are grid-searched to minimise RMSE on training data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class HWResult:
    """Container for fitted Holt-Winters state."""
    alpha:        float
    beta:         float
    gamma:        float
    fitted:       np.ndarray
    level:        np.ndarray          # L_t series
    trend:        np.ndarray          # T_t series
    seasonal:     np.ndarray          # S_t series (length n + m)
    residuals:    np.ndarray
    rmse:         float
    y_train:      np.ndarray


class HoltWinters:
    """
    Additive Holt-Winters with grid-search parameter optimisation.

    Parameters
    ----------
    season_periods : int
        Seasonal cycle length in observations (52 for weekly data).
    """

    # Grid-search ranges
    ALPHA_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    BETA_GRID  = [0.05, 0.10, 0.15, 0.20]
    GAMMA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    def __init__(self, season_periods: int = 52):
        self.m = season_periods
        self.result_: Optional[HWResult] = None

    def fit(self, y: np.ndarray, verbose: bool = False) -> "HoltWinters":
        """
        Fit model to training series y.
        Performs exhaustive grid search over α, β, γ.
        """
        y = np.asarray(y, dtype=float)
        best_rmse = np.inf
        best_params = (0.3, 0.1, 0.15)

        for alpha in self.ALPHA_GRID:
            for beta in self.BETA_GRID:
                for gamma in self.GAMMA_GRID:
                    try:
                        fitted, *_ = self._smooth(y, alpha, beta, gamma)
                        rmse = np.sqrt(np.mean((y - fitted) ** 2))
                        if np.isfinite(rmse) and rmse < best_rmse:
                            best_rmse = rmse
                            best_params = (alpha, beta, gamma)
                    except Exception:
                        continue

        a, b, g = best_params
        fitted, L, T, S = self._smooth(y, a, b, g)
        residuals = y - fitted

        self.result_ = HWResult(
            alpha=a, beta=b, gamma=g,
            fitted=fitted, level=L, trend=T, seasonal=S,
            residuals=residuals,
            rmse=np.sqrt(np.mean(residuals ** 2)),
            y_train=y,
        )

        if verbose:
            log.info("HW fitted: α=%.2f β=%.2f γ=%.2f  RMSE=%.0f", a, b, g, best_rmse)

        return self

    def forecast(self, h: int) -> np.ndarray:
        """Generate h-step ahead point forecasts from the fitted state."""
        if self.result_ is None:
            raise RuntimeError("Call .fit() before .forecast()")
        r = self.result_
        n = len(r.level)
        preds = []
        for step in range(1, h + 1):
            s_idx = n - self.m + ((step - 1) % self.m)
            s_idx = min(s_idx, len(r.seasonal) - 1)
            preds.append(r.level[-1] + step * r.trend[-1] + r.seasonal[s_idx])
        return np.array(preds)

    def get_components(self) -> dict:
        """Return the decomposed level, trend, seasonal, residual components."""
        if self.result_ is None:
            raise RuntimeError("Call .fit() first")
        r = self.result_
        n = len(r.y_train)
        return {
            "level":    r.level,
            "trend":    r.trend,
            "seasonal": np.array([r.seasonal[max(0, i - self.m)] for i in range(n)]),
            "residual": r.residuals,
        }

    # ── Internal smoother ──────────────────────────────────────────

    def _smooth(
        self, y: np.ndarray, alpha: float, beta: float, gamma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Core triple-exponential smoothing recursion."""
        n = len(y)
        m = self.m
        L = np.zeros(n)
        T = np.zeros(n)
        S = np.zeros(n + m)

        # ── Initialisation
        if n >= 2 * m:
            L[0] = np.mean(y[:m])
            T[0] = (np.mean(y[m : 2 * m]) - np.mean(y[:m])) / m
        else:
            L[0] = y[0]
            T[0] = (y[-1] - y[0]) / max(n - 1, 1)

        for i in range(m):
            S[i] = y[i] - L[0] if n >= m else 0.0

        fitted = np.zeros(n)
        fitted[0] = L[0] + T[0] + S[0]

        # ── Recursion
        for t in range(1, n):
            s_prev = S[t - m] if t >= m else 0.0
            L[t] = alpha * (y[t] - s_prev) + (1 - alpha) * (L[t - 1] + T[t - 1])
            T[t] = beta  * (L[t] - L[t - 1]) + (1 - beta)  * T[t - 1]
            S[t] = gamma * (y[t] - L[t])      + (1 - gamma) * s_prev
            fitted[t] = L[t] + T[t] + (S[t - m] if t >= m else 0.0)

        return fitted, L, T, S
