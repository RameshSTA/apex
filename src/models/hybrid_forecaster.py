"""
Hybrid Holt-Winters + GradientBoosting demand forecaster.

Architecture
------------
1. Holt-Winters (holt_winters.py) captures level, trend, and seasonality.
2. GradientBoostingRegressor (sklearn) learns the residual:
       ε_t  =  y_t  −  ŷ_t^{HW}
3. Final forecast:
       ŷ_{t+h}^{Hybrid}  =  ŷ_{t+h}^{HW}  +  GB_correction_{t+h}
4. Bootstrap CI: resample training residuals to propagate uncertainty.
5. Walk-forward TimeSeriesSplit CV to measure generalisation.
6. Permutation importance as SHAP proxy (no external shap dependency).

Why hybrid over pure ML?
  Airline demand has strong structural seasonality and trend that tree
  models are poor at extrapolating. The HW layer encodes this structure
  parametrically; the GB layer focuses on residual variation (events,
  macro shocks, competitor moves) where ML excels.
  Result: hybrid consistently outperforms either model alone — consistent
  with AGIFORS 2025 research on parametric+ML hybrid forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_percentage_error as mape,
    mean_squared_error as mse,
    r2_score,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FEATURE_COLS, FEATURE_NAMES, GB_PARAMS, CV_FOLDS, CV_TEST_SIZE, TEST_WEEKS, N_BOOTSTRAP
from src.models.holt_winters import HoltWinters
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class ForecastResult:
    """All outputs from a fitted hybrid forecaster."""
    route:          str
    # Train metrics
    hw_mape:        float
    hybrid_mape:    float
    hybrid_rmse:    float
    hybrid_r2:      float
    mape_improvement_pct: float
    # Cross-validation
    cv_mape_mean:   float
    cv_mape_std:    float
    cv_folds:       list[dict]
    # HW params
    hw_alpha:       float
    hw_beta:        float
    hw_gamma:       float
    # Test set arrays
    y_test:         np.ndarray
    hw_forecast:    np.ndarray
    hybrid_forecast: np.ndarray
    ci_low:         np.ndarray
    ci_high:        np.ndarray
    test_dates:     list[str]
    # Feature importance
    feature_importance: pd.DataFrame
    # Training fitted values
    train_fitted:   np.ndarray
    train_residuals: np.ndarray
    train_dates:    list[str]


class HybridForecaster:
    """
    Hybrid Holt-Winters + GradientBoosting demand forecaster.

    Parameters
    ----------
    gb_params : dict
        GradientBoostingRegressor keyword arguments.
    test_weeks : int
        Number of weeks held out for evaluation.
    n_bootstrap : int
        Bootstrap samples for confidence intervals.
    cv_folds : int
        Walk-forward cross-validation folds.
    cv_test_size : int
        Validation window size per fold (weeks).
    """

    def __init__(
        self,
        gb_params:   dict = GB_PARAMS,
        test_weeks:  int  = TEST_WEEKS,
        n_bootstrap: int  = N_BOOTSTRAP,
        cv_folds:    int  = CV_FOLDS,
        cv_test_size: int = CV_TEST_SIZE,
    ):
        self.gb_params    = gb_params
        self.test_weeks   = test_weeks
        self.n_bootstrap  = n_bootstrap
        self.cv_folds     = cv_folds
        self.cv_test_size = cv_test_size

        self._hw     = HoltWinters()
        self._gb     = None
        self._scaler = StandardScaler()
        self.result_: Optional[ForecastResult] = None

    # ── Public API ───────────────────────────────────────────────────

    def fit(self, df_feat: pd.DataFrame, route: str = "") -> "HybridForecaster":
        """
        Fit the full hybrid pipeline.

        Parameters
        ----------
        df_feat : pd.DataFrame
            Feature-engineered single-route DataFrame (from FeatureEngineer).
        route : str
            Route identifier for logging and result labelling.
        """
        y = df_feat["pax"].values
        n = len(y)
        n_train = n - self.test_weeks

        feat_cols = [c for c in FEATURE_COLS if c in df_feat.columns]
        X_all   = df_feat[feat_cols].values
        X_train = X_all[:n_train]
        X_test  = X_all[n_train:]

        log.info("[%s] Fitting Holt-Winters on %d training weeks...", route, n_train)
        self._hw.fit(y[:n_train], verbose=True)
        hw_result   = self._hw.result_
        hw_fitted   = hw_result.fitted
        hw_residuals = y[:n_train] - hw_fitted

        # HW-only test forecast (baseline)
        hw_test_fc = np.maximum(self._hw.forecast(self.test_weeks), 0)
        hw_mape_score = mape(y[n_train:], hw_test_fc)

        # ── Walk-forward CV ──────────────────────────────────────────
        log.info("[%s] Walk-forward CV (%d folds)...", route, self.cv_folds)
        tscv = TimeSeriesSplit(n_splits=self.cv_folds, test_size=self.cv_test_size)
        cv_records = []

        for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            scaler_f = StandardScaler()
            gb_f = GradientBoostingRegressor(**{**self.gb_params, "n_estimators": 200})
            gb_f.fit(scaler_f.fit_transform(X_train[tr_idx]), hw_residuals[tr_idx])
            gb_val = gb_f.predict(scaler_f.transform(X_train[val_idx]))
            hybrid_val = np.maximum(hw_fitted[val_idx] + gb_val, 0)
            fold_mape = mape(y[:n_train][val_idx], hybrid_val)
            cv_records.append({
                "fold": fold_i + 1,
                "mape": fold_mape,
                "rmse": np.sqrt(mse(y[:n_train][val_idx], hybrid_val)),
                "n_train": len(tr_idx),
                "n_val":   len(val_idx),
            })

        # ── Final production model ───────────────────────────────────
        log.info("[%s] Fitting final GB on full training set...", route)
        self._scaler.fit(X_train)
        self._gb = GradientBoostingRegressor(**self.gb_params)
        self._gb.fit(self._scaler.transform(X_train), hw_residuals)
        self._feat_cols = feat_cols

        # Train fitted (for residual bootstrap)
        gb_train_pred   = self._gb.predict(self._scaler.transform(X_train))
        train_fitted    = np.maximum(hw_fitted + gb_train_pred, 0)
        train_residuals = y[:n_train] - train_fitted

        # ── Test evaluation ──────────────────────────────────────────
        gb_test_corr  = self._gb.predict(self._scaler.transform(X_test))
        hybrid_test   = np.maximum(hw_test_fc + gb_test_corr, 0)
        y_test        = y[n_train:]
        hybrid_mape   = mape(y_test, hybrid_test)
        hybrid_rmse   = np.sqrt(mse(y_test, hybrid_test))
        hybrid_r2     = r2_score(y_test, hybrid_test)
        improvement   = (hw_mape_score - hybrid_mape) / hw_mape_score * 100

        log.info(
            "[%s] HW MAPE=%.2f%%  Hybrid MAPE=%.2f%%  Δ=%+.1f%%  R²=%.4f",
            route, hw_mape_score * 100, hybrid_mape * 100, improvement, hybrid_r2,
        )

        # ── Bootstrap CI ─────────────────────────────────────────────
        ci_low, ci_high = self._bootstrap_ci(hybrid_test, train_residuals)

        # ── Permutation importance ───────────────────────────────────
        perm = permutation_importance(
            self._gb,
            self._scaler.transform(X_train),
            hw_residuals,
            n_repeats=15,
            random_state=42,
            scoring="neg_mean_squared_error",
        )
        feat_names = [
            FEATURE_NAMES[FEATURE_COLS.index(c)] if c in FEATURE_COLS else c
            for c in feat_cols
        ]
        imp_df = pd.DataFrame({
            "feature":        feat_names,
            "feature_code":   feat_cols,
            "importance":     perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        cv_mapes = [r["mape"] for r in cv_records]

        self.result_ = ForecastResult(
            route             = route,
            hw_mape           = hw_mape_score,
            hybrid_mape       = hybrid_mape,
            hybrid_rmse       = hybrid_rmse,
            hybrid_r2         = hybrid_r2,
            mape_improvement_pct = improvement,
            cv_mape_mean      = float(np.mean(cv_mapes)),
            cv_mape_std       = float(np.std(cv_mapes)),
            cv_folds          = cv_records,
            hw_alpha          = hw_result.alpha,
            hw_beta           = hw_result.beta,
            hw_gamma          = hw_result.gamma,
            y_test            = y_test,
            hw_forecast       = hw_test_fc,
            hybrid_forecast   = hybrid_test,
            ci_low            = ci_low,
            ci_high           = ci_high,
            test_dates        = df_feat["date"].iloc[n_train:].dt.strftime("%Y-%m-%d").tolist(),
            feature_importance= imp_df,
            train_fitted      = train_fitted,
            train_residuals   = train_residuals,
            train_dates       = df_feat["date"].iloc[:n_train].dt.strftime("%Y-%m-%d").tolist(),
        )
        return self

    def to_metrics_dict(self) -> dict:
        """Return a flat dict of scalar metrics for CSV export."""
        r = self.result_
        return {
            "route":                r.route,
            "n_train":              len(r.train_fitted),
            "n_test":               len(r.y_test),
            "hw_mape":              round(r.hw_mape, 4),
            "hybrid_mape":          round(r.hybrid_mape, 4),
            "hybrid_rmse":          round(r.hybrid_rmse, 2),
            "hybrid_r2":            round(r.hybrid_r2, 4),
            "mape_improvement_pct": round(r.mape_improvement_pct, 2),
            "cv_mape_mean":         round(r.cv_mape_mean, 4),
            "cv_mape_std":          round(r.cv_mape_std, 4),
            "hw_alpha":             r.hw_alpha,
            "hw_beta":              r.hw_beta,
            "hw_gamma":             r.hw_gamma,
        }

    # ── Private ──────────────────────────────────────────────────────

    def _bootstrap_ci(
        self, point_fc: np.ndarray, residuals: np.ndarray, ci: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = (1 - ci) / 2
        boot = np.array([
            np.maximum(point_fc + np.random.choice(residuals, size=len(point_fc), replace=True), 0)
            for _ in range(self.n_bootstrap)
        ])
        return np.percentile(boot, alpha * 100, axis=0), np.percentile(boot, (1 - alpha) * 100, axis=0)
