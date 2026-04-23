"""
Step 5 — Build the interactive HTML dashboard.

Reads all result files from results/tables/, inlines the frontend assets
from frontend/ (index.html, style.css, app.js), and writes the self-contained
outputs/apex_dashboard.html file.

The dashboard requires no server — it runs entirely in the browser.

Usage:
    python pipeline/05_build_dashboard.py

Output:
    outputs/apex_dashboard.html
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import PATHS
from src.utils import get_logger, load_csv, load_json, ensure_dirs

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"


# ─────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────

def load_all_results() -> dict:
    """Load every result file into a dict of Python objects."""
    t = PATHS["tables"]
    log.info("Loading result tables from %s", t)

    def safe_csv(name):
        p = t / name
        if p.exists():
            return load_csv(p)
        log.warning("  %s not found — skipping", name)
        return pd.DataFrame()

    def safe_json(name, default=None):
        p = t / name
        if p.exists():
            return load_json(p)
        log.warning("  %s not found — skipping", name)
        return default if default is not None else {}

    return {
        "metrics":      safe_csv("model_metrics.csv"),
        "forecasts":    safe_csv("forecasts.csv"),
        "shap":         safe_csv("feature_importance.csv"),
        "elasticity":   safe_csv("elasticity.csv"),
        "lp":           safe_csv("lp_results.csv"),
        "cv":           safe_csv("cv_results.csv"),
        "did":          safe_json("did_results.json"),
        "monthly_ts":   safe_json("monthly_ts.json", default=[]),
        "syd_weekly":   safe_json("syd_mel_weekly.json", default=[]),
    }


def build_data_js(data: dict) -> str:
    """Serialise all result data as inline JS variable declarations."""

    def df_to_json(df: pd.DataFrame) -> str:
        if df.empty:
            return "[]"
        return df.to_json(orient="records")

    def safe_json_dumps(obj) -> str:
        def _convert(o):
            import numpy as np
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)):    return bool(o)
            if isinstance(o, (np.ndarray,)):  return o.tolist()
            raise TypeError(type(o))
        return json.dumps(obj, default=_convert)

    lines = [
        f"const METRICS     = {df_to_json(data['metrics'])};",
        f"const FORECASTS   = {df_to_json(data['forecasts'])};",
        f"const SHAP_DATA   = {df_to_json(data['shap'])};",
        f"const ELASTICITY  = {df_to_json(data['elasticity'])};",
        f"const LP_DATA     = {df_to_json(data['lp'])};",
        f"const CV_DATA     = {df_to_json(data['cv'])};",
        f"const DID_DATA    = {safe_json_dumps(data['did'])};",
        f"const MONTHLY_TS  = {safe_json_dumps(data['monthly_ts'])};",
        f"const SYD_WEEKLY  = {safe_json_dumps(data['syd_weekly'])};",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# TEMPLATE BUILDER  — reads frontend/ assets and inlines them
# ─────────────────────────────────────────────────────────────────────

def build_html(data_js: str) -> str:
    """
    Read frontend/index.html, inline frontend/style.css and frontend/app.js,
    then inject the serialised data JS.
    """
    template = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css      = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")
    app_js   = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    # Inline stylesheet
    template = template.replace(
        '<link rel="stylesheet" href="style.css">',
        f"<style>\n{css}</style>"
    )
    # Inline app JS
    template = template.replace(
        '<script src="app.js"></script>',
        f"<script>\n{app_js}\n</script>"
    )
    # Inject data
    template = template.replace("%%DATA_JS%%", data_js)

    return template


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 55)
    log.info("  Step 5 — Build Interactive Dashboard")
    log.info("=" * 55)

    ensure_dirs(PATHS["outputs"])

    data    = load_all_results()
    data_js = build_data_js(data)
    html    = build_html(data_js)

    out_path = PATHS["outputs"] / "apex_dashboard.html"
    out_path.write_text(html, encoding="utf-8")

    size_kb = out_path.stat().st_size // 1024
    log.info("Dashboard written: %s  (%d KB)", out_path, size_kb)
    log.info("Open in browser: file://%s", out_path.resolve())
    log.info("Step 5 complete.")


if __name__ == "__main__":
    main()
