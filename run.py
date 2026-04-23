#!/usr/bin/env python3
"""
Single entry point — runs the complete Apex pipeline end-to-end.

Usage:
    python run.py            # run all 5 steps
    python run.py --step 2   # run only step 2 (forecasts)

Steps:
    1  Generate dataset      (src/data/generator.py)
    2  Fit hybrid forecaster (src/models/hybrid_forecaster.py)
    3  Causal inference      (src/causal/)
    4  LP seat optimiser     (src/optimiser/seat_optimiser.py)
    5  Build dashboard       (pipeline/05_build_dashboard.py)
"""

import argparse
import sys
import time
from pathlib import Path

# Make project importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import get_logger, ensure_dirs
from config import PATHS

log = get_logger("run")

STEPS = {
    1: ("Generate dataset",       "pipeline.01_generate_data"),
    2: ("Fit hybrid forecaster",  "pipeline.02_run_forecasts"),
    3: ("Causal inference",       "pipeline.03_causal_inference"),
    4: ("LP seat optimiser",      "pipeline.04_seat_optimiser"),
    5: ("Build dashboard",        "pipeline.05_build_dashboard"),
}


def run_step(step_num: int) -> None:
    """Import and execute the ``main()`` function of a pipeline step."""
    label, module_path = STEPS[step_num]
    log.info("")
    log.info("━" * 55)
    log.info("  STEP %d / 5 — %s", step_num, label)
    log.info("━" * 55)

    import importlib
    mod = importlib.import_module(module_path)
    t0  = time.perf_counter()
    mod.main()
    elapsed = time.perf_counter() - t0
    log.info("  ✓ Step %d complete (%.1fs)", step_num, elapsed)


def main() -> None:
    """Parse CLI arguments and run the requested pipeline step(s)."""
    parser = argparse.ArgumentParser(description="Apex pipeline runner")
    parser.add_argument("--step", type=int, choices=range(1, 6),
                        help="Run only this step (1–5). Default: run all.")
    args = parser.parse_args()

    # Ensure output directories exist
    for path in PATHS.values():
        ensure_dirs(path)

    log.info("=" * 55)
    log.info("  APEX — Revenue Intelligence Platform")
    log.info("  Qantas Data Scientist (RM) Role · R111813")
    log.info("=" * 55)

    if args.step:
        run_step(args.step)
    else:
        t_total = time.perf_counter()
        for step_num in STEPS:
            run_step(step_num)
        elapsed = time.perf_counter() - t_total
        log.info("")
        log.info("=" * 55)
        log.info("  ALL STEPS COMPLETE  (%.1fs total)", elapsed)
        log.info("  Open: outputs/apex_dashboard.html")
        log.info("=" * 55)


if __name__ == "__main__":
    main()
