"""
Step 1 of the reproducible pipeline.
Generates the BITRE-grounded Australian aviation dataset.

Usage:
    python pipeline/01_generate_data.py

Output:
    data/raw/australian_aviation.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generator import DatasetGenerator
from src.utils import get_logger

log = get_logger(__name__)


def main():
    """Generate and persist the BITRE-grounded aviation dataset."""
    log.info("=" * 60)
    log.info("  Step 1 — Dataset Generation")
    log.info("=" * 60)

    gen = DatasetGenerator(seed=42)
    df  = gen.generate()
    out = gen.save(df)

    log.info("Done. Dataset saved → %s", out)
    log.info("Shape: %d rows × %d cols", *df.shape)
    log.info(
        "Routes: %s",
        ", ".join(sorted(df["route"].unique()))
    )


if __name__ == "__main__":
    main()
