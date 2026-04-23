"""
All file read / write operations.
Never scatter pd.read_csv() calls across model files.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── CSV ────────────────────────────────────────────────────────────

def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV with consistent logging."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    log.info("Loading CSV: %s", path.name)
    df = pd.read_csv(path, **kwargs)
    log.info("  → %d rows × %d cols", len(df), df.shape[1])
    return df


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save DataFrame to CSV, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    log.info("Saved CSV: %s  (%d rows)", path.name, len(df))


# ── JSON ───────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    """Load a JSON file and return the parsed object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    log.info("Loading JSON: %s", path.name)
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    """Serialise *obj* to JSON, handling numpy scalars and arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy/pandas types to native Python for JSON serialisation
    def _convert(o):
        import numpy as np
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)):  return o.tolist()
        if isinstance(o, (np.bool_,)):    return bool(o)
        raise TypeError(f"Not serialisable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_convert)
    log.info("Saved JSON: %s", path.name)


# ── Ensure dirs exist ──────────────────────────────────────────────

def ensure_dirs(*paths) -> None:
    """Create each directory (and any missing parents) if it doesn't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
