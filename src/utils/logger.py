"""
Centralised logging configuration.
All modules import `get_logger(__name__)` — never print() directly.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with consistent formatting.

    Usage:
        from src.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Training model...")
        log.warning("Low sample size on fold 3")
        log.error("LP solver did not converge")
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on reimport
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
