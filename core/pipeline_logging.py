"""Lightweight logging for the FAERS risk intelligence pipeline."""

from __future__ import annotations

import logging
from pathlib import Path


LOG_FILE = Path("pipeline_trace.log")


def get_logger(name: str = "faers_risk") -> logging.Logger:
    """Return a configured project logger with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger

