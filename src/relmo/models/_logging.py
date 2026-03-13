"""Small logging helpers for relmo models."""

from __future__ import annotations

import logging
from functools import cache


def _setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(name)s:%(lineno)d] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


@cache
def get_logger(name: str) -> logging.Logger:
    return _setup_logger(name)
