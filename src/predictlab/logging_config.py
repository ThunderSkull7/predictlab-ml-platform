"""Centralized logging configuration for PredictLab."""

import logging
import sys

from predictlab.config import get_settings

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure root logger with consistent formatting and handlers.

    Parameters
    ----------
    log_level : Optional[str]
        Desired logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If omitted, defaults to the level defined in application settings.
    log_format : str
        Format string for log records.
    date_format : str
        Format string for timestamp display.

    Returns
    -------
    logging.Logger
        Configured root logger instance.
    """
    settings = get_settings()
    level_name = (log_level or settings.log_level).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger("predictlab")
    root_logger.setLevel(numeric_level)

    # Prevent duplicate handlers if called multiple times
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Avoid propagation to default root handler to prevent double output
    root_logger.propagate = False

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger child of the predictlab namespace.

    Parameters
    ----------
    name : str
        Name of the module requesting a logger.

    Returns
    -------
    logging.Logger
        Namespaced logger.
    """
    if name.startswith("predictlab.") or name == "predictlab":
        return logging.getLogger(name)
    return logging.getLogger(f"predictlab.{name}")
