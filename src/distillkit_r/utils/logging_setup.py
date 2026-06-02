"""Project-wide logging configuration.

Provides a single ``configure_logging`` entry point so every script and module
emits consistently formatted records. Modules themselves never call
``basicConfig``; they only ever do ``logger = logging.getLogger(__name__)``.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO

logger = logging.getLogger(__name__)


def configure_logging(level: int = DEFAULT_LEVEL) -> None:
    """Configure the root logger for the whole process.

    Idempotent: repeated calls replace the existing stream handler rather than
    stacking duplicates, which keeps log lines from being emitted N times when
    several entry points configure logging in one process.

    Parameters
    ----------
    level : int
        Logging level applied to the root logger (e.g. ``logging.INFO``).

    Returns
    -------
    None
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)

    logger.debug("Logging configured at level %s", logging.getLevelName(level))
