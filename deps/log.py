"""Log into the console and the file."""

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

# https://docs.python.org/3/library/logging.html#logging.Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a file handler to write logs to a file
file_handler = RotatingFileHandler(
    "app.log",
    mode="a",
    maxBytes=5 * 1024 * 1024,
    backupCount=2,
    encoding=None,
    delay=False,
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create a stream handler to print logs to the console
console_handler = logging.StreamHandler()
# You can set the desired log level for console output
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def print_log(message: str) -> None:
    """Print the message to the log"""
    logger.info(message)


def print_error_log(message: str) -> None:
    """Print the error to the log"""
    logger.error(message)


def print_warning_log(message: str) -> None:
    """Print the warning to the log"""
    logger.warning(message)
