import logging
import sys
from utils.config import config


def get_logger(name: str = "CustomerSupportAI") -> logging.Logger:
    """
    Creates and returns a standardized, color-friendly logger instance.

    Args:
        name: Name of the module requesting logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    level = logging.DEBUG if config.DEBUG_MODE else logging.INFO
    logger.setLevel(level)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Log Format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Primary application logger
logger = get_logger("SystemLogger")
