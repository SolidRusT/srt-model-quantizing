import logging
import os
from logging.handlers import TimedRotatingFileHandler

def create_logger(name, log_dir="logs", level=logging.DEBUG):
    """
    Create and configure a logger with the specified name and log directory.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be stored.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1)
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Ensure that we do not add multiple handlers to the logger
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
