"""
Logging module.

Taken from: https://github.com/KostasValeckas/PyLongslit/blob/main/pylongslit/logger.py
"""

import logging
from colorama import Fore, Style, init
from datetime import datetime

def get_logger(output_path, logger_name = "muse_astrometric_registration"):

    """
    Create and configure a logger.

    Parameters
    ----------
    output_path : str or Path
        The directory where the log file will be saved.

    logger_name : str, optional
        The name of the logger. Default is "muse_astrometric_registration".

    Returns
    -------
    logger : logging.Logger
        Configured logger object.
    """

    # Initialize colorama
    init(autoreset=True)

    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Close and remove leftover handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # Configure logging level
    logger.setLevel(logging.INFO)
    
    log_file_path = output_path / datetime.now().strftime("aligment_log_%Y-%m-%d_%H-%M-%S.log")

    # we add different information when we print to console and when we write to
    # file, so we need two handlers
    fh = logging.FileHandler(log_file_path)
    ch = logging.StreamHandler()


    # Create a custom formatter - this allows different colors for different log levels
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                record.levelname = f"{Fore.GREEN}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.WARNING:
                record.levelname = f"{Fore.YELLOW}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.ERROR:
                record.levelname = f"{Fore.RED}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.CRITICAL:
                record.levelname = (
                    f"{Fore.RED}{record.levelname}{Style.BRIGHT}{Style.RESET_ALL}"
                )
            return super().format(record)


    # For filelogging we add the dates, for console logging we don't
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    color_formatter = CustomFormatter(
        "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    fh.setFormatter(formatter)
    ch.setFormatter(color_formatter)

    # Add both handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Log messages
    logger.info("Logger initialized. Log will be saved in " + fh.baseFilename + ".")

    return logger