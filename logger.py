import logging
import sys
from pathlib import Path


def get_module_logger(name: str, log_path: Path) -> logging.Logger:
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(log_path / f"logs.log")
    fileHandler.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fileHandler)
    return logger
