# -*- coding: utf-8 -*-
"""Initialization

Perform any necessary operations for the module to run.

Attributes:
    logging_format (str): Format for logging messages, includes the time,
        logging level, and message data.

"""
import datetime
import logging
import os

from .config import (
    DATA_DIRECTORY_PATH,
    GROUPME_EXPORT_DIRECTORY_PATH,
    LOGGING_DIRECTORY_PATH,
    MODELS_DIRECTORY_PATH,
    OUTPUT_TRAINING_DATA_DIRECTORY_PATH,
)

for directory_path in (
    DATA_DIRECTORY_PATH,
    GROUPME_EXPORT_DIRECTORY_PATH,
    LOGGING_DIRECTORY_PATH,
    MODELS_DIRECTORY_PATH,
    OUTPUT_TRAINING_DATA_DIRECTORY_PATH,
):  # make important dirs
    os.makedirs(directory_path, exist_ok=True)

# Set up logging for project
logging_format: str = "%(asctime)s %(levelname)-8s %(message)s"

logging.basicConfig(
    format=logging_format,
    level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=os.path.join(
        LOGGING_DIRECTORY_PATH,
        f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
    ),
)

logger = logging.getLogger(__name__)
