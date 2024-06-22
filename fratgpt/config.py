# -*- coding: utf-8 -*-
"""Configurable values used in the project.

This module contains important values used by other modules.

Attributes:
    ROOT_DIRECTORY_PATH (str): Path for the root directory of the project.
    SRC_DIRECTORY_PATH (str): Path for the "/src" directory.
    LOGGING_DIRECTORY_PATH (str): Path for logs to be saved to.
    DATA_DIRECTORY_PATH (str): Path for the parent directory of export/input
        and training/output data.
    GROUPME_EXPORT_DIRECTORY_PATH (str): Path for the directory containing the
        GroupMe export data that is for training.
    OUTPUT_TRAINING_DATA_DIRECTORY_PATH (str): Path for the directory where
        training data is outputted to.
    MODELS_DIRECTORY_PATH (str): Path for the directory where models are kept.
    message_training_data_output_path (str): Path for the location of the file
        with message data used for training.
    PHONE_NUMBER_REGEX (re.Pattern[str]): Regular expression pattern for
        recognizing phone numbers
    URL_REGEX (re.Pattern[str]): Regular expression pattern for recognizing
        URLs

"""

import os
import re

# Directory Paths

ROOT_DIRECTORY_PATH: str = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__))
)
SRC_DIRECTORY_PATH: str = os.path.join(ROOT_DIRECTORY_PATH, "src")

LOGGING_DIRECTORY_PATH: str = os.path.join(SRC_DIRECTORY_PATH, "logs")

DATA_DIRECTORY_PATH: str = os.path.join(ROOT_DIRECTORY_PATH, "data")
GROUPME_EXPORT_DIRECTORY_PATH: str = os.path.join(
    DATA_DIRECTORY_PATH, "groupme_exports"
)
OUTPUT_TRAINING_DATA_DIRECTORY_PATH: str = os.path.join(
    DATA_DIRECTORY_PATH, "output_training_data"
)

MODELS_DIRECTORY_PATH: str = os.path.join(ROOT_DIRECTORY_PATH, "models")

for directory_path in (
    ROOT_DIRECTORY_PATH,
    SRC_DIRECTORY_PATH,
    LOGGING_DIRECTORY_PATH,
    DATA_DIRECTORY_PATH,
    GROUPME_EXPORT_DIRECTORY_PATH,
    OUTPUT_TRAINING_DATA_DIRECTORY_PATH,
    MODELS_DIRECTORY_PATH,
):  # Handle directories
    if not os.path.exists(directory_path):  # Create directory if needed
        os.mkdir(directory_path)

    assert os.path.isdir(
        directory_path
    ), f"{directory_path} is not a directory."

# Regular Expression Patterns

PHONE_NUMBER_REGEX = re.compile(
    r"(\+\d{1,3}\s?)?"
    + r"((\(\d{3}\)\s?)|(\d{3})(\s|-?))"
    + r"(\d{3}(\s|-?))"
    + r"(\d{4})"
    + r"(\s?(([E|e]xt[:|.|]?)|x|X)(\s?\d+))?"
)
URL_REGEX = re.compile(r"http\S+")
