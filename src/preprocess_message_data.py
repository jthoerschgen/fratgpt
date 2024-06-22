# -*- coding: utf-8 -*-
"""Preprocess raw GroupMe exported data.

This module is used to preprocess message data from a GroupMe export so that it
can be used when training.

"""

import csv
import json
import logging
import os
import re
import typing
import unicodedata
from datetime import datetime

from .config import (
    GROUPME_EXPORT_DIRECTORY_PATH,
    OUTPUT_TRAINING_DATA_DIRECTORY_PATH,
    PHONE_NUMBER_REGEX,
    URL_REGEX,
)

logger = logging.getLogger(__name__)


def preprocess_message(message: dict) -> typing.Union[str, None]:
    """Process message text information.

    Args:
        message (dict): GroupMe message object.

    Returns:
        typing.Union[str, None]: Message text if it is valid after
            preprocessing, None if message text is invalid.

    """
    if (
        (message["text"] is None)  # Skip if no message text data
        or (
            message["text"] == "This message was deleted"
        )  # Skip deleted messages
        or (
            message["name"] in ("GroupMe", "system")
        )  # Skip if message is from system
        or ("event" in message.keys())  # Skip if event (poll, calendar, etc.)
    ):  # Skip conditions
        return None

    message_string: str = message["text"].strip()

    for pattern in (
        PHONE_NUMBER_REGEX,
        URL_REGEX,
    ):  # Patterns to remove from string
        message_string = re.sub(pattern, "", message_string)

    message_string.replace(
        "\u2019", "'"
    )  # Replace Unicode apostrophe with ASCII

    message_string = (
        unicodedata.normalize("NFKD", message_string)
        .encode("ascii", "ignore")
        .decode()
    )  # Make ASCII

    split_message: typing.List[str] = [
        part.rstrip("\n")
        for part in message_string.split("\n")
        if len(part) > 0
    ]  # Handle newline characters
    message_string = "".join(split_message)

    if len(message_string) <= 1 or message_string is None:
        # If message string is too short after cleaning or None, return None
        return None

    return message_string


def parse_groupme_export(
    directory_path: str = GROUPME_EXPORT_DIRECTORY_PATH,
    output_file_path: str = os.path.join(
        OUTPUT_TRAINING_DATA_DIRECTORY_PATH,
        f'output_data_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv',
    ),
) -> None:
    """Parse GroupMe chat exports for their message data to a csv.

    Args:
        directory_path (str, optional): Path containing GroupMe chat exports.
            Defaults to config.GROUPME_EXPORT_DIRECTORY_PATH.
        output_file_path (str, optional): Location where data will be saved.
            Defaults to os.path.join(
                    config.OUTPUT_TRAINING_DATA_DIRECTORY_PATH,
                    f'output_data_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv'
                ).
    """
    logger.info("Getting data from:   %s", GROUPME_EXPORT_DIRECTORY_PATH)
    logger.info("Data will output to: %s", output_file_path)
    for chat_dir in os.listdir(
        path=directory_path
    ):  # iterate over each chat export
        messages_path: str = os.path.join(
            directory_path, chat_dir, "message.json"
        )
        conversation_path: str = os.path.join(
            directory_path, chat_dir, "conversation.json"
        )

        for path in (messages_path, conversation_path):
            assert os.path.isfile(
                messages_path
            ), f"Cannot find file at: {path}"

        with open(
            messages_path, mode="r", encoding="utf-8-sig"
        ) as message_data_file:
            messages: typing.List[dict] = json.loads(message_data_file.read())

        # with open(
        #     conversation_path, mode="r", encoding="utf-8-sig"
        # ) as conversation_data_file:
        #     conversation: dict = json.loads(conversation_data_file.read())

        with open(
            output_file_path, mode="a", newline="", encoding="utf-8"
        ) as output_data_file:
            column_headers: tuple = (
                # "group_id",
                # "group_name",
                # "created_at",
                # "sender_id",
                # "sender_name",
                # "sender_type",
                "text",
            )
            writer = csv.writer(output_data_file, delimiter=",")
            writer.writerow(column_headers)
            for message in messages:
                # print(json.dumps(message, indent=2))
                message_text: typing.Union[str, None] = preprocess_message(
                    message
                )

                if message_text is None:  # Skip unusable messages
                    continue

                # conversation_name = (
                #     unicodedata.normalize("NFKD", conversation["name"])
                #     .encode("ascii", "ignore")
                #     .decode()
                # )  # Make ASCII
                # sender_name = (
                #     unicodedata.normalize("NFKD", message["name"])
                #     .encode("ascii", "ignore")
                #     .decode()
                # )  # Make ASCII

                writer.writerow(
                    (
                        # conversation["group_id"],
                        # conversation_name,
                        # message["created_at"],
                        # message["sender_id"],
                        # sender_name,
                        # message["sender_type"],
                        message_text,
                    )
                )
