# -*- coding: utf-8 -*-
"""Command-line Interface for FratGPT

This module handles arguments for the CLI.
"""

import argparse
import inspect

from src.generate_from_model import generate
from src.preprocess_message_data import parse_groupme_export

parser = argparse.ArgumentParser(
    prog="fratgpt",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="FratGPT",
)
subparsers = parser.add_subparsers(dest="command", required=True)

generate_message_parser = subparsers.add_parser(
    "generate-message", help="Generate text from a model."
)

generate_defaults: dict = {
    param.name: param.default
    for param in inspect.signature(generate).parameters.values()
    if param.default is not inspect.Parameter.empty
}

generate_message_parser.add_argument(
    "--model",
    "-m",
    action="store",
    type=str,
    help="Model Name.",
    metavar="MODEL NAME"
)
generate_message_parser.add_argument(
    "--prompt",
    "-p",
    action="store",
    type=str,
    help="Prompt used for generating text.",
    metavar="PROMPT TEXT"
)
generate_message_parser.add_argument(
    "--max-length",
    "-l",
    action="store",
    type=int,
    help=(
        f"Max len for output text. Default: {generate_defaults["max_length"]}"
    ),
    metavar="MAX LEN VAL",
    default=generate_defaults["max_length"],
)
generate_message_parser.add_argument(
    "--top_k",
    action="store",
    type=int,
    help=f"Top K value. Default: {generate_defaults["top_k"]}",
    metavar="TOP_K VAL",
    default=generate_defaults["top_k"],
)
generate_message_parser.add_argument(
    "--top_p",
    action="store",
    type=float,
    help=f"Top P value. Default: {generate_defaults["top_p"]}",
    metavar="TOP_P VAL",
    default=generate_defaults["top_p"],
)
generate_message_parser.add_argument(
    "--temperature",
    "-t",
    action="store",
    type=float,
    help=f"Temperature value. Default: {generate_defaults["temperature"]}",
    metavar="TEMP VAL",
    default=generate_defaults["temperature"],
)

preprocess_export_parser = subparsers.add_parser(
    "preprocess-export",
    help="Preprocess GroupMe exports in ./data/groupme_exports",
)

args = parser.parse_args()

if args.command == "generate-message":
    generated_text: str = generate(
        prompt=args.prompt,
        model_name=args.model,
        max_length=args.max_length,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    print(generated_text)

if args.command == "preprocess-export":
    parse_groupme_export()
