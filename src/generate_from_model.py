# -*- coding: utf-8 -*-
"""Generate text from a prompt for a model in ./models

This module contains tools used for generating text from a GPT-2 model that
has been trained by tools in this project.
"""

import logging
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .config import MODELS_DIRECTORY_PATH

logging.getLogger("transformers").setLevel(logging.ERROR)


def prepare_model(model_name: str) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Prepare model and tokenizer for a given model name.

    Args:
        model_name (str): Name of model to be used, dir in ./models.

    Returns:
        tuple[GPT2LMHeadModel, GPT2Tokenizer]: tuple of model and tokenizer.
    """
    model_path: str = os.path.join(MODELS_DIRECTORY_PATH, model_name)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(
    prompt: str,
    model_name: str,
    max_length: int = 512,
    retries: int = 15,
    top_k: int = 15,
    top_p: float = 0.85,
    temperature: float = 1.5,
) -> str:
    """Generate text from a given prompt given a model name.

    Args:
        prompt (str): String used to generate text.
        model_name (str): Name of model dir name in ./models.
        max_length (int, optional): Max length of generated text. Defaults to
            512.
        retries (int, optional): Number of times text is allowed to be
            regenerated if attempts to generate are unsuccessful. Defaults to
            15.
        top_k (int, optional) Top K value. Defaults to 15.
        top_p (float, optional) Top P value. Defaults to 0.85.
        temperature (float, optional) Temperature value. Defaults to 1.5.

    Returns:
        str: Generated text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = prepare_model(model_name=model_name)
    model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # If you add ending punctuation to the prompt if it doesn't have it, the
    # output will, in general, be more like a response rather than simply
    # adding onto or repeating exactly the prompt.
    prompt = prompt + "." if prompt[-1] not in {".", "!", "?"} else prompt

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_length = len(
        tokenizer.decode(input_ids[0], skip_special_tokens=True)
    )
    attention_mask = torch.ones(input_ids.shape, device=device)

    gen_text = ""
    while len(gen_text) == 0 and retries > 0:
        retries = retries - 1
        gen_tokens = model.generate(
            input_ids=input_ids,
            max_length=max_length + len(input_ids[0]),
            num_return_sequences=5,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            attention_mask=attention_mask,
            do_sample=True,
        )
        gen_tokens_decoded = tokenizer.batch_decode(
            gen_tokens, skip_special_tokens=True
        )
        # print(gen_tokens_decoded)
        gen_tokens_decoded = max(
            gen_tokens_decoded, key=len
        )  # keep longest output
        gen_text = "".join(gen_tokens_decoded)[
            prompt_length:
        ].strip()  # combine

    return gen_text
