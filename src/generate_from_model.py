# -*- coding: utf-8 -*-
"""Generate text from a prompt for a model in ./models

This module contains tools used for generating text from a GPT-2 model that
has been trained by tools in this project.
"""

import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)


def prepare_model(model_path: str) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Prepare model and tokenizer for a given model name.

    Args:
        model_path (str): Path to model to be used.

    Returns:
        tuple[GPT2LMHeadModel, GPT2Tokenizer]: tuple of model and tokenizer.
    """
    model = GPT2LMHeadModel.from_pretrained(model_path)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(
    prompt: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    max_length: int = 512,
    retries: int = 15,
    top_k: int = 15,
    top_p: float = 0.85,
    temperature: float = 1.5,
) -> str:
    """Generate text from a given prompt given a model name.

    Args:
        prompt (str): String used to generate text.
        model (GPT2LMHeadModel): Model used for generation.
        tokenizer (GPT2Tokenizer): Tokenizer used for tokenization.
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


def generate_from_path(
    prompt: str,
    model_path: str,
    max_length: int = 512,
    retries: int = 15,
    top_k: int = 15,
    top_p: float = 0.85,
    temperature: float = 1.5,
) -> str:
    """Generate text from a given prompt given a model name.

    Args:
        prompt (str): String used to generate text.
        model_path (str): Path to model to be used.
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
    model, tokenizer = prepare_model(model_path)
    return generate(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        retries=retries,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
