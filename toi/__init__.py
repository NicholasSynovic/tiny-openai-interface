from typing import List

import tiktoken
from pandas import DataFrame
from tiktoken.core import Encoding

OPEN_AI_MODELS: dict[str, List[str | int]] = {
    "model": [
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "babbage-002",
        "davinci-002",
    ],
    "inputTokens": [
        128000,
        128000,
        128000,
        128000,
        8192,
        8192,
        32768,
        32768,
        16385,
        4096,
        16385,
        4096,
        16385,
        4096,
        16385,
        16385,
        16385,
    ],
    "outputTokens": [
        4096,
        4096,
        4096,
        4096,
        0,
        0,
        0,
        0,
        4096,
        4096,
        4096,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
}


def countTokens(text: str, model: str) -> int:
    encoding: Encoding = tiktoken.encoding_for_model(model_name=model)
    encodedText: List[int] = encoding.encode(text=text)
    return len(encodedText)


def validateTokenLength(tokenAmount: int, model: str) -> bool:
    idx: int = list(OPEN_AI_MODELS.keys()).index(model)

    if tokenAmount < OPEN_AI_MODELS["inputTokens"][idx]:
        return True
    else:
        return False
