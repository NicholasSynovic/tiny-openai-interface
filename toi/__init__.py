from json import loads
from typing import List

import tiktoken
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from tiktoken.core import Encoding

OPEN_AI_MODELS: dict[str, List[str | int]] = {
    "model": [
        "gpt-4-1106-vision-preview",
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
    idx: int = OPEN_AI_MODELS["model"].index(model)

    if tokenAmount < OPEN_AI_MODELS["inputTokens"][idx]:
        return True
    else:
        return False


def chat(text: str, systemPrompt: str, apiKey: str, model: str, seed: int = 42) -> dict:
    userTextTokenCount: int = countTokens(text=text, model=model)

    if validateTokenLength(tokenAmount=userTextTokenCount, model=model) is False:
        print("ERROR: Input text token count is larger than the model token limit")
        return {}

    systemPromptTokenCount: int = countTokens(text=systemPrompt, model=model)
    if validateTokenLength(tokenAmount=systemPromptTokenCount, model=model) is False:
        print("ERROR: System prompt token count is larger than the model token limit")
        return {}

    totalTokenCount: int = userTextTokenCount + systemPromptTokenCount
    if validateTokenLength(tokenAmount=totalTokenCount, model=model) is False:
        print(
            "ERROR: Too many tokens shared between input text and system prompt which is larger than the model token limit"
        )
        return {}

    headers: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": text},
    ]

    client: OpenAI = OpenAI(api_key=apiKey)

    response: ChatCompletion = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=headers,
        seed=seed,
    )

    content: str | None = response.choices[0].message.content

    if type(content) is str:
        return loads(s=content)
    else:
        return {}
