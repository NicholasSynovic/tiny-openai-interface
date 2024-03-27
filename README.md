# Tiny OpenAi Interface

> A tiny wrapper library around the OpenAI Python library for simplified access
> to their model APIs

## About

`toi` is a small wrapper library around the OpenAi library to simplify the
process of interfacing with OpenAi's models.

It's main feature is it's `chat()` function that validates that the *token
length* of a given string fits within a model's supported input token range. It
also takes into consideration the system prompt's token size so as to not
overflow the model's input. This is all in the effort to reduce the amount of
frivolous spending in sending API calls to OpenAI.

## How to Install

This utility library should be cross compatible across platforms as it only
relies on standard Python. However, this has not been tested. So far only Linux
x86-64 has been tested and validated to work with this utility.

### Dependencies

- Python 3.10
- `poetry`

### Installation steps

1. Run `make` in the root directory of this repository

## How to Run

1. Import `toi` into your project
