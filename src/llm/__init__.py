"""LLM factory utilities."""
from typing import Any

from .base import LLM
from .openai import OpenAILLM


def get_llm(model_name: str, **kwargs: Any) -> LLM:
    """Return an LLM instance for the given model name.

    If the model name contains "gpt" (case-insensitive), an OpenAILLM instance
    is created and returned with the provided keyword arguments. Otherwise, a
    ValueError is raised.
    """
    if "gpt" in model_name.lower():
        return OpenAILLM(model_name=model_name, **kwargs)

    raise ValueError(
        f"Unsupported model_name '{model_name}'. Only GPT models are supported."
    )
