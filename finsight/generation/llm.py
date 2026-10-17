"""OpenAI-compatible LLM client (Groq, OpenAI, Ollama, ...)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import APIConnectionError, BadRequestError, OpenAI, RateLimitError

@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""


class DailyLimitError(Exception):
    """The model's daily token budget is used up: retrying for a few seconds will not help."""


class OpenAICompatibleLLM:
    """Works with Groq, OpenAI, Ollama, or any OpenAI-compatible API."""

    def __init__(self, base_url, model, api_key=None, temperature=0.0, max_tokens=2000,
                 reasoning_effort=None, max_retries=5, timeout=120):
        if not model:
            raise ValueError("No LLM model set: set generation.model in config.yaml or the LLM_MODEL environment variable.")
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout, max_retries=0)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.reasoning_effort, self.max_retries = reasoning_effort, max_retries

    def chat(self, messages) -> LLMResponse:
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except BadRequestError:
                if "extra_body" not in kwargs:
                    raise
                kwargs.pop("extra_body")           # model does not support reasoning_effort
                self.reasoning_effort = None
            except (RateLimitError, APIConnectionError) as exc:
                if "per day" in str(exc):          # daily budget gone: stop now instead of retrying
                    raise DailyLimitError(str(exc)) from exc
                if attempt == self.max_retries:
                    raise
                wait = 2 ** attempt * 5
                print(f"Rate limited, retrying in {wait}s ...")
                time.sleep(wait)
        choice, usage = resp.choices[0], resp.usage
        return LLMResponse(choice.message.content or "", getattr(usage, "prompt_tokens", 0) or 0,
                           getattr(usage, "completion_tokens", 0) or 0, choice.finish_reason or "")


def llm_from_config(gcfg, model=None) -> OpenAICompatibleLLM:
    """model=None uses LLM_MODEL; pass another name to use a different model (e.g. for drafting or judging)."""
    return OpenAICompatibleLLM(base_url=os.getenv("LLM_BASE_URL") or gcfg["base_url"],
                               model=model or os.getenv("LLM_MODEL") or gcfg.get("model"), api_key=os.getenv("LLM_API_KEY"),
                               temperature=gcfg.get("temperature", 0.0), max_tokens=gcfg.get("max_tokens", 2000),
                               reasoning_effort=gcfg.get("reasoning_effort"),
                               max_retries=gcfg.get("max_retries", 5), timeout=gcfg.get("timeout", 120))
