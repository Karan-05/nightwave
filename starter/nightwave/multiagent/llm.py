"""Provider-neutral LLM client adapter.

Select provider via NIGHTWAVE_LLM_PROVIDER env var (default: anthropic).
Select model via NIGHTWAVE_LLM_MODEL env var.

Contract: chat(system, messages, max_tokens, temperature) -> (text, input_tokens, output_tokens)
Callers own retry logic; this layer only wraps the provider SDK.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        """Return (response_text, input_tokens, output_tokens)."""
        ...


class AnthropicAdapter:
    """Wraps the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        text = resp.content[0].text
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class OpenAIAdapter:
    """Wraps the OpenAI Chat Completions API.

    Note: OpenAI's native `response_format={"type": "json_object"}` could be
    used here for structured output, but we keep the same prompt-only JSON
    enforcement as the Anthropic path so the synthesizer's _extract_json fallback
    works identically across providers.
    """

    def __init__(self, api_key: str, model: str) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        all_messages = [{"role": "system", "content": system}] + messages
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=all_messages,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, usage.prompt_tokens, usage.completion_tokens


def get_llm_client() -> LLMClient:
    """Factory: reads NIGHTWAVE_LLM_PROVIDER and NIGHTWAVE_LLM_MODEL from env."""
    provider = os.getenv("NIGHTWAVE_LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        model = os.getenv("NIGHTWAVE_LLM_MODEL", "gpt-4o")
        return OpenAIAdapter(api_key=api_key, model=model)
    # Default: Anthropic. SYNTH_MODEL is the legacy alias.
    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = os.getenv("NIGHTWAVE_LLM_MODEL", os.getenv("SYNTH_MODEL", "claude-sonnet-4-6"))
    return AnthropicAdapter(api_key=api_key, model=model)
