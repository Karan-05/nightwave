"""Tests for the provider-neutral LLM adapter.

All tests are deterministic — no real API calls. FakeAdapter stands in for both
AnthropicAdapter and OpenAIAdapter to verify the protocol contract and factory
dispatch logic.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from nightwave.multiagent.llm import LLMClient, get_llm_client


# ── Fake adapter ──────────────────────────────────────────────────────────────

class FakeAdapter:
    """Minimal LLMClient implementation for testing."""

    model: str = "fake-model"

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        return '{"response":"ok","confidence":0.9,"reasoning":"r","citations":[]}', 10, 5


# ── Protocol conformance ──────────────────────────────────────────────────────

def test_fake_adapter_satisfies_protocol() -> None:
    """FakeAdapter must satisfy the LLMClient Protocol at runtime."""
    adapter = FakeAdapter()
    assert isinstance(adapter, LLMClient), (
        "FakeAdapter does not satisfy LLMClient Protocol — check model attr and chat signature"
    )


def test_chat_returns_three_tuple() -> None:
    adapter = FakeAdapter()
    result = adapter.chat(system="s", messages=[{"role": "user", "content": "q"}])
    assert isinstance(result, tuple) and len(result) == 3, (
        f"chat() must return (text, input_tokens, output_tokens), got {result!r}"
    )


def test_chat_text_is_string() -> None:
    adapter = FakeAdapter()
    text, _, _ = adapter.chat(system="s", messages=[])
    assert isinstance(text, str)


def test_chat_token_counts_are_ints() -> None:
    adapter = FakeAdapter()
    _, inp, out = adapter.chat(system="s", messages=[])
    assert isinstance(inp, int), f"input_tokens must be int, got {type(inp)}"
    assert isinstance(out, int), f"output_tokens must be int, got {type(out)}"


def test_chat_token_counts_non_negative() -> None:
    adapter = FakeAdapter()
    _, inp, out = adapter.chat(system="s", messages=[])
    assert inp >= 0 and out >= 0


# ── Factory dispatch ──────────────────────────────────────────────────────────

def test_factory_returns_anthropic_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without NIGHTWAVE_LLM_PROVIDER set, factory must return AnthropicAdapter."""
    monkeypatch.delenv("NIGHTWAVE_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.delenv("NIGHTWAVE_LLM_MODEL", raising=False)
    monkeypatch.delenv("SYNTH_MODEL", raising=False)

    with patch("anthropic.Anthropic"):
        client = get_llm_client()

    from nightwave.multiagent.llm import AnthropicAdapter
    assert isinstance(client, AnthropicAdapter), (
        f"Expected AnthropicAdapter by default, got {type(client)}"
    )


def test_factory_returns_openai_when_provider_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With NIGHTWAVE_LLM_PROVIDER=openai, factory must return OpenAIAdapter."""
    monkeypatch.setenv("NIGHTWAVE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("NIGHTWAVE_LLM_MODEL", raising=False)

    mock_openai_mod = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai_mod}):
        client = get_llm_client()

    from nightwave.multiagent.llm import OpenAIAdapter
    assert isinstance(client, OpenAIAdapter), (
        f"Expected OpenAIAdapter when NIGHTWAVE_LLM_PROVIDER=openai, got {type(client)}"
    )


def test_factory_default_anthropic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default Anthropic model must be claude-sonnet-4-6 when no model env vars set."""
    monkeypatch.delenv("NIGHTWAVE_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.delenv("NIGHTWAVE_LLM_MODEL", raising=False)
    monkeypatch.delenv("SYNTH_MODEL", raising=False)

    with patch("anthropic.Anthropic"):
        client = get_llm_client()

    assert client.model == "claude-sonnet-4-6", (
        f"Expected default model 'claude-sonnet-4-6', got {client.model!r}"
    )


def test_factory_custom_model_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """NIGHTWAVE_LLM_MODEL overrides the default model for any provider."""
    monkeypatch.delenv("NIGHTWAVE_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("NIGHTWAVE_LLM_MODEL", "claude-opus-4-7")
    monkeypatch.delenv("SYNTH_MODEL", raising=False)

    with patch("anthropic.Anthropic"):
        client = get_llm_client()

    assert client.model == "claude-opus-4-7", (
        f"Expected model from NIGHTWAVE_LLM_MODEL, got {client.model!r}"
    )


def test_factory_legacy_synth_model_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """SYNTH_MODEL is honoured as legacy alias when NIGHTWAVE_LLM_MODEL is not set."""
    monkeypatch.delenv("NIGHTWAVE_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.delenv("NIGHTWAVE_LLM_MODEL", raising=False)
    monkeypatch.setenv("SYNTH_MODEL", "claude-opus-4-7")

    with patch("anthropic.Anthropic"):
        client = get_llm_client()

    assert client.model == "claude-opus-4-7", (
        f"Expected SYNTH_MODEL alias to be respected, got {client.model!r}"
    )


def test_factory_missing_anthropic_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ANTHROPIC_API_KEY must raise KeyError, not a silent failure."""
    monkeypatch.delenv("NIGHTWAVE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(KeyError):
        get_llm_client()


def test_factory_missing_openai_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing OPENAI_API_KEY when provider=openai must raise KeyError."""
    monkeypatch.setenv("NIGHTWAVE_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_openai_mod = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai_mod}):
        with pytest.raises(KeyError):
            get_llm_client()
