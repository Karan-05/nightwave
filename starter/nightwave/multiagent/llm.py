"""Provider-neutral LLM client adapter.

Select provider via NIGHTWAVE_LLM_PROVIDER env var (default: anthropic).
Supported: anthropic, openai, gemini, bedrock, ollama
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
            messages=messages,  # type: ignore[arg-type]
        )
        text = getattr(resp.content[0], "text", "")
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class OpenAIAdapter:
    """Wraps the OpenAI Chat Completions API.

    Note: OpenAI's native `response_format={"type": "json_object"}` could be
    used here for structured output, but we keep the same prompt-only JSON
    enforcement as the Anthropic path so the synthesizer's _extract_json fallback
    works identically across providers.
    """

    def __init__(self, api_key: str, model: str) -> None:
        import openai  # type: ignore[import-not-found]

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


class GeminiAdapter:
    """Wraps the Google Generative AI API (Gemini).

    Requires: pip install google-generativeai
    Key: GOOGLE_API_KEY
    Default model: gemini-2.0-flash
    """

    def __init__(self, api_key: str, model: str) -> None:
        import google.generativeai as genai  # type: ignore[import-not-found]

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        model_client = self._genai.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        # Convert OpenAI-style role names to Gemini's ("assistant" → "model")
        contents = [
            {
                "role": "model" if msg["role"] == "assistant" else msg["role"],
                "parts": [msg["content"]],
            }
            for msg in messages
        ]
        resp = model_client.generate_content(contents)
        usage = resp.usage_metadata
        return resp.text, usage.prompt_token_count, usage.candidates_token_count


class BedrockAdapter:
    """Wraps AWS Bedrock Converse API.

    Supports Claude, Titan, Llama, Mistral, and other Bedrock-hosted models.
    Auth via boto3 credential chain (env vars, ~/.aws/credentials, or IAM role).
    Requires: pip install boto3
    Region: AWS_REGION or AWS_DEFAULT_REGION (default: us-east-1)
    Default model: anthropic.claude-3-5-sonnet-20241022-v2:0
    """

    def __init__(self, model: str, region: str = "us-east-1") -> None:
        import boto3  # type: ignore[import-not-found]

        self._client = boto3.client("bedrock-runtime", region_name=region)
        self.model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        bedrock_messages = [
            {"role": msg["role"], "content": [{"text": msg["content"]}]} for msg in messages
        ]
        resp = self._client.converse(
            modelId=self.model,
            system=[{"text": system}],
            messages=bedrock_messages,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        usage = resp["usage"]
        return text, usage["inputTokens"], usage["outputTokens"]


class OllamaAdapter:
    """Wraps Ollama via its OpenAI-compatible REST API.

    Requires: Ollama running locally (https://ollama.ai) + pip install openai
    Base URL: OLLAMA_BASE_URL (default: http://localhost:11434/v1)
    No API key required — uses the dummy value "ollama" expected by the SDK.
    Default model: llama3.2
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1") -> None:
        import openai  # type: ignore[import-not-found]

        self._client = openai.OpenAI(api_key="ollama", base_url=base_url)
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
    if provider == "gemini":
        api_key = os.environ["GOOGLE_API_KEY"]
        model = os.getenv("NIGHTWAVE_LLM_MODEL", "gemini-2.0-flash")
        return GeminiAdapter(api_key=api_key, model=model)
    if provider == "bedrock":
        model = os.getenv("NIGHTWAVE_LLM_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        return BedrockAdapter(model=model, region=region)
    if provider == "ollama":
        model = os.getenv("NIGHTWAVE_LLM_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OllamaAdapter(model=model, base_url=base_url)
    # Default: Anthropic. SYNTH_MODEL is the legacy alias.
    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = os.getenv("NIGHTWAVE_LLM_MODEL", os.getenv("SYNTH_MODEL", "claude-sonnet-4-6"))
    return AnthropicAdapter(api_key=api_key, model=model)
