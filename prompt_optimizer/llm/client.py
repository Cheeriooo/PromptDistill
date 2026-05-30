"""
Unified LLM client supporting OpenAI and Anthropic.

Usage:
    from prompt_optimizer.llm.client import LLMClient

    client = LLMClient()
    response, token_count = client.complete("Explain gravity.")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

Provider = Literal["openai", "anthropic"]


@dataclass
class CompletionResult:
    """Structured result from an LLM completion call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: Provider
    latency_ms: float

    @property
    def token_count(self) -> int:
        return self.total_tokens


class LLMClient:
    """
    Unified LLM client for OpenAI and Anthropic.

    Automatically falls back to environment variables for API keys.
    Supports retry logic with exponential backoff.

    Args:
        provider:    "openai" or "anthropic"  (default: from env DEFAULT_LLM_PROVIDER)
        model:       model name               (default: from env DEFAULT_LLM_MODEL)
        temperature: sampling temperature
        max_tokens:  max tokens to generate
    """

    def __init__(
        self,
        provider: Provider | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.provider: Provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")  # type: ignore[assignment]
        self.model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> CompletionResult:
        """
        Send a prompt and return a CompletionResult.

        Args:
            prompt: The full prompt string to send.

        Returns:
            CompletionResult with text, token counts, latency.
        """
        t0 = time.perf_counter()
        if self.provider == "openai":
            result = self._complete_openai(prompt)
        elif self.provider == "anthropic":
            result = self._complete_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")
        result.latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "LLM call | provider=%s model=%s tokens=%d latency=%.0fms",
            self.provider,
            self.model,
            result.total_tokens,
            result.latency_ms,
        )
        return result

    def count_tokens(self, text: str) -> int:
        """
        Count tokens for a string using the model's tokenizer.
        Falls back to a word-split estimate if tokenizer is unavailable.
        """
        if self.provider == "openai":
            return self._count_tokens_openai(text)
        # Anthropic: use a rough estimate (1 token ≈ 4 chars)
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Internal — OpenAI
    # ------------------------------------------------------------------

    def _build_client(self):
        if self.provider == "openai":
            try:
                import openai  # noqa: PLC0415
                return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("Run: uv add openai") from None
        elif self.provider == "anthropic":
            try:
                import anthropic  # noqa: PLC0415
                return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("Run: uv add anthropic") from None
        raise ValueError(f"Unknown provider: {self.provider!r}")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _complete_openai(self, prompt: str) -> CompletionResult:
        import openai  # noqa: PLC0415

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return CompletionResult(
                text=response.choices[0].message.content or "",
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=self.model,
                provider="openai",
                latency_ms=0.0,
            )
        except openai.RateLimitError:
            logger.warning("OpenAI rate limit hit — backing off...")
            raise
        except openai.APIError as e:
            logger.error("OpenAI API error: %s", e)
            raise

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _complete_anthropic(self, prompt: str) -> CompletionResult:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return CompletionResult(
            text=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            model=self.model,
            provider="anthropic",
            latency_ms=0.0,
        )

    def _count_tokens_openai(self, text: str) -> int:
        try:
            import tiktoken  # noqa: PLC0415

            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            # tiktoken not installed or model unknown — use char estimate
            return max(1, len(text) // 4)
