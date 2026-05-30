"""
Unified LLM client supporting OpenAI, Anthropic, and Google Gemini.

Usage:
    from prompt_optimizer.llm.client import LLMClient

    # Uses env vars DEFAULT_LLM_PROVIDER + DEFAULT_LLM_MODEL
    client = LLMClient()
    result = client.complete("Explain gradient descent.")
    print(result.text, result.total_tokens)

    # Explicit Gemini usage
    client = LLMClient(provider="gemini", model="gemini-1.5-flash")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
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

Provider = Literal["openai", "anthropic", "gemini"]


@dataclass
class CompletionResult:
    """Structured result from an LLM completion call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    latency_ms: float

    @property
    def token_count(self) -> int:
        return self.total_tokens


class LLMClient:
    """
    Unified LLM client for OpenAI, Anthropic, and Google Gemini.

    Reads defaults from environment:
        DEFAULT_LLM_PROVIDER  → "gemini" | "openai" | "anthropic"
        DEFAULT_LLM_MODEL     → model name (e.g. "gemini-1.5-flash")

    API keys read from:
        GEMINI_API_KEY
        OPENAI_API_KEY
        ANTHROPIC_API_KEY

    Args:
        provider:    LLM provider to use.
        model:       Model name string.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens:  Max tokens to generate in the response.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.provider: Provider = provider or os.getenv("DEFAULT_LLM_PROVIDER", "gemini")  # type: ignore[assignment]
        self.model = model or os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-flash")
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
            prompt: The full prompt string.

        Returns:
            CompletionResult with text, token counts, and latency.
        """
        t0 = time.perf_counter()

        if self.provider == "gemini":
            result = self._complete_gemini(prompt)
        elif self.provider == "openai":
            result = self._complete_openai(prompt)
        elif self.provider == "anthropic":
            result = self._complete_anthropic(prompt)
        else:
            raise ValueError(
                f"Unknown provider: {self.provider!r}. "
                "Use 'gemini', 'openai', or 'anthropic'."
            )

        result.latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "LLM | provider=%s model=%s tokens=%d latency=%.0fms",
            self.provider,
            self.model,
            result.total_tokens,
            result.latency_ms,
        )
        return result

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        Uses the model's native tokenizer when available, else word-split.
        """
        if self.provider == "gemini":
            return self._count_tokens_gemini(text)
        if self.provider == "openai":
            return self._count_tokens_tiktoken(text)
        # Anthropic / fallback: ~4 chars per token
        return max(1, len(text.split()))

    # ------------------------------------------------------------------
    # Internal — client construction
    # ------------------------------------------------------------------

    def _build_client(self):
        if self.provider == "gemini":
            return self._build_gemini_client()
        if self.provider == "openai":
            return self._build_openai_client()
        if self.provider == "anthropic":
            return self._build_anthropic_client()
        raise ValueError(f"Unknown provider: {self.provider!r}")

    def _build_gemini_client(self):
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError:
            raise ImportError("Run: uv add google-generativeai") from None

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Get one at https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

    def _build_openai_client(self):
        try:
            import openai  # noqa: PLC0415
            return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Run: uv add openai") from None

    def _build_anthropic_client(self):
        try:
            import anthropic  # noqa: PLC0415
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("Run: uv add anthropic") from None

    # ------------------------------------------------------------------
    # Internal — Gemini
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _complete_gemini(self, prompt: str) -> CompletionResult:
        try:
            response = self._client.generate_content(prompt)

            # Gemini response structure
            text = response.text if hasattr(response, "text") else ""

            # Token usage (available in usage_metadata)
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (
                prompt_tokens + completion_tokens
            )

            return CompletionResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=self.model,
                provider="gemini",
                latency_ms=0.0,
            )
        except Exception as e:
            logger.warning("Gemini call failed (%s) — retrying...", type(e).__name__)
            raise

    def _count_tokens_gemini(self, text: str) -> int:
        """Use Gemini's native count_tokens endpoint (exact, not estimated)."""
        try:
            result = self._client.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fallback: word-level estimate
            return max(1, len(text.split()))

    # ------------------------------------------------------------------
    # Internal — OpenAI
    # ------------------------------------------------------------------

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
            logger.warning("OpenAI rate limit — backing off...")
            raise
        except openai.APIError as e:
            logger.error("OpenAI API error: %s", e)
            raise

    def _count_tokens_tiktoken(self, text: str) -> int:
        try:
            import tiktoken  # noqa: PLC0415
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text.split()))

    # ------------------------------------------------------------------
    # Internal — Anthropic
    # ------------------------------------------------------------------

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
