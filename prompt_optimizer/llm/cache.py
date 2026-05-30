"""
Disk-based LLM response cache.

Responses are stored as JSON files, keyed by a SHA-256 hash of
(prompt, model, provider, temperature).  This means identical calls
are free after the first API round-trip.

Usage:
    from prompt_optimizer.llm.cache import ResponseCache
    from prompt_optimizer.llm.client import LLMClient, CompletionResult

    cache = ResponseCache()
    client = LLMClient()

    result = cache.get_or_fetch(prompt="Explain gravity.", client=client)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from prompt_optimizer.llm.client import CompletionResult, LLMClient

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache/llm_responses"))
_CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"


class ResponseCache:
    """
    Disk-based cache for LLM responses.

    Args:
        cache_dir:    Directory to store cached responses.
        enabled:      If False, always calls the LLM (useful to force refresh).
    """

    def __init__(
        self,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
        enabled: bool = _CACHE_ENABLED,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_fetch(self, prompt: str, client: LLMClient) -> CompletionResult:
        """
        Return a cached result if it exists, otherwise call the LLM and cache it.

        Args:
            prompt: The prompt string.
            client: An LLMClient instance (used for cache key + actual call).

        Returns:
            CompletionResult (possibly from cache).
        """
        if not self.enabled:
            return client.complete(prompt)

        key = self._cache_key(prompt, client)
        cached = self._load(key)
        if cached is not None:
            logger.debug("Cache HIT for key %s", key[:12])
            return cached

        logger.debug("Cache MISS for key %s — calling LLM", key[:12])
        result = client.complete(prompt)
        self._save(key, result)
        return result

    def invalidate(self, prompt: str, client: LLMClient) -> None:
        """Delete a specific cached entry."""
        key = self._cache_key(prompt, client)
        path = self._key_path(key)
        if path.exists():
            path.unlink()
            logger.info("Invalidated cache key %s", key[:12])

    def clear_all(self) -> int:
        """Delete all cached entries. Returns number of files deleted."""
        files = list(self.cache_dir.glob("*.json"))
        for f in files:
            f.unlink()
        logger.info("Cleared %d cache entries", len(files))
        return len(files)

    @property
    def size(self) -> int:
        """Number of cached entries on disk."""
        return len(list(self.cache_dir.glob("*.json")))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(prompt: str, client: LLMClient) -> str:
        payload = json.dumps(
            {
                "prompt": prompt,
                "model": client.model,
                "provider": client.provider,
                "temperature": client.temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _key_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load(self, key: str) -> CompletionResult | None:
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CompletionResult(**data)
        except Exception as e:
            logger.warning("Corrupt cache file %s: %s — skipping", path.name, e)
            return None

    def _save(self, key: str, result: CompletionResult) -> None:
        path = self._key_path(key)
        data = {
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "model": result.model,
            "provider": result.provider,
            "latency_ms": result.latency_ms,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
