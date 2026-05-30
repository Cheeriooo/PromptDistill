"""
Base class for all baseline compression methods.

Every baseline must implement:
    compress(prompt: str) -> str
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Output from a compression baseline."""

    original: str
    compressed: str
    method: str

    @property
    def original_tokens(self) -> int:
        return len(self.original.split())

    @property
    def compressed_tokens(self) -> int:
        return len(self.compressed.split())

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.original_tokens - self.compressed_tokens) / self.original_tokens

    def __str__(self) -> str:
        return (
            f"[{self.method}] {self.original_tokens} → {self.compressed_tokens} tokens "
            f"({self.compression_ratio:.1%} reduction)"
        )


class BaseCompressor(ABC):
    """Abstract base class for all compression baselines."""

    name: str = "base"

    @abstractmethod
    def compress(self, prompt: str) -> str:
        """Compress the prompt and return the compressed string."""
        ...

    def compress_result(self, prompt: str) -> CompressionResult:
        """Compress and return a structured CompressionResult."""
        compressed = self.compress(prompt)
        return CompressionResult(
            original=prompt,
            compressed=compressed,
            method=self.name,
        )
