"""
Identity Baseline — no compression.

Returns the prompt unchanged. This is the upper bound for quality
and the lower bound for compression. Every method must beat this
on the Pareto frontier.
"""

from prompt_optimizer.baselines.base import BaseCompressor


class IdentityCompressor(BaseCompressor):
    """
    No-op baseline: returns the prompt exactly as-is.

    Metrics:
        compression_ratio = 0.0
        quality = 1.0 (by definition, since nothing changed)
    """

    name = "identity"

    def compress(self, prompt: str) -> str:
        return prompt
