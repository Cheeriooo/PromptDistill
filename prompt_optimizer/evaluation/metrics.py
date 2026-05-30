"""
Core metrics for the prompt optimizer.

Provides:
- BERTScore F1 (semantic quality)
- Compression ratio
- Token delta
- ROUGE-L (secondary)
- Estimated API cost savings

Usage:
    from prompt_optimizer.evaluation.metrics import compute_metrics

    scores = compute_metrics(
        original_prompt="...",
        compressed_prompt="...",
        original_response="The capital of France is Paris.",
        compressed_response="Paris is France's capital.",
    )
    print(scores.bertscore_f1, scores.compression_ratio)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table ($ per 1K tokens, as of mid-2025 — update as needed)
# ---------------------------------------------------------------------------
_TOKEN_PRICES: dict[str, float] = {
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.00015,
    "gpt-4-turbo": 0.01,
    "claude-3-5-sonnet-20241022": 0.003,
    "claude-3-haiku-20240307": 0.00025,
    "default": 0.001,
}


@dataclass
class EvalScores:
    """All metrics for a single (original, compressed) prompt pair."""

    # ----- Quality -----
    bertscore_f1: float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    rouge_l: float = 0.0

    # ----- Compression -----
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0   # 0.4 means 40% fewer tokens
    token_delta: int = 0              # tokens saved (positive = fewer tokens)

    # ----- Cost -----
    cost_savings_usd: float = 0.0    # per single call
    model: str = "unknown"

    # ----- Meta -----
    quality_degradation: float = 0.0  # 1 - bertscore_f1 (lower is better)

    def __str__(self) -> str:
        return (
            f"BERTScore F1: {self.bertscore_f1:.4f} | "
            f"Compression: {self.compression_ratio:.1%} | "
            f"Tokens: {self.original_tokens} → {self.compressed_tokens} "
            f"(-{self.token_delta}) | "
            f"Cost saved: ${self.cost_savings_usd:.6f}"
        )


class MetricsComputer:
    """
    Computes quality and compression metrics for a prompt pair.

    BERTScore is expensive (loads a BERT model on first call). The scorer
    is lazily initialized and cached for reuse.

    Args:
        bertscore_model: HuggingFace model for BERTScore.
        device:          "cpu" or "cuda" (auto-detected if None).
    """

    def __init__(
        self,
        bertscore_model: str | None = None,
        device: str | None = None,
    ) -> None:
        self.bertscore_model = bertscore_model or os.getenv(
            "BERTSCORE_MODEL", "roberta-large"
        )
        self.device = device  # None = auto
        self._bertscore = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        original_prompt: str,
        compressed_prompt: str,
        original_response: str,
        compressed_response: str,
        model: str = "default",
    ) -> EvalScores:
        """
        Compute all metrics for an (original, compressed) pair.

        Args:
            original_prompt:     The raw, uncompressed prompt.
            compressed_prompt:   The compressed version.
            original_response:   LLM response to the original prompt.
            compressed_response: LLM response to the compressed prompt.
            model:               LLM model name (for cost estimation).

        Returns:
            EvalScores with all metrics populated.
        """
        scores = EvalScores(model=model)

        # --- Token counts ---
        scores.original_tokens = self._count_tokens(original_prompt)
        scores.compressed_tokens = self._count_tokens(compressed_prompt)
        scores.token_delta = scores.original_tokens - scores.compressed_tokens
        if scores.original_tokens > 0:
            scores.compression_ratio = scores.token_delta / scores.original_tokens
        else:
            scores.compression_ratio = 0.0

        # --- BERTScore ---
        b_p, b_r, b_f1 = self._bertscore_single(original_response, compressed_response)
        scores.bertscore_precision = b_p
        scores.bertscore_recall = b_r
        scores.bertscore_f1 = b_f1

        # --- ROUGE-L ---
        scores.rouge_l = self._rouge_l(original_response, compressed_response)

        # --- Quality degradation ---
        scores.quality_degradation = 1.0 - scores.bertscore_f1

        # --- Cost savings ---
        price_per_k = _TOKEN_PRICES.get(model, _TOKEN_PRICES["default"])
        scores.cost_savings_usd = (scores.token_delta / 1000) * price_per_k

        return scores

    def compute_batch(
        self,
        pairs: list[dict],
        model: str = "default",
    ) -> list[EvalScores]:
        """
        Compute metrics for a batch of pairs.

        Each dict in `pairs` should have keys:
            original_prompt, compressed_prompt,
            original_response, compressed_response
        """
        return [
            self.compute(
                original_prompt=p["original_prompt"],
                compressed_prompt=p["compressed_prompt"],
                original_response=p["original_response"],
                compressed_response=p["compressed_response"],
                model=model,
            )
            for p in pairs
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Word-level token count (fast approximation)."""
        return len(text.split())

    def _bertscore_single(
        self, reference: str, candidate: str
    ) -> tuple[float, float, float]:
        """Compute BERTScore P/R/F1 for a single pair. Returns (P, R, F1)."""
        try:
            from bert_score import score as bert_score_fn  # noqa: PLC0415

            P, R, F1 = bert_score_fn(
                cands=[candidate],
                refs=[reference],
                model_type=self.bertscore_model,
                device=self.device,
                verbose=False,
                batch_size=int(os.getenv("BERTSCORE_BATCH_SIZE", "8")),
            )
            return float(P[0]), float(R[0]), float(F1[0])
        except Exception as e:
            logger.warning("BERTScore failed: %s — returning 0.0", e)
            return 0.0, 0.0, 0.0

    def _rouge_l(self, reference: str, candidate: str) -> float:
        """Compute ROUGE-L F1 score."""
        try:
            from rouge_score import rouge_scorer  # noqa: PLC0415

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            result = scorer.score(reference, candidate)
            return result["rougeL"].fmeasure
        except ImportError:
            logger.debug("rouge_score not installed — skipping ROUGE-L")
            return 0.0
        except Exception as e:
            logger.warning("ROUGE-L failed: %s", e)
            return 0.0


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_computer: MetricsComputer | None = None


def compute_metrics(
    original_prompt: str,
    compressed_prompt: str,
    original_response: str,
    compressed_response: str,
    model: str = "default",
) -> EvalScores:
    """
    Module-level convenience wrapper around MetricsComputer.
    Reuses a shared instance (lazy-loads BERTScore only once).
    """
    global _default_computer
    if _default_computer is None:
        _default_computer = MetricsComputer()
    return _default_computer.compute(
        original_prompt=original_prompt,
        compressed_prompt=compressed_prompt,
        original_response=original_response,
        compressed_response=compressed_response,
        model=model,
    )
