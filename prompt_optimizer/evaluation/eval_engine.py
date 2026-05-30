"""
Evaluation Engine — orchestrates the full eval pipeline.

For a given (original_prompt, compressed_prompt) pair:
  1. Send both to the LLM (using cache to avoid duplicate API calls)
  2. Compute all metrics via MetricsComputer
  3. Return a structured EvalResult

Usage:
    from prompt_optimizer.evaluation.eval_engine import EvalEngine

    engine = EvalEngine()
    result = engine.evaluate(
        original_prompt="Could you please explain what photosynthesis is?",
        compressed_prompt="Explain photosynthesis.",
        task_type="qa",
    )
    print(result.scores)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from prompt_optimizer.evaluation.metrics import EvalScores, MetricsComputer
from prompt_optimizer.llm.cache import ResponseCache
from prompt_optimizer.llm.client import CompletionResult, LLMClient

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Full evaluation result for a single prompt pair."""

    original_prompt: str
    compressed_prompt: str
    task_type: str

    original_response: CompletionResult = field(default_factory=lambda: CompletionResult("", 0, 0, 0, "", "openai", 0.0))  # noqa: E501
    compressed_response: CompletionResult = field(default_factory=lambda: CompletionResult("", 0, 0, 0, "", "openai", 0.0))  # noqa: E501

    scores: EvalScores = field(default_factory=EvalScores)

    def summary(self) -> dict:
        """Flat dict suitable for logging or CSV export."""
        return {
            "task_type": self.task_type,
            "original_tokens": self.scores.original_tokens,
            "compressed_tokens": self.scores.compressed_tokens,
            "compression_ratio": round(self.scores.compression_ratio, 4),
            "token_delta": self.scores.token_delta,
            "bertscore_f1": round(self.scores.bertscore_f1, 4),
            "bertscore_precision": round(self.scores.bertscore_precision, 4),
            "bertscore_recall": round(self.scores.bertscore_recall, 4),
            "rouge_l": round(self.scores.rouge_l, 4),
            "quality_degradation": round(self.scores.quality_degradation, 4),
            "cost_savings_usd": round(self.scores.cost_savings_usd, 8),
            "original_latency_ms": round(self.original_response.latency_ms, 1),
            "compressed_latency_ms": round(self.compressed_response.latency_ms, 1),
        }


class EvalEngine:
    """
    Orchestrates the full evaluation pipeline:
      original prompt → LLM → response
      compressed prompt → LLM (cached) → response
      both responses → MetricsComputer → EvalScores

    Args:
        client:       LLMClient instance (or None to create from env).
        cache:        ResponseCache instance (or None to create default).
        metrics:      MetricsComputer instance (or None to create default).
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        cache: ResponseCache | None = None,
        metrics: MetricsComputer | None = None,
    ) -> None:
        self.client = client or LLMClient()
        self.cache = cache or ResponseCache()
        self.metrics = metrics or MetricsComputer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        original_prompt: str,
        compressed_prompt: str,
        task_type: str = "general",
        reference_response: str | None = None,
    ) -> EvalResult:
        """
        Evaluate a single (original, compressed) prompt pair.

        Args:
            original_prompt:    Verbose, uncompressed prompt.
            compressed_prompt:  Compressed version to evaluate.
            task_type:          One of qa / summarization / code / creative / general.
            reference_response: If provided, use this instead of calling the LLM
                                 for the original prompt (saves API calls).

        Returns:
            EvalResult with both responses and all computed scores.
        """
        result = EvalResult(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            task_type=task_type,
        )

        # --- Step 1: Get original response ---
        if reference_response is not None:
            # Use pre-computed reference to save API calls
            from prompt_optimizer.llm.client import CompletionResult  # noqa: PLC0415
            result.original_response = CompletionResult(
                text=reference_response,
                prompt_tokens=self.client.count_tokens(original_prompt),
                completion_tokens=self.client.count_tokens(reference_response),
                total_tokens=self.client.count_tokens(original_prompt + reference_response),
                model=self.client.model,
                provider=self.client.provider,
                latency_ms=0.0,
            )
        else:
            logger.info("Calling LLM for original prompt...")
            result.original_response = self.cache.get_or_fetch(
                prompt=original_prompt, client=self.client
            )

        # --- Step 2: Get compressed response ---
        logger.info("Calling LLM for compressed prompt...")
        result.compressed_response = self.cache.get_or_fetch(
            prompt=compressed_prompt, client=self.client
        )

        # --- Step 3: Compute metrics ---
        result.scores = self.metrics.compute(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            original_response=result.original_response.text,
            compressed_response=result.compressed_response.text,
            model=self.client.model,
        )

        return result

    def evaluate_batch(
        self,
        pairs: list[dict],
        show_progress: bool = True,
    ) -> list[EvalResult]:
        """
        Evaluate a batch of prompt pairs.

        Each dict in `pairs` should have:
            original_prompt, compressed_prompt, task_type
        Optionally: reference_response

        Args:
            pairs:         List of dicts with prompt pair data.
            show_progress: If True, show a rich progress bar.

        Returns:
            List of EvalResult objects.
        """
        results = []
        total = len(pairs)

        if show_progress:
            try:
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn  # noqa: PLC0415, E501
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task("Evaluating...", total=total)
                    for pair in pairs:
                        r = self.evaluate(
                            original_prompt=pair["original_prompt"],
                            compressed_prompt=pair["compressed_prompt"],
                            task_type=pair.get("task_type", "general"),
                            reference_response=pair.get("reference_response"),
                        )
                        results.append(r)
                        progress.advance(task)
            except ImportError:
                show_progress = False

        if not show_progress:
            for i, pair in enumerate(pairs):
                logger.info("Evaluating pair %d/%d", i + 1, total)
                r = self.evaluate(
                    original_prompt=pair["original_prompt"],
                    compressed_prompt=pair["compressed_prompt"],
                    task_type=pair.get("task_type", "general"),
                    reference_response=pair.get("reference_response"),
                )
                results.append(r)

        return results
