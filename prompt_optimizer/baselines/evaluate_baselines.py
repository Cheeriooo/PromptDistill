"""
Baseline Evaluation Runner.

Evaluates all 4 baseline methods on the dataset and outputs:
- Per-baseline compression statistics (no LLM calls needed)
- CSV file with results
- Rich console table

Usage:
    uv run python -m prompt_optimizer.baselines.evaluate_baselines
    uv run python -m prompt_optimizer.baselines.evaluate_baselines --data data/prompts.jsonl
    uv run python -m prompt_optimizer.baselines.evaluate_baselines --task qa
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompt_optimizer.baselines.base import BaseCompressor, CompressionResult
from prompt_optimizer.baselines.identity import IdentityCompressor
from prompt_optimizer.baselines.sentence_compression import SentenceCompressionCompressor
from prompt_optimizer.baselines.stopword_removal import StopwordRemovalCompressor
from prompt_optimizer.baselines.tfidf_selector import TFIDFCompressor
from prompt_optimizer.dataset.curate import Prompt, PromptDataset

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def build_baselines(task_type: str = "general") -> list[BaseCompressor]:
    """Instantiate all baseline compressors."""
    return [
        IdentityCompressor(),
        StopwordRemovalCompressor(),
        TFIDFCompressor(keep_ratio=0.6),
        SentenceCompressionCompressor(task_type=task_type),
    ]


def evaluate_baseline_on_dataset(
    baseline: BaseCompressor,
    dataset: PromptDataset,
) -> list[dict[str, Any]]:
    """
    Run a single baseline on every prompt in the dataset.
    Returns a list of result dicts (one per prompt).
    """
    results = []
    for prompt in dataset:
        cr: CompressionResult = baseline.compress_result(prompt.prompt)
        results.append({
            "id": prompt.id,
            "task": prompt.task,
            "method": baseline.name,
            "original_tokens": cr.original_tokens,
            "compressed_tokens": cr.compressed_tokens,
            "compression_ratio": round(cr.compression_ratio, 4),
            "token_delta": cr.original_tokens - cr.compressed_tokens,
            "original_prompt": prompt.prompt,
            "compressed_prompt": cr.compressed,
        })
    return results


def aggregate_results(results: list[dict]) -> dict[str, Any]:
    """Compute mean statistics across a list of result dicts."""
    if not results:
        return {}
    n = len(results)
    return {
        "count": n,
        "avg_original_tokens": round(sum(r["original_tokens"] for r in results) / n, 1),
        "avg_compressed_tokens": round(sum(r["compressed_tokens"] for r in results) / n, 1),
        "avg_compression_ratio": round(sum(r["compression_ratio"] for r in results) / n, 4),
        "avg_token_delta": round(sum(r["token_delta"] for r in results) / n, 1),
    }


def save_csv(all_results: list[dict], path: Path) -> None:
    """Save all results to a CSV file."""
    if not all_results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_results[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✅ Results saved to {path}")


def print_summary_table(summary: dict[str, dict]) -> None:
    """Print a rich summary table comparing all baselines."""
    try:
        from rich.table import Table  # noqa: PLC0415
        from rich.console import Console  # noqa: PLC0415

        console = Console()
        table = Table(title="📊 Baseline Comparison (Compression Metrics)", show_lines=True)
        table.add_column("Method", style="cyan", min_width=22)
        table.add_column("Avg Original\nTokens", justify="right")
        table.add_column("Avg Compressed\nTokens", justify="right")
        table.add_column("Avg Compression\nRatio", justify="right", style="green")
        table.add_column("Avg Token\nDelta", justify="right")
        table.add_column("Count", justify="right")

        for method, agg in summary.items():
            table.add_row(
                method,
                str(agg.get("avg_original_tokens", "—")),
                str(agg.get("avg_compressed_tokens", "—")),
                f"{agg.get('avg_compression_ratio', 0):.1%}",
                str(agg.get("avg_token_delta", "—")),
                str(agg.get("count", "—")),
            )

        console.print()
        console.print(table)
        console.print()
        console.print(
            "[dim]Note: Quality scores (BERTScore) require LLM API calls.\n"
            "Run with --with-llm flag to include quality evaluation.[/dim]"
        )
    except ImportError:
        # Fallback without rich
        print("\n=== Baseline Comparison ===")
        for method, agg in summary.items():
            print(f"{method}: {agg}")


def print_sample_compressions(
    dataset: PromptDataset,
    baselines: list[BaseCompressor],
    n_samples: int = 2,
) -> None:
    """Show sample compressions for quick visual inspection."""
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415

        console = Console()
        samples = list(dataset)[:n_samples]

        for prompt_obj in samples:
            console.print(f"\n[bold yellow]Prompt ID:[/bold yellow] {prompt_obj.id} [{prompt_obj.task}]")
            console.print(Panel(prompt_obj.prompt, title="Original", border_style="blue"))
            for baseline in baselines:
                cr = baseline.compress_result(prompt_obj.prompt)
                console.print(
                    Panel(
                        cr.compressed,
                        title=f"{baseline.name} ({cr.compression_ratio:.1%} reduction)",
                        border_style="green" if cr.compression_ratio > 0 else "dim",
                    )
                )
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate compression baselines")
    parser.add_argument(
        "--data",
        default="data/prompts.jsonl",
        help="Path to prompts JSONL file",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Filter to a specific task type (qa / summarization / code / creative)",
    )
    parser.add_argument(
        "--output",
        default="results/baseline_comparison.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of sample compressions to show",
    )
    args = parser.parse_args()

    # --- Load dataset ---
    dataset = PromptDataset.from_jsonl(args.data)
    if args.task:
        dataset = dataset.by_task(args.task)
        print(f"Filtered to task={args.task}: {len(dataset)} prompts")

    dataset.print_stats()

    # --- Build baselines ---
    baselines = build_baselines(task_type=args.task or "general")

    # --- Fit TF-IDF on full corpus ---
    corpus = [p.prompt for p in dataset]
    tfidf_baseline = next(b for b in baselines if b.name == "tfidf_selector")
    tfidf_baseline.fit(corpus)  # type: ignore[attr-defined]

    # --- Evaluate ---
    all_results: list[dict] = []
    summary: dict[str, dict] = {}

    for baseline in baselines:
        print(f"\n⚡ Evaluating: {baseline.name}...")
        results = evaluate_baseline_on_dataset(baseline, dataset)
        all_results.extend(results)
        summary[baseline.name] = aggregate_results(results)

    # --- Show sample compressions ---
    print_sample_compressions(dataset, baselines, n_samples=args.samples)

    # --- Print summary table ---
    print_summary_table(summary)

    # --- Save results ---
    save_csv(all_results, Path(args.output))

    # Also save summary JSON
    summary_path = Path(args.output).parent / "baseline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
