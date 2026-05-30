"""
Model inspection script — shows what the trained model actually compresses.

Loads the trained checkpoint and compresses sample prompts side-by-side.
Also runs Gemini eval if GEMINI_API_KEY is set.

Usage:
    uv run python scripts/inspect.py
    uv run python scripts/inspect.py --checkpoint checkpoints/epoch_010.pt
    uv run python scripts/inspect.py --no-llm   # skip Gemini, show compression only
    uv run python scripts/inspect.py --task qa  # filter by task type
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(width=100)


def load_model(checkpoint_path: str):
    """Load the trained PromptOptimizerModel from a checkpoint."""
    from prompt_optimizer.training.config import TrainingConfig
    from prompt_optimizer.model.optimizer_model import PromptOptimizerModel

    cfg = TrainingConfig()
    model = PromptOptimizerModel(cfg)
    model.load(checkpoint_path)
    model.eval()
    return model, cfg


def compression_stats(original: str, compressed: str) -> dict:
    orig_words = len(original.split())
    comp_words = len(compressed.split())
    ratio = (orig_words - comp_words) / max(orig_words, 1)
    return {
        "orig_words": orig_words,
        "comp_words": comp_words,
        "reduction_pct": ratio * 100,
        "kept_pct": (1 - ratio) * 100,
    }


def run_inspection(
    checkpoint: str,
    task_filter: str | None = None,
    use_llm: bool = True,
    n_samples: int = 8,
) -> None:
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Prompt Optimizer — Model Inspection[/bold cyan]\n"
        f"[dim]Checkpoint: {checkpoint}[/dim]"
    ))

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    console.print("\n[bold]Loading model...[/bold]")
    model, cfg = load_model(checkpoint)
    console.print(f"  Trainable params: [green]{model.trainable_params:,}[/green]")
    console.print(f"  Gumbel tau end:   [green]{cfg.gumbel_tau_end}[/green]")
    console.print(f"  Lambda:           [green]{cfg.lambda_}[/green]")

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    from prompt_optimizer.dataset.curate import PromptDataset
    ds = PromptDataset.from_jsonl("data/prompts.jsonl")

    if task_filter:
        ds = ds.by_task(task_filter)
        console.print(f"\nFiltered to task=[cyan]{task_filter}[/cyan]: {len(ds)} prompts")

    samples = list(ds)[:n_samples]

    # ----------------------------------------------------------------
    # Compress all samples
    # ----------------------------------------------------------------
    console.print(f"\n[bold]Compressing {len(samples)} prompts...[/bold]")
    prompts = [p.prompt for p in samples]
    task_types = [p.task for p in samples]

    import torch
    with torch.no_grad():
        compressed_texts = model.compress(prompts, task_types)

    # ----------------------------------------------------------------
    # Display side-by-side + stats table
    # ----------------------------------------------------------------
    all_stats = []

    for i, (prompt_obj, compressed) in enumerate(zip(samples, compressed_texts)):
        stats = compression_stats(prompt_obj.prompt, compressed)
        all_stats.append({**stats, "task": prompt_obj.task, "id": prompt_obj.id})

        console.print(f"\n[bold yellow]── Sample {i+1}/{len(samples)} [{prompt_obj.task}] {prompt_obj.id} ──[/bold yellow]")

        # Color the reduction number
        pct = stats["reduction_pct"]
        color = "green" if pct >= 30 else "yellow" if pct >= 10 else "red"

        console.print(Panel(
            prompt_obj.prompt,
            title=f"[blue]ORIGINAL[/blue] ({stats['orig_words']} words)",
            border_style="blue",
        ))

        if compressed.strip():
            console.print(Panel(
                compressed,
                title=f"[{color}]COMPRESSED[/{color}] ({stats['comp_words']} words | [{color}]{pct:.1f}% reduction[/{color}])",
                border_style=color,
            ))
        else:
            console.print(Panel(
                "[red italic]⚠ Empty output — model dropped all tokens![/red italic]",
                title="[red]COMPRESSED[/red]",
                border_style="red",
            ))

    # ----------------------------------------------------------------
    # Summary stats table
    # ----------------------------------------------------------------
    console.print()
    table = Table(title="Compression Summary", show_lines=True)
    table.add_column("ID", style="dim")
    table.add_column("Task", style="cyan")
    table.add_column("Orig Words", justify="right")
    table.add_column("Comp Words", justify="right")
    table.add_column("Reduction", justify="right", style="green")
    table.add_column("Quality", justify="right")  # filled if LLM eval runs

    for s in all_stats:
        table.add_row(
            s["id"],
            s["task"],
            str(s["orig_words"]),
            str(s["comp_words"]),
            f"{s['reduction_pct']:.1f}%",
            "—",  # will fill with Gemini eval below
        )

    console.print(table)

    avg_reduction = sum(s["reduction_pct"] for s in all_stats) / max(len(all_stats), 1)
    console.print(f"\n[bold]Average compression: [green]{avg_reduction:.1f}%[/green] reduction[/bold]")

    # ----------------------------------------------------------------
    # Optional: Gemini true quality eval
    # ----------------------------------------------------------------
    if use_llm:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or "your-key" in api_key:
            console.print(
                "\n[yellow]No GEMINI_API_KEY set — skipping true quality evaluation.[/yellow]\n"
                "Add your key to .env to enable BERTScore evaluation against Gemini responses."
            )
            return

        console.print("\n[bold]Running Gemini quality evaluation (this takes ~60s)...[/bold]")
        console.print(
            "[dim]Tip: Free tier = 5 req/min. We pause 15s between calls to avoid hitting it.[/dim]"
        )

        # Create engine ONCE — BERTScore model (1.4GB) loads only on first call then cached
        from prompt_optimizer.evaluation.eval_engine import EvalEngine
        engine = EvalEngine()

        results = []
        eval_samples = samples[:4]  # limit to 4 to save quota
        for idx, (prompt_obj, compressed) in enumerate(zip(eval_samples, compressed_texts[:4])):
            if not compressed.strip():
                continue
            try:
                with console.status(f"Evaluating {prompt_obj.id} ({idx+1}/{len(eval_samples)})..."):
                    result = engine.evaluate(
                        original_prompt=prompt_obj.prompt,
                        compressed_prompt=compressed,
                        task_type=prompt_obj.task,
                    )
                results.append((prompt_obj.id, result))
                console.print(
                    f"  [green]✓[/green] {prompt_obj.id}: "
                    f"BERTScore F1=[bold green]{result.scores.bertscore_f1:.4f}[/bold green] | "
                    f"Compression=[cyan]{result.scores.compression_ratio:.1%}[/cyan]"
                )
                # Free tier rate limit: 5 requests/min → wait 15s between calls
                if idx < len(eval_samples) - 1:
                    import time as _time  # noqa: PLC0415
                    console.print(f"  [dim]Waiting 15s (free tier rate limit)...[/dim]")
                    _time.sleep(15)
            except Exception as e:
                err = str(e)
                if "quota" in err.lower() or "429" in err:
                    console.print(
                        f"  [red]Rate limit hit on {prompt_obj.id}[/red] — "
                        "wait 60s then re-run, or upgrade to paid tier."
                    )
                else:
                    console.print(f"  [red]Error on {prompt_obj.id}: {e}[/red]")

        if results:
            avg_bs = sum(r.scores.bertscore_f1 for _, r in results) / len(results)
            avg_comp = sum(r.scores.compression_ratio for _, r in results) / len(results)

            # Interpret the score
            if avg_bs >= 0.90:
                quality_label = "[bold green]EXCELLENT[/bold green]"
            elif avg_bs >= 0.85:
                quality_label = "[bold yellow]GOOD[/bold yellow]"
            else:
                quality_label = "[bold red]NEEDS IMPROVEMENT[/bold red]"

            console.print(f"\n[bold]True Eval Summary ({len(results)} samples):[/bold]")
            console.print(f"  BERTScore F1:    [bold green]{avg_bs:.4f}[/bold green] — {quality_label}")
            console.print(f"  Compression:     [cyan]{avg_comp:.1%}[/cyan] tokens removed")
            console.print(
                f"\n  [dim]Thresholds: >0.90 = Excellent | >0.85 = Good | <0.85 = Needs work[/dim]"
            )

            # Save results
            Path("results").mkdir(exist_ok=True)
            save_data = {
                "summary": {
                    "avg_bertscore_f1": round(avg_bs, 4),
                    "avg_compression": round(avg_comp, 4),
                    "n_samples": len(results),
                    "quality_label": "excellent" if avg_bs >= 0.90 else "good" if avg_bs >= 0.85 else "needs_improvement",
                },
                "samples": [
                    {
                        "id": pid,
                        "bertscore_f1": round(r.scores.bertscore_f1, 4),
                        "compression_ratio": round(r.scores.compression_ratio, 4),
                        "original_tokens": r.scores.original_tokens,
                        "compressed_tokens": r.scores.compressed_tokens,
                    }
                    for pid, r in results
                ],
            }
            with open("results/gemini_eval.json", "w") as f:
                json.dump(save_data, f, indent=2)
            console.print("  [dim]Saved to results/gemini_eval.json[/dim]")



def main():
    parser = argparse.ArgumentParser(description="Inspect trained model compressions")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--task", default=None, help="Filter: qa / summarization / code / creative")
    parser.add_argument("--no-llm", action="store_true", help="Skip Gemini eval")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    run_inspection(
        checkpoint=args.checkpoint,
        task_filter=args.task,
        use_llm=not args.no_llm,
        n_samples=args.samples,
    )


if __name__ == "__main__":
    main()
