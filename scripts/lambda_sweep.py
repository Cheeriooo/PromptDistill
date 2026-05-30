"""
Phase 4 Lambda Sweep.

Sweeps lambda hyperparameter ∈ [0.01, 0.05, 0.1, 0.3, 0.6], trains a model
for each value, and runs final LLM evaluation to collect (compression, quality) metrics.
Saves the results to results/lambda_sweep.json.

Usage:
    uv run python scripts/lambda_sweep.py --epochs 10 --eval-size 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prompt_optimizer.dataset.curate import PromptDataset
from prompt_optimizer.evaluation.eval_engine import EvalEngine
from prompt_optimizer.model.optimizer_model import PromptOptimizerModel
from prompt_optimizer.training.config import TrainingConfig
from prompt_optimizer.training.trainer import Trainer

console = Console(width=100)


def run_sweep(epochs: int, eval_size: int) -> None:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or "your-key" in api_key:
        console.print("[red]Error: GEMINI_API_KEY not found in environment. LLM evaluation requires a valid API key.[/red]")
        sys.exit(1)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Phase 4 — Hyperparameter Lambda Sweep[/bold cyan]\n"
        "[dim]Sweeping lambda to construct the Pareto frontier[/dim]"
    ))

    # Lambda values to sweep
    lambdas = [0.01, 0.05, 0.1, 0.5]
    
    # Load dataset to split same way
    dataset = PromptDataset.from_jsonl("data/prompts.jsonl")
    _, val_ds, _ = dataset.split(train=0.6, val=0.2, seed=42)
    eval_prompts = list(val_ds)[:eval_size]

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Sweeping lambdas:   {lambdas}")
    console.print(f"  Training epochs:    {epochs}")
    console.print(f"  Evaluation prompts: {len(eval_prompts)} prompts")

    # Initialize EvalEngine ONCE (caches BERTScore model)
    engine = EvalEngine()
    
    sweep_results = []

    for l_val in lambdas:
        console.print(f"\n[bold yellow]============================================================[/bold yellow]")
        console.print(f"[bold yellow]Sweep Step: Lambda = {l_val}[/bold yellow]")
        console.print(f"[bold yellow]============================================================[/bold yellow]")

        # 1. Setup training config for this lambda
        chk_dir = f"checkpoints/sweep_lambda_{str(l_val).replace('.', '_')}"
        cfg = TrainingConfig(
            lambda_coeff=l_val,
            num_epochs=epochs,
            checkpoint_dir=chk_dir,
            llm_eval_every_n_steps=999999,  # Disable LLM evaluation during training for speed
            eval_every_n_steps=100,
            device="auto",
        )

        # 2. Train model
        trainer = Trainer(cfg)
        console.print(f"  Training model for lambda={l_val}...")
        trainer.train(data_path="data/prompts.jsonl")

        # 3. Load trained model for eval
        model = PromptOptimizerModel(cfg)
        ckpt_path = f"{chk_dir}/final.pt"
        if os.path.exists(ckpt_path):
            model.load(ckpt_path)
        else:
            console.print(f"[red]Error: Checkpoint {ckpt_path} not found. Skipping evaluation.[/red]")
            continue
        model.to(cfg.device)
        model.eval()

        # 4. Evaluate on validation subset
        console.print(f"  Evaluating model on {len(eval_prompts)} prompts using Gemini...")
        
        raw_prompts = [p.prompt for p in eval_prompts]
        task_types = [p.task for p in eval_prompts]
        
        with torch.no_grad():
            compressed_texts = model.compress(raw_prompts, task_types)

        run_scores = []
        for idx, (prompt_obj, compressed) in enumerate(zip(eval_prompts, compressed_texts)):
            if not compressed.strip():
                # Guaranteed token selection fallback should prevent this, but handle just in case
                compressed = "Fallback"
            
            with console.status(f"Evaluating prompt {idx+1}/{len(eval_prompts)}..."):
                res = engine.evaluate(
                    original_prompt=prompt_obj.prompt,
                    compressed_prompt=compressed,
                    task_type=prompt_obj.task,
                )
            
            run_scores.append(res.scores)
            
            console.print(f"    - {prompt_obj.id}: F1={res.scores.bertscore_f1:.4f} | Comp={res.scores.compression_ratio:.1%}")
            
            # Rate limiting sleep (15s for free tier)
            if idx < len(eval_prompts) - 1:
                time.sleep(15)

        # 5. Aggregate metrics
        avg_f1 = sum(s.bertscore_f1 for s in run_scores) / len(run_scores)
        avg_comp = sum(s.compression_ratio for s in run_scores) / len(run_scores)
        avg_orig_tok = sum(s.original_tokens for s in run_scores) / len(run_scores)
        avg_comp_tok = sum(s.compressed_tokens for s in run_scores) / len(run_scores)
        
        console.print(f"\n[green]Lambda = {l_val} Summary:[/green]")
        console.print(f"  Average BERTScore F1: {avg_f1:.4f}")
        console.print(f"  Average Compression:  {avg_comp:.1%}")

        sweep_results.append({
            "lambda": l_val,
            "avg_bertscore_f1": round(avg_f1, 4),
            "avg_compression": round(avg_comp, 4),
            "avg_original_tokens": round(avg_orig_tok, 1),
            "avg_compressed_tokens": round(avg_comp_tok, 1),
            "checkpoint_dir": chk_dir,
        })

    # Save results
    Path("results").mkdir(exist_ok=True)
    out_path = "results/lambda_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    console.print(f"\n[bold green]✓ Lambda sweep complete! Saved results to {out_path}[/bold green]\n")

    # Present summary table
    table = Table(title="Lambda Sweep Summary", show_lines=True)
    table.add_column("Lambda", justify="right", style="cyan")
    table.add_column("Avg Orig Tokens", justify="right")
    table.add_column("Avg Comp Tokens", justify="right")
    table.add_column("Avg Compression Ratio", justify="right", style="green")
    table.add_column("Avg BERTScore F1", justify="right", style="bold green")

    for r in sweep_results:
        table.add_row(
            str(r["lambda"]),
            str(r["avg_original_tokens"]),
            str(r["avg_compressed_tokens"]),
            f"{r['avg_compression']:.1%}",
            f"{r['avg_bertscore_f1']:.4f}",
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Sweep lambda and train multiple models")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per lambda")
    parser.add_argument("--eval-size", type=int, default=5, help="Number of prompts to evaluate per lambda")
    args = parser.parse_args()

    run_sweep(epochs=args.epochs, eval_size=args.eval_size)


if __name__ == "__main__":
    main()
