"""
Phase 3 Few-Shot Task Adaptation Script.

This script demonstrates few-shot adaptation of the Prompt Optimizer model
to an unseen task type using a small support set (5-10 prompts).
It prints the compression behavior on an unseen query prompt before and after adaptation,
and evaluates the quality of both compressed prompts using Gemini (if api key is available).

Usage:
    uv run python scripts/adapt_to_task.py --task translation
    uv run python scripts/adapt_to_task.py --task sql_generation --epochs 15
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
import torch.optim as optim
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prompt_optimizer.model.optimizer_model import PromptOptimizerModel
from prompt_optimizer.training.config import TrainingConfig
from prompt_optimizer.training.losses import proxy_loss

console = Console(width=100)

# Pre-packaged few-shot support datasets
DEFAULT_SUPPORT_PROMPTS = {
    "translation": [
        "Translate the following English sentence to French, ensuring natural flow: 'Artificial intelligence is changing the world very quickly.'",
        "Could you please translate this paragraph to Spanish for me: 'Learning a new language opens up many opportunities and allows you to understand different cultures.'",
        "Please translate the text below to German, keeping it formal: 'Please review the attached contract and let me know if you have any questions or concerns.'",
        "Translate the following sentence to Italian, keeping it simple and informal: 'Where is the nearest train station and what time does the next train arrive?'",
        "Can you translate this quote to Japanese, keeping it polite: 'Success is not final, failure is not fatal: it is the courage to continue that counts.'"
    ],
    "sql_generation": [
        "Write a SQL query to select all columns from the users table where the age is greater than 18 and sort the results by name.",
        "Could you generate a SQL query to join the orders and customers tables on customer_id, filtering for orders in the last 30 days?",
        "Please write a SQL database query to calculate the average salary of employees in each department from the employees table.",
        "I need a SQL query to insert a new row into the products table with name 'Widget', price 9.99, and stock 100.",
        "Generate a SQL query to delete all records from the logs table where the timestamp is older than 6 months."
    ]
}

DEFAULT_QUERY_PROMPTS = {
    "translation": "Please translate the following text into standard Chinese, and make sure that the meaning is kept exact: 'The early bird catches the worm.'",
    "sql_generation": "Can you please write a SQL query to find the top 5 highest paying customers from the orders table grouped by customer_id and sorted in descending order?"
}


def run_adaptation(
    checkpoint_path: str,
    task_name: str,
    epochs: int,
    lr: float,
    lambda_: float,
    save_to: str | None,
    use_llm: bool,
) -> None:
    # Validate task
    if task_name not in DEFAULT_SUPPORT_PROMPTS:
        console.print(f"[red]Error: Unsupported task '{task_name}'. Available default tasks: {list(DEFAULT_SUPPORT_PROMPTS.keys())}[/red]")
        sys.exit(1)

    console.print()
    console.print(Panel.fit(
        f"[bold magenta]Phase 3 — Few-Shot Task Adaptation[/bold magenta]\n"
        f"[dim]Adapting to unseen task type: {task_name}[/dim]"
    ))

    # 1. Load configuration and model
    cfg = TrainingConfig(device="auto")
    # Override lambda if specified
    if lambda_ is not None:
        cfg.lambda_ = lambda_

    console.print("\n[bold]Loading base model...[/bold]")
    model = PromptOptimizerModel(cfg)
    
    if os.path.exists(checkpoint_path):
        model.load(checkpoint_path)
    else:
        console.print(f"[yellow]Warning: Base checkpoint '{checkpoint_path}' not found. Initializing from scratch for adaptation.[/yellow]")
    
    model.to(cfg.device)

    # 2. Get support prompts and query prompt
    support_prompts = DEFAULT_SUPPORT_PROMPTS[task_name]
    query_prompt = DEFAULT_QUERY_PROMPTS[task_name]

    console.print(f"\n[bold]Support prompts for few-shot learning ({len(support_prompts)} samples):[/bold]")
    for i, p in enumerate(support_prompts):
        console.print(f"  {i+1}. {p[:80]}...")

    console.print(f"\n[bold]Unseen test/query prompt:[/bold]\n  [cyan]\"{query_prompt}\"[/cyan]")

    # 3. Compress before adaptation (for baseline comparison)
    model.eval()
    with torch.no_grad():
        compressed_before = model.compress([query_prompt], ["general"])[0]

    console.print(f"\n[bold yellow]Compression BEFORE adaptation:[/bold yellow]")
    console.print(f"  [dim]{compressed_before}[/dim]")
    console.print(f"  Tokens: {len(query_prompt.split())} → {len(compressed_before.split())} "
                  f"({(len(query_prompt.split()) - len(compressed_before.split())) / len(query_prompt.split()) * 100:.1f}% reduction)")

    # 4. Adaptation Training Loop
    console.print(f"\n[bold]Running adaptation training for {epochs} epochs on device '{cfg.device}'...[/bold]")
    
    # Optimizer (only train task embedding and token scorer MLP, frozen base model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    
    model.train()
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.zero_grad()
        
        # Forward pass on support prompts
        # Map unseen task type to "general"
        task_types = ["general"] * len(support_prompts)
        out = model(support_prompts, task_types, hard=False)
        
        # Compute proxy loss
        loss, metrics = proxy_loss(
            out,
            lambda_=cfg.lambda_,
            min_keep_ratio=0.20,
            target_keep_ratio=0.40,  # Push for good compression during adaptation
            emptiness_penalty_weight=5.0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.grad_clip)
        optimizer.step()
        
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            console.print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Sim: {metrics['similarity']:.4f} | "
                f"Comp: {metrics['compression_ratio']:.1%}"
            )
            
    elapsed = time.time() - start_time
    console.print(f"[green]✓ Adaptation completed in {elapsed:.2f}s[/green]")

    # 5. Compress after adaptation
    model.eval()
    with torch.no_grad():
        compressed_after = model.compress([query_prompt], ["general"])[0]

    console.print(f"\n[bold green]Compression AFTER adaptation:[/bold green]")
    console.print(f"  [green]{compressed_after}[/green]")
    console.print(f"  Tokens: {len(query_prompt.split())} → {len(compressed_after.split())} "
                  f"({(len(query_prompt.split()) - len(compressed_after.split())) / len(query_prompt.split()) * 100:.1f}% reduction)")

    # Save adapted weights
    if save_to:
        model.save(save_to)
        console.print(f"  [dim]Saved adapted model weights to {save_to}[/dim]")

    # 6. Optional: True LLM evaluation with Gemini
    api_key = os.getenv("GEMINI_API_KEY", "")
    if use_llm and api_key and "your-key" not in api_key:
        console.print("\n[bold]Running Gemini evaluation to verify quality preservation...[/bold]")
        try:
            from prompt_optimizer.evaluation.eval_engine import EvalEngine
            engine = EvalEngine()
            
            with console.status("Querying Gemini on original prompt..."):
                res_orig = engine.client.complete(query_prompt)
                
            with console.status("Querying Gemini on BEFORE-adaptation prompt..."):
                res_before = engine.client.complete(compressed_before)
                
            with console.status("Querying Gemini on AFTER-adaptation prompt..."):
                res_after = engine.client.complete(compressed_after)

            # Compute BERTScores
            # We instantiate metrics calculator once
            from prompt_optimizer.evaluation.metrics import MetricsComputer
            metrics_comp = MetricsComputer()
            
            with console.status("Computing semantic similarities..."):
                metrics_before = metrics_comp.compute(
                    original_prompt=query_prompt,
                    compressed_prompt=compressed_before,
                    original_response=res_orig.text,
                    compressed_response=res_before.text
                )
                metrics_after = metrics_comp.compute(
                    original_prompt=query_prompt,
                    compressed_prompt=compressed_after,
                    original_response=res_orig.text,
                    compressed_response=res_after.text
                )

            # Display side by side responses
            console.print("\n[bold]Gemini Responses & Semantic Quality Comparison:[/bold]")
            
            table = Table(show_lines=True)
            table.add_column("Version", style="bold")
            table.add_column("Prompt Words", justify="right")
            table.add_column("Gemini Response Sample", width=45)
            table.add_column("BERTScore F1", justify="right", style="green")
            
            table.add_row(
                "Original",
                str(len(query_prompt.split())),
                res_orig.text.strip().replace("\n", " ")[:120] + "...",
                "1.0000"
            )
            table.add_row(
                "Compressed (Before)",
                str(len(compressed_before.split())),
                res_before.text.strip().replace("\n", " ")[:120] + "...",
                f"{metrics_before.bertscore_f1:.4f}"
            )
            table.add_row(
                "Compressed (Adapted)",
                str(len(compressed_after.split())),
                res_after.text.strip().replace("\n", " ")[:120] + "...",
                f"{metrics_after.bertscore_f1:.4f}"
            )
            console.print(table)
            
            # Print interpretation
            diff = metrics_after.bertscore_f1 - metrics_before.bertscore_f1
            if diff > 0.001:
                console.print(f"\n[green]★ Adaptation increased semantic response similarity by +{diff:.4f}![/green]")
            elif abs(diff) <= 0.001:
                console.print("\n[yellow]★ Adaptation maintained similar response quality with adapted token compression.[/yellow]")
            else:
                console.print(f"\n[red]★ Adaptation response similarity changed by {diff:.4f}[/red]")

        except Exception as e:
            console.print(f"[red]Error during LLM evaluation: {e}[/red]")
    else:
        console.print("\n[yellow]Skipping Gemini LLM evaluation (use_llm=False or GEMINI_API_KEY not set).[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Few-shot adapt Prompt Optimizer to new tasks")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt", help="Path to base model checkpoint")
    parser.add_argument("--task", default="translation", choices=list(DEFAULT_SUPPORT_PROMPTS.keys()), help="Unseen task to adapt to")
    parser.add_argument("--epochs", type=int, default=10, help="Number of adaptation epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adaptation")
    parser.add_argument("--lambda-coeff", type=float, default=None, help="Compression coefficient (None to keep base value)")
    parser.add_argument("--save-to", default=None, help="Path to save adapted weights. Defaults to checkpoints/adapted_<task>.pt")
    parser.add_argument("--no-llm", action="store_true", help="Skip Gemini response quality evaluation")
    args = parser.parse_args()

    save_path = args.save_to or f"checkpoints/adapted_{args.task}.pt"

    run_adaptation(
        checkpoint_path=args.checkpoint,
        task_name=args.task,
        epochs=args.epochs,
        lr=args.lr,
        lambda_=args.lambda_coeff,
        save_to=save_path,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()
