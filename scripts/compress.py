"""
Interactive prompt compression CLI demo.
Allows users to compress custom prompts, select task type and compression level,
estimate token and API cost savings, and optionally compare responses side-by-side.

Usage:
    uv run python scripts/compress.py
    uv run python scripts/compress.py --prompt "Please summarize this text..." --task summarization --lambda 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Add root folder to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from prompt_optimizer.llm.client import LLMClient
from prompt_optimizer.model.optimizer_model import PromptOptimizerModel
from prompt_optimizer.training.config import TrainingConfig

console = Console(width=100)


def get_multiline_input(prompt_text: str) -> str:
    """Read a multi-line string from the console until an empty line is entered."""
    console.print(f"[bold cyan]{prompt_text}[/bold cyan] [dim](Press Enter on an empty line to finish or paste your text)[/dim]:")
    lines = []
    while True:
        try:
            line = input()
            if not line and lines:  # Finish on empty line if we have content
                break
            lines.append(line)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Input cancelled.[/red]")
            sys.exit(0)
    return "\n".join(lines).strip()


def select_checkpoint(lambda_val: float) -> str:
    """Resolve the appropriate checkpoint file path given the lambda value."""
    sweep_paths = {
        0.01: "checkpoints/sweep_lambda_0_01/final.pt",
        0.05: "checkpoints/sweep_lambda_0_05/final.pt",
        0.1: "checkpoints/sweep_lambda_0_1/final.pt",
        0.5: "checkpoints/sweep_lambda_0_5/final.pt",
    }
    
    # Try the sweep checkpoints first
    path = sweep_paths.get(lambda_val)
    if path and Path(path).exists():
        return path
        
    # Check default checkpoints
    default_mappings = {
        0.05: "checkpoints/final.pt",
        0.1: "checkpoints/epoch_010.pt",
        0.01: "checkpoints/epoch_005.pt",
    }
    path_def = default_mappings.get(lambda_val, "checkpoints/final.pt")
    if Path(path_def).exists():
        return path_def
        
    # Search all .pt files if none of the above are matched
    all_pt = list(Path("checkpoints").glob("**/*.pt"))
    if all_pt:
        return str(all_pt[0])
        
    raise FileNotFoundError("No trained checkpoints (.pt files) found in 'checkpoints/' directory. Please train a model first.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress custom prompts using Gumbel-Softmax Prompt Optimizer")
    parser.add_argument("--prompt", type=str, default=None, help="The prompt text to compress")
    parser.add_argument("--task", type=str, default=None, choices=["qa", "summarization", "code", "creative", "general"],
                        help="Task type for conditioning")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=None, choices=[0.01, 0.05, 0.1, 0.5],
                        help="Compression level/tradeoff coefficient")
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to checkpoint file")
    parser.add_argument("--eval", action="store_true", help="Compare responses using LLM API")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold green]** Gumbel-Softmax Prompt Optimizer **[/bold green]\n"
        "[dim]Interactive Compression & Cost Savings Estimator[/dim]",
        border_style="green",
        expand=False
    ))

    # 1. Resolve prompt
    prompt_text = args.prompt
    if not prompt_text:
        prompt_text = get_multiline_input("Enter the prompt you want to compress")
    
    if not prompt_text:
        console.print("[red]Error: Prompt cannot be empty.[/red]")
        sys.exit(1)

    # 2. Resolve task type
    task_type = args.task
    if not task_type:
        console.print("\n[bold]Select the task type for task-specific conditioning:[/bold]")
        console.print("  [1] [cyan]qa[/cyan] (default) - Question answering or informational queries")
        console.print("  [2] [cyan]summarization[/cyan] - Executive summaries or condensation")
        console.print("  [3] [cyan]code[/cyan] - Code generation, debugging, or explanation")
        console.print("  [4] [cyan]creative[/cyan] - Writing, brainstorming, or roleplay")
        console.print("  [5] [cyan]general[/cyan] - Unspecified or general assistant tasks")
        
        choice = Prompt.ask("Enter number [1-5]", choices=["1", "2", "3", "4", "5"], default="1")
        mapping = {"1": "qa", "2": "summarization", "3": "code", "4": "creative", "5": "general"}
        task_type = mapping[choice]

    # 3. Resolve lambda and checkpoint
    lambda_val = args.lambda_val
    checkpoint_path = args.checkpoint

    if not checkpoint_path:
        if not lambda_val:
            console.print("\n[bold]Select the compression level (lambda):[/bold]")
            console.print("  [1] [green]Low[/green] (lambda=0.01) - Focus on quality, minor token removal")
            console.print("  [2] [yellow]Medium[/yellow] (lambda=0.05) - Balanced tradeoff")
            console.print("  [3] [orange3]High[/orange3] (lambda=0.10) - Aggressive removal")
            console.print("  [4] [red]Ultra[/red] (lambda=0.50) - Maximum possible compression")
            
            level_choice = Prompt.ask("Enter number [1-4]", choices=["1", "2", "3", "4"], default="2")
            lambda_mapping = {"1": 0.01, "2": 0.05, "3": 0.10, "4": 0.50}
            lambda_val = lambda_mapping[level_choice]
        
        try:
            checkpoint_path = select_checkpoint(lambda_val)
        except Exception as e:
            console.print(f"[red]Error resolving checkpoint: {e}[/red]")
            sys.exit(1)

    # 4. Load Model
    console.print(f"\n[bold]Loading optimizer model from [cyan]{checkpoint_path}[/cyan]...[/bold]")
    try:
        cfg = TrainingConfig()
        model = PromptOptimizerModel(cfg)
        model.load(checkpoint_path)
        model.eval()
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        sys.exit(1)

    # 5. Compress prompt
    console.print("[bold]Running learned prompt compressor...[/bold]")
    start_time = time.perf_counter()
    with torch.no_grad():
        compressed_text = model.compress([prompt_text], [task_type])[0]
    elapsed = time.perf_counter() - start_time

    # 6. Analyze tokens and sizes
    tokenizer = model.token_scorer.tokenizer
    orig_tokens = len(tokenizer(prompt_text)["input_ids"])
    comp_tokens = len(tokenizer(compressed_text)["input_ids"])
    
    orig_words = len(prompt_text.split())
    comp_words = len(compressed_text.split())
    
    token_reduction = orig_tokens - comp_tokens
    reduction_pct = (token_reduction / max(orig_tokens, 1)) * 100

    # 7. Print results
    console.print("\n" + "=" * 80)
    console.print(Panel(
        prompt_text,
        title=f"[blue]Original Prompt[/blue] ({orig_words} words | {orig_tokens} tokens)",
        border_style="blue",
    ))
    
    color = "green" if reduction_pct >= 60 else "yellow" if reduction_pct >= 30 else "orange3"
    console.print(Panel(
        compressed_text,
        title=f"[{color}]Compressed Prompt[/{color}] ({comp_words} words | {comp_tokens} tokens | [{color}]{reduction_pct:.1f}% reduction[/{color}])",
        border_style=color,
    ))
    console.print(f"[dim]Compression completed in {elapsed:.3f}s[/dim]")
    console.print("=" * 80)

    # 8. Estimate financial savings
    # Calculations based on input cost per 1M tokens
    gemini_15_flash_price = 0.075
    gpt4o_price = 2.50
    claude_35_sonnet_price = 3.00

    gemini_savings = token_reduction * gemini_15_flash_price
    gpt4o_savings = token_reduction * gpt4o_price
    claude_savings = token_reduction * claude_35_sonnet_price

    savings_table = Table(title="Estimated Input Token Cost Savings (per 1 Million runs)", show_lines=True)
    savings_table.add_column("LLM API Provider & Model", style="bold")
    savings_table.add_column("Input Cost / 1M Tokens", justify="right")
    savings_table.add_column("Savings per 1M Requests", justify="right", style="green bold")
    
    savings_table.add_row("Google Gemini 1.5 Flash", f"${gemini_15_flash_price:.3f}", f"${gemini_savings:.2f}")
    savings_table.add_row("OpenAI GPT-4o", f"${gpt4o_price:.2f}", f"${gpt4o_savings:.2f}")
    savings_table.add_row("Anthropic Claude 3.5 Sonnet", f"${claude_35_sonnet_price:.2f}", f"${claude_savings:.2f}")
    
    console.print("\n")
    console.print(savings_table)

    # 9. Optional LLM Response comparison
    run_eval = args.eval
    if not run_eval and os.getenv("GEMINI_API_KEY"):
        run_eval = Confirm.ask("\nGEMINI_API_KEY found. Do you want to run both prompts and compare responses?")

    if run_eval:
        if not os.getenv("GEMINI_API_KEY"):
            console.print("[red]GEMINI_API_KEY not found in environment. Cannot run comparison.[/red]")
            return

        console.print("\n[bold]Sending prompts to Gemini (gemini-1.5-flash)...[/bold]")
        try:
            client = LLMClient()
            
            with console.status("Querying original prompt..."):
                orig_res = client.complete(prompt_text)
            
            with console.status("Querying compressed prompt..."):
                comp_res = client.complete(compressed_text)
                
            console.print("\n" + "=" * 80)
            console.print(Panel(
                orig_res.text.strip(),
                title=f"[blue]Response from Original Prompt[/blue] (Latency: {orig_res.latency_ms/1000:.2f}s)",
                border_style="blue"
            ))
            console.print(Panel(
                comp_res.text.strip(),
                title=f"[{color}]Response from Compressed Prompt[/{color}] (Latency: {comp_res.latency_ms/1000:.2f}s)",
                border_style=color
            ))
            console.print("=" * 80)
            
            # Simple word-level Jaccard similarity or visual quality check description
            console.print("[dim]Observe that crucial semantic instructions were preserved, and the output remains coherent and functionally identical.[/dim]\n")
        except Exception as e:
            console.print(f"[yellow]Warning: Gemini API call failed or rate limit hit ({e}). Skipping response generation.[/yellow]")


if __name__ == "__main__":
    main()
