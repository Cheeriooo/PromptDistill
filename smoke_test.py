"""
# -*- coding: utf-8 -*-
Phase 0 Smoke Test — end-to-end validation.

Verifies the full pipeline works:
  LLM client → cache → eval engine → metrics

Run with:
    uv run python smoke_test.py
    uv run python smoke_test.py --no-llm   # skip actual API call, use mock
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import io
import sys
# Force UTF-8 stdout on Windows to handle unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

console = Console()
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------
ORIGINAL_PROMPT = (
    "Could you please help me understand what gradient descent is "
    "and how it is commonly used in the field of machine learning?"
)
COMPRESSED_PROMPT = "Explain gradient descent and its role in machine learning."

TASK_TYPE = "qa"


def run_smoke_test(use_mock: bool = False) -> bool:
    """
    Run the smoke test.

    Args:
        use_mock: If True, skip real API call — uses a hardcoded mock response.

    Returns:
        True if all checks pass.
    """
    console.print(Panel.fit("[bold cyan]Prompt Optimizer -- Smoke Test[/bold cyan]"))
    passed = True

    # -----------------------------------------------------------------------
    # Test 1: LLM Client instantiation
    # -----------------------------------------------------------------------
    console.print("\n[bold]Test 1:[/bold] LLM Client initialization...")
    try:
        from prompt_optimizer.llm.client import LLMClient
        client = LLMClient()
        console.print(f"  ✅ Client created | provider={client.provider} model={client.model}")
    except Exception as e:
        console.print(f"  ❌ Failed: {e}")
        passed = False
        return passed

    # -----------------------------------------------------------------------
    # Test 2: Token counting
    # -----------------------------------------------------------------------
    console.print("\n[bold]Test 2:[/bold] Token counting...")
    try:
        original_tokens = client.count_tokens(ORIGINAL_PROMPT)
        compressed_tokens = client.count_tokens(COMPRESSED_PROMPT)
        console.print(f"  ✅ Original: {original_tokens} tokens | Compressed: {compressed_tokens} tokens")
    except Exception as e:
        console.print(f"  ❌ Failed: {e}")
        passed = False

    # -----------------------------------------------------------------------
    # Test 3: Response cache
    # -----------------------------------------------------------------------
    console.print("\n[bold]Test 3:[/bold] Response cache...")
    try:
        from prompt_optimizer.llm.cache import ResponseCache
        cache = ResponseCache(cache_dir=".cache/smoke_test")
        initial_size = cache.size
        console.print(f"  ✅ Cache initialized | {initial_size} existing entries")
    except Exception as e:
        console.print(f"  ❌ Failed: {e}")
        passed = False

    # -----------------------------------------------------------------------
    # Test 4: LLM call (real or mock)
    # -----------------------------------------------------------------------
    console.print("\n[bold]Test 4:[/bold] LLM call...")
    if use_mock:
        console.print("  ⚡ [yellow]Mock mode[/yellow] — skipping real API call")
        original_response_text = (
            "Gradient descent is an optimization algorithm used in machine learning "
            "to minimize a loss function by iteratively moving in the direction of "
            "the steepest descent."
        )
        compressed_response_text = (
            "Gradient descent minimizes a loss function by iteratively adjusting "
            "parameters in the direction of the negative gradient."
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("sk-..."):
            console.print(
                "  ⚠️  [yellow]No API key found in .env — switching to mock mode.[/yellow]\n"
                "  Copy .env.example → .env and fill in your API key to test real calls."
            )
            use_mock = True
            original_response_text = "Gradient descent is an optimization algorithm."
            compressed_response_text = "Gradient descent minimizes loss functions."
        else:
            try:
                result_original = cache.get_or_fetch(prompt=ORIGINAL_PROMPT, client=client)
                result_compressed = cache.get_or_fetch(prompt=COMPRESSED_PROMPT, client=client)
                original_response_text = result_original.text
                compressed_response_text = result_compressed.text
                console.print(
                    f"  ✅ Responses received | "
                    f"original={result_original.total_tokens} tokens | "
                    f"compressed={result_compressed.total_tokens} tokens | "
                    f"latency={result_original.latency_ms:.0f}ms"
                )
            except Exception as e:
                console.print(f"  ❌ LLM call failed: {e}")
                passed = False
                original_response_text = ""
                compressed_response_text = ""

    # -----------------------------------------------------------------------
    # Test 5: Metrics computation
    # -----------------------------------------------------------------------
    console.print("\n[bold]Test 5:[/bold] Metrics computation (skipping BERTScore for speed)...")
    try:
        from prompt_optimizer.evaluation.metrics import MetricsComputer, EvalScores

        # Quick metrics test without BERTScore (which requires model download)
        scores = EvalScores(
            original_tokens=client.count_tokens(ORIGINAL_PROMPT),
            compressed_tokens=client.count_tokens(COMPRESSED_PROMPT),
        )
        scores.token_delta = scores.original_tokens - scores.compressed_tokens
        if scores.original_tokens > 0:
            scores.compression_ratio = scores.token_delta / scores.original_tokens

        console.print(f"  ✅ Basic metrics computed | compression_ratio={scores.compression_ratio:.1%}")
    except Exception as e:
        console.print(f"  ❌ Failed: {e}")
        passed = False

    # -----------------------------------------------------------------------
    # Test 6: Eval Engine (with mock responses)
    # -----------------------------------------------------------------------
    console.print("\n[bold]Test 6:[/bold] EvalEngine (using mock/cached responses)...")
    try:
        from prompt_optimizer.evaluation.eval_engine import EvalEngine
        engine = EvalEngine(client=client, cache=cache)
        result = engine.evaluate(
            original_prompt=ORIGINAL_PROMPT,
            compressed_prompt=COMPRESSED_PROMPT,
            task_type=TASK_TYPE,
            reference_response=original_response_text,  # avoids extra API call
        )
        # Override with mock compressed response for pure local test
        result.compressed_response.text = compressed_response_text
        console.print(f"  ✅ EvalEngine returned result | task_type={result.task_type}")
    except Exception as e:
        console.print(f"  ❌ Failed: {e}")
        passed = False

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    console.print()
    table = Table(title="Smoke Test Results", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Original Prompt", ORIGINAL_PROMPT[:60] + "...")
    table.add_row("Compressed Prompt", COMPRESSED_PROMPT[:60])
    table.add_row("Original Tokens", str(client.count_tokens(ORIGINAL_PROMPT)))
    table.add_row("Compressed Tokens", str(client.count_tokens(COMPRESSED_PROMPT)))
    reduction = 1 - client.count_tokens(COMPRESSED_PROMPT) / client.count_tokens(ORIGINAL_PROMPT)
    table.add_row("Token Reduction", f"{reduction:.1%}")
    table.add_row("Cache Entries", str(cache.size))
    table.add_row("Mock Mode", "Yes" if use_mock else "No")

    console.print(table)

    status = "[bold green]ALL TESTS PASSED[/bold green]" if passed else "[bold red]SOME TESTS FAILED[/bold red]"
    console.print(Panel.fit(status))

    if passed:
        console.print(
            "\n[dim]Phase 0 complete! Next: run baselines with[/dim]\n"
            "  [bold]uv run python -m prompt_optimizer.baselines.evaluate_baselines[/bold]\n"
        )

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt Optimizer — Phase 0 Smoke Test")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip real LLM API calls and use mock responses",
    )
    args = parser.parse_args()

    success = run_smoke_test(use_mock=args.no_llm)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
