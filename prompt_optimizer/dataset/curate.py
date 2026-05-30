"""
Dataset loading and management utilities.

Usage:
    from prompt_optimizer.dataset.curate import PromptDataset

    ds = PromptDataset.from_jsonl("data/prompts.jsonl")
    print(len(ds))            # total prompts
    print(ds.by_task("qa"))   # filter by task type
    train, val, test = ds.split(train=0.6, val=0.2)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

VALID_TASK_TYPES = {"qa", "summarization", "code", "creative", "general"}


@dataclass
class Prompt:
    """A single prompt entry from the dataset."""

    id: str
    task: str
    prompt: str
    tags: list[str] = field(default_factory=list)
    reference_response: str | None = None  # filled in after LLM call

    def token_count(self) -> int:
        """Quick word-level token estimate."""
        return len(self.prompt.split())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "prompt": self.prompt,
            "tags": self.tags,
            "reference_response": self.reference_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prompt":
        return cls(
            id=data["id"],
            task=data["task"],
            prompt=data["prompt"],
            tags=data.get("tags", []),
            reference_response=data.get("reference_response"),
        )


class PromptDataset:
    """
    Container for a collection of Prompt objects.

    Supports filtering by task type, train/val/test splitting,
    and serialization to/from JSONL.
    """

    def __init__(self, prompts: list[Prompt]) -> None:
        self._prompts = prompts

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PromptDataset":
        """Load from a JSONL file where each line is a JSON object."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        prompts = []
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    prompts.append(Prompt.from_dict(data))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Skipping malformed line %d in %s: %s", lineno, path, e)

        logger.info("Loaded %d prompts from %s", len(prompts), path)
        return cls(prompts)

    @classmethod
    def from_list(cls, prompts: list[dict]) -> "PromptDataset":
        """Create from a list of dicts."""
        return cls([Prompt.from_dict(p) for p in prompts])

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def by_task(self, task: str) -> "PromptDataset":
        """Return a new dataset filtered to a single task type."""
        return PromptDataset([p for p in self._prompts if p.task == task])

    def by_tag(self, tag: str) -> "PromptDataset":
        """Return prompts that include a specific tag."""
        return PromptDataset([p for p in self._prompts if tag in p.tags])

    def with_references(self) -> "PromptDataset":
        """Return only prompts that have a reference_response."""
        return PromptDataset([p for p in self._prompts if p.reference_response])

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(
        self,
        train: float = 0.6,
        val: float = 0.2,
        seed: int = 42,
    ) -> tuple["PromptDataset", "PromptDataset", "PromptDataset"]:
        """
        Split into train / val / test sets.

        Args:
            train: Fraction for training (default 0.6).
            val:   Fraction for validation (default 0.2).
            seed:  Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        assert abs(train + val - 1.0) < 0.3, "train + val should be < 1.0"

        rng = random.Random(seed)
        shuffled = list(self._prompts)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * val)

        return (
            PromptDataset(shuffled[:n_train]),
            PromptDataset(shuffled[n_train : n_train + n_val]),
            PromptDataset(shuffled[n_train + n_val :]),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_jsonl(self, path: str | Path) -> None:
        """Save dataset to JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for p in self._prompts:
                f.write(json.dumps(p.to_dict()) + "\n")
        logger.info("Saved %d prompts to %s", len(self._prompts), path)

    # ------------------------------------------------------------------
    # Iteration & access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self._prompts)

    def __getitem__(self, idx: int) -> Prompt:
        return self._prompts[idx]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics for the dataset."""
        from collections import Counter  # noqa: PLC0415

        task_counts = Counter(p.task for p in self._prompts)
        token_counts = [p.token_count() for p in self._prompts]

        return {
            "total": len(self._prompts),
            "task_distribution": dict(task_counts),
            "avg_tokens": round(sum(token_counts) / max(len(token_counts), 1), 1),
            "min_tokens": min(token_counts, default=0),
            "max_tokens": max(token_counts, default=0),
            "with_references": sum(1 for p in self._prompts if p.reference_response),
        }

    def print_stats(self) -> None:
        """Print a nicely formatted stats summary."""
        try:
            from rich.table import Table  # noqa: PLC0415
            from rich.console import Console  # noqa: PLC0415

            s = self.stats()
            console = Console()
            table = Table(title="📦 Dataset Statistics", show_lines=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Total Prompts", str(s["total"]))
            table.add_row("With References", str(s["with_references"]))
            table.add_row("Avg Tokens", str(s["avg_tokens"]))
            table.add_row("Min Tokens", str(s["min_tokens"]))
            table.add_row("Max Tokens", str(s["max_tokens"]))
            for task, count in s["task_distribution"].items():
                table.add_row(f"  {task}", str(count))
            console.print(table)
        except ImportError:
            print(self.stats())
