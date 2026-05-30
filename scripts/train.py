"""
Phase 2 training entry point.

Run with:
    uv run python scripts/train.py
    uv run python scripts/train.py --lambda 0.05
    uv run python scripts/train.py --lambda 0.5 --epochs 30
    uv run python scripts/train.py --dry-run   # one batch only, verify setup
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Gumbel-Softmax Prompt Optimizer")
    parser.add_argument("--data", default="data/prompts.jsonl", help="Dataset path")
    parser.add_argument("--lambda", dest="lambda_coeff", type=float, default=None,
                        help="Compression-quality tradeoff (overrides .env LAMBDA)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda | auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one batch only to verify setup (no real training)")
    args = parser.parse_args()

    from prompt_optimizer.training.config import TrainingConfig  # noqa: PLC0415
    from prompt_optimizer.training.trainer import Trainer  # noqa: PLC0415

    # Build config — CLI args override .env defaults
    cfg_kwargs = {}
    if args.lambda_coeff is not None:
        cfg_kwargs["lambda_coeff"] = args.lambda_coeff
    if args.epochs is not None:
        cfg_kwargs["num_epochs"] = args.epochs
    if args.lr is not None:
        cfg_kwargs["learning_rate"] = args.lr
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.device is not None:
        cfg_kwargs["device"] = args.device
    if args.dry_run:
        cfg_kwargs["num_epochs"] = 1
        cfg_kwargs["eval_every_n_steps"] = 1
        cfg_kwargs["llm_eval_every_n_steps"] = 999999  # skip LLM eval in dry run

    cfg = TrainingConfig(**cfg_kwargs)

    logger.info("=" * 60)
    logger.info("Prompt Optimizer — Phase 2 Training")
    logger.info("Config: %s", cfg.summary())
    logger.info("=" * 60)

    trainer = Trainer(cfg)

    if args.dry_run:
        logger.info("DRY RUN mode — running one batch only")
        from prompt_optimizer.dataset.curate import PromptDataset  # noqa: PLC0415
        ds = PromptDataset.from_jsonl(args.data)
        batch = list(ds)[:cfg.batch_size]
        metrics = trainer._train_step(batch)
        logger.info("Dry run batch metrics: %s", metrics)
        logger.info("Dry run SUCCESS — model forward/backward pass works!")
        return

    trainer.train(data_path=args.data)


if __name__ == "__main__":
    main()
