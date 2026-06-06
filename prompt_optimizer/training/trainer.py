"""
Training loop for the Gumbel-Softmax Prompt Optimizer.

What happens each step:
    1. Sample a batch of prompts from the dataset
    2. Forward pass through PromptOptimizerModel
       → TokenScorer scores each token
       → GumbelSelector creates soft differentiable mask
       → Masked embeddings computed (differentiable!)
    3. Compute proxy loss (cosine similarity + compression penalty)
       → No LLM calls! Pure embedding math.
    4. Backprop + optimizer step
    5. Anneal Gumbel temperature τ
    6. Every N steps: log metrics
    7. Every M steps: true LLM eval with Gemini (periodic, not every step)

Usage:
    from prompt_optimizer.training.trainer import Trainer
    from prompt_optimizer.training.config import TrainingConfig

    cfg = TrainingConfig(lambda_=0.1, num_epochs=20)
    trainer = Trainer(cfg)
    trainer.train("data/prompts.jsonl")
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from prompt_optimizer.dataset.curate import Prompt, PromptDataset
from prompt_optimizer.model.optimizer_model import PromptOptimizerModel
from prompt_optimizer.training.config import TrainingConfig
from prompt_optimizer.training.losses import proxy_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Accumulated metrics across training steps."""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    similarity: float = 0.0
    compression_ratio: float = 0.0
    tau: float = 0.0
    lr: float = 0.0
    elapsed_s: float = 0.0

    # LLM eval metrics (populated periodically)
    llm_bertscore_f1: Optional[float] = None
    llm_compression_ratio: Optional[float] = None


class Trainer:
    """
    Manages the full training loop for the PromptOptimizerModel.

    Args:
        cfg:   TrainingConfig.
        model: Optional pre-built model (creates one if not provided).
    """

    def __init__(
        self,
        cfg: TrainingConfig | None = None,
        model: PromptOptimizerModel | None = None,
    ) -> None:
        self.cfg = cfg or TrainingConfig()
        self._set_seed(self.cfg.seed)

        self.device = torch.device(self.cfg.device)
        logger.info("Trainer | device=%s | %s", self.device, self.cfg.summary())

        # Model
        self.model = model or PromptOptimizerModel(self.cfg)
        self.model.to(self.device)

        # Optimizer — only trainable params
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # LR scheduler with linear warmup
        self.scheduler = self._build_scheduler()

        # Metrics log
        self.history: list[TrainingMetrics] = []
        self._global_step = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self, data_path: str | Path = "data/prompts.jsonl") -> None:
        """
        Run the full training loop.

        Args:
            data_path: Path to prompts JSONL dataset.
        """
        dataset = PromptDataset.from_jsonl(data_path)
        train_ds, val_ds, _ = dataset.split(train=0.6, val=0.2, seed=self.cfg.seed)

        logger.info(
            "Dataset loaded | train=%d val=%d", len(train_ds), len(val_ds)
        )

        try:
            from rich.console import Console  # noqa: PLC0415
            from rich.progress import (  # noqa: PLC0415
                BarColumn, MofNCompleteColumn, Progress,
                SpinnerColumn, TextColumn, TimeElapsedColumn,
            )
            console = Console()
            _use_rich = True
        except ImportError:
            _use_rich = False

        # Dynamically scale Gumbel temperature annealing steps to match total steps
        steps_per_epoch = (len(train_ds) + self.cfg.batch_size - 1) // self.cfg.batch_size
        total_steps = steps_per_epoch * self.cfg.num_epochs
        self.cfg.gumbel_tau_anneal_steps = total_steps
        logger.info("Automatically scaled gumbel_tau_anneal_steps to %d based on epoch & dataset size", total_steps)

        self.model.train()

        for epoch in range(1, self.cfg.num_epochs + 1):
            prompts = list(train_ds)
            random.shuffle(prompts)

            epoch_losses = []
            epoch_sim = []
            epoch_comp = []

            # Mini-batch training
            for batch_start in range(0, len(prompts), self.cfg.batch_size):
                batch = prompts[batch_start : batch_start + self.cfg.batch_size]
                if not batch:
                    continue

                metrics = self._train_step(batch)
                epoch_losses.append(metrics["loss"])
                epoch_sim.append(metrics["similarity"])
                epoch_comp.append(metrics["compression_ratio"])

                self._global_step += 1

                # Log every N steps
                if self._global_step % self.cfg.eval_every_n_steps == 0:
                    self._log_step(epoch, metrics)

                # LLM eval every M steps (expensive — uses Gemini)
                if self._global_step % self.cfg.llm_eval_every_n_steps == 0:
                    self._run_llm_eval(val_ds)

            # End of epoch summary
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            avg_sim = sum(epoch_sim) / max(len(epoch_sim), 1)
            avg_comp = sum(epoch_comp) / max(len(epoch_comp), 1)

            logger.info(
                "Epoch %d/%d | loss=%.4f sim=%.4f comp=%.4f tau=%.3f",
                epoch, self.cfg.num_epochs,
                avg_loss, avg_sim, avg_comp,
                self.model.gumbel_selector.current_tau,
            )

            # Checkpoint
            if epoch % self.cfg.save_every_n_epochs == 0:
                ckpt_path = f"{self.cfg.checkpoint_dir}/epoch_{epoch:03d}.pt"
                self.model.save(ckpt_path)

        logger.info("Training complete! Total steps: %d", self._global_step)
        self.model.save(f"{self.cfg.checkpoint_dir}/final.pt")
        self._save_history()

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: list[Prompt]) -> dict:
        """Run one forward + backward step on a batch."""
        self.optimizer.zero_grad()

        prompts = [p.prompt for p in batch]
        task_types = [p.task for p in batch]

        # Forward
        output = self.model(prompts=prompts, task_types=task_types, hard=True)

        # Decode compressed texts for fluency signal (cheap — no LLM call)
        # We detach and decode to get actual strings the model would output
        try:
            from prompt_optimizer.model.gumbel_selector import apply_mask_to_tokens  # noqa: PLC0415
            with torch.no_grad():
                encoded = self.model.token_scorer.tokenize(prompts, device=self.device)
                hard_mask = (output.soft_mask >= 0.5).float()
                decoded = apply_mask_to_tokens(
                    input_ids=encoded["input_ids"],
                    mask=hard_mask,
                    tokenizer=self.model.token_scorer.tokenizer,
                )
        except Exception:
            decoded = None

        # Loss
        loss, metrics = proxy_loss(
            output,
            lambda_=self.cfg.lambda_,
            min_keep_ratio=0.20,       # never drop below 20% of tokens
            target_keep_ratio=0.45,    # aim for ~45% token keep (good compression + readable)
            emptiness_penalty_weight=5.0,
            fluency_weight=0.15,       # light fluency signal — guides grammar without overriding compression
            decoded_texts=decoded,
            original_texts=prompts,
        )

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=self.cfg.grad_clip,
        )

        self.optimizer.step()
        self.scheduler.step()

        # Anneal Gumbel temperature
        self.model.step_tau(self._global_step)

        metrics["lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    # ------------------------------------------------------------------
    # LLM evaluation (uses Gemini — periodic)
    # ------------------------------------------------------------------

    def _run_llm_eval(self, val_ds: PromptDataset) -> None:
        """
        Run true LLM evaluation on a small sample of val prompts.

        Sends original + compressed prompts to Gemini, computes BERTScore.
        Results are logged but don't affect training.
        """
        try:
            from prompt_optimizer.evaluation.eval_engine import EvalEngine  # noqa: PLC0415
            from prompt_optimizer.llm.client import LLMClient  # noqa: PLC0415

            sample = list(val_ds)[:5]  # small sample — keep it cheap
            prompts = [p.prompt for p in sample]
            task_types = [p.task for p in sample]

            compressed = self.model.compress(prompts, task_types)

            engine = EvalEngine()
            total_bertscore = 0.0
            total_comp = 0.0

            for orig, comp, task in zip(prompts, compressed, task_types):
                try:
                    result = engine.evaluate(orig, comp, task)
                    total_bertscore += result.scores.bertscore_f1
                    total_comp += result.scores.compression_ratio
                except Exception as e:
                    logger.warning("LLM eval failed for one sample: %s", e)

            n = len(sample)
            avg_bs = total_bertscore / max(n, 1)
            avg_comp = total_comp / max(n, 1)

            logger.info(
                "LLM Eval | step=%d bertscore_f1=%.4f compression=%.4f",
                self._global_step, avg_bs, avg_comp,
            )
        except Exception as e:
            logger.warning("LLM eval skipped: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_step(self, epoch: int, metrics: dict) -> None:
        elapsed = time.time() - self._t0
        m = TrainingMetrics(
            step=self._global_step,
            epoch=epoch,
            loss=metrics["loss"],
            similarity=metrics["similarity"],
            compression_ratio=metrics["compression_ratio"],
            tau=metrics["tau"],
            lr=metrics.get("lr", 0.0),
            elapsed_s=elapsed,
        )
        self.history.append(m)
        logger.info(
            "Step %d | loss=%.4f sim=%.4f comp=%.4f tau=%.3f lr=%.2e",
            self._global_step,
            metrics["loss"],
            metrics["similarity"],
            metrics["compression_ratio"],
            metrics["tau"],
            metrics.get("lr", 0.0),
        )

    def _save_history(self) -> None:
        """Save training history to results/training_history.json."""
        Path("results").mkdir(exist_ok=True)
        history_data = [
            {
                "step": m.step,
                "epoch": m.epoch,
                "loss": m.loss,
                "similarity": m.similarity,
                "compression_ratio": m.compression_ratio,
                "tau": m.tau,
                "lr": m.lr,
                "elapsed_s": m.elapsed_s,
            }
            for m in self.history
        ]
        with open("results/training_history.json", "w") as f:
            json.dump(history_data, f, indent=2)
        logger.info("Training history saved to results/training_history.json")

    def _build_scheduler(self):
        """Linear warmup + cosine annealing."""
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR  # noqa: PLC0415

        total_steps = self.cfg.num_epochs * 10  # rough estimate
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.cfg.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - self.cfg.warmup_steps, 1),
            eta_min=1e-6,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.cfg.warmup_steps],
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
