"""
Full end-to-end Prompt Optimizer model.

Wires together:
    TokenScorer (frozen encoder + trainable MLP)
    → GumbelSelector (differentiable discrete selection)
    → Compressed prompt (string or masked embedding)

Training mode:
    Returns soft_mask + masked embeddings for proxy loss computation.
    No LLM calls — fast, fully differentiable.

Inference mode:
    Returns hard binary mask + actual compressed prompt string.
    Can then be sent to Gemini for true quality evaluation.

Usage:
    from prompt_optimizer.model.optimizer_model import PromptOptimizerModel
    from prompt_optimizer.training.config import TrainingConfig

    cfg = TrainingConfig()
    model = PromptOptimizerModel(cfg)

    # Training step (differentiable)
    out = model(prompts=["..."], task_types=["qa"])
    # out.soft_mask: [batch, seq_len]
    # out.masked_embedding: [batch, encoder_dim]
    # out.original_embedding: [batch, encoder_dim]
    # out.compression_ratio: [batch]

    # Inference (returns compressed strings)
    compressed = model.compress(["Could you please explain gradient descent?"], ["qa"])
    # ["Explain gradient descent."]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prompt_optimizer.model.gumbel_selector import GumbelSelector, apply_mask_to_tokens
from prompt_optimizer.model.token_scorer import TokenScorer
from prompt_optimizer.training.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizerOutput:
    """
    Output from a forward pass through the PromptOptimizerModel.

    Used during training for loss computation.
    """
    # Differentiable mask ∈ (0,1) — used for proxy loss
    soft_mask: torch.Tensor              # [batch, seq_len]

    # Soft-weighted token embedding (for semantic similarity loss)
    masked_embedding: torch.Tensor      # [batch, encoder_dim]

    # Mean-pooled original token embedding (the "target")
    original_embedding: torch.Tensor    # [batch, encoder_dim]

    # Fraction of tokens kept (mean of soft_mask over real tokens)
    compression_ratio: torch.Tensor     # [batch]   ∈ (0,1) — lower = more compressed

    # Raw logits from TokenScorer (for debugging / visualization)
    logits: torch.Tensor                # [batch, seq_len]

    # Current Gumbel temperature
    tau: float


class PromptOptimizerModel(nn.Module):
    """
    End-to-end differentiable prompt compression model.

    Architecture:
        Input texts
        → Tokenize (DistilBERT tokenizer)
        → TokenScorer: frozen encoder + task emb + MLP → importance logits
        → GumbelSelector: Gumbel-Sigmoid → soft/hard mask
        → Training: masked embedding for proxy loss
        → Inference: token filtering → decoded string

    Args:
        cfg: TrainingConfig with all hyperparameters.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_scorer = TokenScorer(cfg)
        self.gumbel_selector = GumbelSelector(
            tau_start=cfg.gumbel_tau_start,
            tau_end=cfg.gumbel_tau_end,
        )

        logger.info(
            "PromptOptimizerModel ready | trainable=%d params | device=%s",
            self.trainable_params,
            cfg.device,
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        prompts: list[str],
        task_types: list[str],
        hard: bool = False,
    ) -> OptimizerOutput:
        """
        Forward pass for TRAINING.

        Returns soft masked embeddings + compression ratios for loss computation.
        This is fully differentiable (no LLM calls).

        Args:
            prompts:    List of raw prompt strings.
            task_types: List of task type strings (same length as prompts).
            hard:       If True, use hard Gumbel mask (for eval during training).

        Returns:
            OptimizerOutput with all tensors needed for loss computation.
        """
        device = next(self.parameters()).device

        # 1. Tokenize
        encoded = self.token_scorer.tokenize(prompts, device=device)
        input_ids = encoded["input_ids"]         # [batch, seq_len]
        attention_mask = encoded["attention_mask"]  # [batch, seq_len]

        # 2. Convert task types to indices
        task_indices = torch.tensor(
            [self.cfg.task_to_idx(t) for t in task_types],
            dtype=torch.long,
            device=device,
        )  # [batch]

        # 3. Score tokens → importance logits
        logits = self.token_scorer(input_ids, attention_mask, task_indices)
        # logits: [batch, seq_len], padding positions are -inf

        # 4. Apply Gumbel-Sigmoid → soft mask
        tau = self.gumbel_selector.current_tau
        soft_mask = self.gumbel_selector(logits, tau=tau, hard=hard)
        # soft_mask: [batch, seq_len] ∈ (0,1)

        # 5. Get token embeddings from encoder (reuse what scorer already computed)
        with torch.no_grad():
            encoder_out = self.token_scorer.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        token_embs = encoder_out.last_hidden_state  # [batch, seq_len, encoder_dim]

        # 6. Compute original embedding (mean pool over real tokens)
        real_mask = attention_mask.float().unsqueeze(-1)  # [batch, seq_len, 1]
        original_embedding = (token_embs * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        # [batch, encoder_dim]

        # 7. Compute masked embedding (soft-weighted pool — differentiable!)
        soft_mask_expanded = soft_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_embedding = (token_embs * soft_mask_expanded * real_mask).sum(dim=1)
        masked_embedding = masked_embedding / (soft_mask_expanded * real_mask).sum(dim=1).clamp(min=1e-6)
        # [batch, encoder_dim]

        # 8. Compression ratio = mean of soft_mask over real tokens
        real_token_count = attention_mask.float().sum(dim=1)  # [batch]
        compression_ratio = (soft_mask * attention_mask.float()).sum(dim=1) / real_token_count.clamp(min=1)
        # [batch] ∈ (0,1) — lower = fewer tokens kept = more compressed

        return OptimizerOutput(
            soft_mask=soft_mask,
            masked_embedding=masked_embedding,
            original_embedding=original_embedding,
            compression_ratio=compression_ratio,
            logits=logits,
            tau=tau,
        )

    # ------------------------------------------------------------------
    # Inference — produce compressed strings
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compress(
        self,
        prompts: list[str],
        task_types: list[str],
        threshold: float = 0.5,
        min_keep_ratio: float = 0.55,
        max_keep_ratio: float = 0.85,
    ) -> list[str]:
        """
        Compress prompts at inference time. Returns decoded strings.

        Args:
            prompts:        List of raw prompt strings.
            task_types:     List of task type strings.
            threshold:      Sigmoid threshold for the hard mask (default 0.5).
            min_keep_ratio: Minimum fraction of real tokens to always keep.
                            Prevents keyword-soup over-compression. Default 0.40
                            means at least 40% of tokens are always preserved.
            max_keep_ratio: Maximum fraction to keep (caps very mild compression).

        Returns:
            List of compressed prompt strings.
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        encoded = self.token_scorer.tokenize(prompts, device=device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        task_indices = torch.tensor(
            [self.cfg.task_to_idx(t) for t in task_types],
            dtype=torch.long,
            device=device,
        )

        logits = self.token_scorer(input_ids, attention_mask, task_indices)

        # Hard mask at inference (no Gumbel noise, just threshold the sigmoid)
        hard_mask = (torch.sigmoid(logits / self.gumbel_selector.tau_end) >= threshold).float()

        # --- Quality guard: enforce min/max keep ratios per sequence ---
        for i in range(hard_mask.shape[0]):
            real_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            real_tokens = len(real_indices)
            if real_tokens == 0:
                continue

            kept = int(hard_mask[i][real_indices].sum().item())
            min_keep = max(3, int(real_tokens * min_keep_ratio))
            max_keep = int(real_tokens * max_keep_ratio)

            if kept < min_keep:
                # Model dropped too much → promote top-scored tokens until floor is met
                # Get logits only for real (non-padding) positions
                real_logits = logits[i][real_indices]
                # Sort by score descending, take top min_keep
                top_real_indices = real_logits.topk(min(min_keep, real_tokens)).indices
                hard_mask[i] = torch.zeros_like(hard_mask[i])
                hard_mask[i][real_indices[top_real_indices]] = 1.0
            elif kept > max_keep:
                # Model kept too much → demote lowest-scored surplus tokens
                real_logits = logits[i][real_indices]
                top_real_indices = real_logits.topk(max_keep).indices
                hard_mask[i] = torch.zeros_like(hard_mask[i])
                hard_mask[i][real_indices[top_real_indices]] = 1.0

        compressed = apply_mask_to_tokens(
            input_ids=input_ids,
            mask=hard_mask,
            tokenizer=self.token_scorer.tokenizer,
            threshold=threshold,
        )

        if was_training:
            self.train()

        return compressed

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def step_tau(self, step: int) -> float:
        """Anneal Gumbel temperature. Call once per training step."""
        return self.gumbel_selector.anneal_tau(step, self.cfg.gumbel_tau_anneal_steps)

    def save(self, path: str) -> None:
        """Save trainable weights only (not the frozen encoder)."""
        import os  # noqa: PLC0415
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "scorer_mlp": self.token_scorer.scorer_mlp.state_dict(),
            "task_embedding": self.token_scorer.task_embedding.state_dict(),
            "cfg": self.cfg.model_dump(),
        }, path)
        logger.info("Saved checkpoint to %s", path)

    def load(self, path: str) -> None:
        """Load trainable weights from a checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        self.token_scorer.scorer_mlp.load_state_dict(ckpt["scorer_mlp"])
        self.token_scorer.task_embedding.load_state_dict(ckpt["task_embedding"])
        logger.info("Loaded checkpoint from %s", path)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
