"""
Token Scorer — the core learnable module.

A lightweight MLP that reads contextual token embeddings (from a frozen
DistilBERT backbone) + a task-type embedding, and outputs a scalar
importance score (logit) per token.

Architecture:
    DistilBERT (frozen) → [seq_len, 768]
    Task Embedding      → [task_embed_dim]    (broadcast to [seq_len, task_embed_dim])
    Concatenate         → [seq_len, 768 + task_embed_dim]
    MLP (2 layers)      → [seq_len, 1] → squeeze → [seq_len]   (importance logits)

Usage:
    from prompt_optimizer.model.token_scorer import TokenScorer

    scorer = TokenScorer(cfg)
    logits = scorer(input_ids, attention_mask, task_idx)
    # logits: [batch, seq_len]  — unbounded importance scores per token
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from prompt_optimizer.training.config import TrainingConfig

logger = logging.getLogger(__name__)


class TokenScorer(nn.Module):
    """
    Scores each token's importance using a frozen encoder + trainable MLP.

    The encoder (DistilBERT by default) is FROZEN — we only train the
    lightweight MLP on top. This keeps training fast and prevents
    catastrophic forgetting of language representations.

    Args:
        cfg: TrainingConfig with all hyperparameters.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ----------------------------------------------------------------
        # Frozen backbone — produces contextual token embeddings
        # ----------------------------------------------------------------
        logger.info("Loading encoder: %s", cfg.scorer_base_model)
        self.encoder = AutoModel.from_pretrained(cfg.scorer_base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_base_model)
        self.encoder_dim = self.encoder.config.hidden_size  # 768 for distilbert

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        logger.info("Encoder frozen (%d params)", sum(p.numel() for p in self.encoder.parameters()))

        # ----------------------------------------------------------------
        # Task-type embedding (trainable)
        # ----------------------------------------------------------------
        self.task_embedding = nn.Embedding(
            num_embeddings=len(cfg.task_types),
            embedding_dim=cfg.task_embed_dim,
        )

        # ----------------------------------------------------------------
        # Token importance MLP (trainable — the actual learning happens here)
        # ----------------------------------------------------------------
        mlp_input_dim = self.encoder_dim + cfg.task_embed_dim
        self.scorer_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, cfg.scorer_hidden_dim),
            nn.LayerNorm(cfg.scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.scorer_dropout),
            nn.Linear(cfg.scorer_hidden_dim, cfg.scorer_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.scorer_dropout),
            nn.Linear(cfg.scorer_hidden_dim // 2, 1),  # → scalar per token
        )

        logger.info(
            "TokenScorer ready | encoder_dim=%d task_embed=%d mlp_input=%d trainable_params=%d",
            self.encoder_dim,
            cfg.task_embed_dim,
            mlp_input_dim,
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token importance logits.

        Args:
            input_ids:      [batch, seq_len] — tokenized input IDs.
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding.
            task_idx:       [batch] — integer task type index.

        Returns:
            logits: [batch, seq_len] — unbounded importance score per token.
                    Padding positions will have -inf injected before Gumbel step.
        """
        batch_size, seq_len = input_ids.shape

        # 1. Get contextual token embeddings from frozen encoder
        with torch.no_grad():
            encoder_out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        token_embs = encoder_out.last_hidden_state  # [batch, seq_len, encoder_dim]

        # 2. Get task embedding and broadcast to [batch, seq_len, task_embed_dim]
        task_emb = self.task_embedding(task_idx)              # [batch, task_embed_dim]
        task_emb = task_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, task_embed_dim]

        # 3. Concatenate token + task features
        combined = torch.cat([token_embs, task_emb], dim=-1)  # [batch, seq_len, mlp_input_dim]

        # 4. Score each token
        logits = self.scorer_mlp(combined).squeeze(-1)  # [batch, seq_len]

        # 5. Mask padding positions with -inf so they're never selected
        logits = logits.masked_fill(attention_mask == 0, float("-inf"))

        return logits

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def tokenize(
        self,
        texts: list[str],
        device: str | torch.device = "cpu",
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize a list of strings and return tensors on the given device.

        Returns dict with keys: input_ids, attention_mask
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def decode_tokens(self, input_ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
