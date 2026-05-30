"""
Loss functions for the Gumbel-Softmax prompt optimizer.

IMPORTANT — Collapse prevention:
    A naive λ × compression_ratio loss drives the model to drop ALL tokens
    (compression_ratio → 0), because cosine similarity of near-zero vectors
    is undefined / artificially high.
    We prevent this with:
      1. An emptiness penalty: high loss when mean(mask) < min_keep_ratio
      2. A target compression range: penalize both too much AND too little compression

Primary Loss (proxy — no LLM calls, used during training):
    L = -semantic_similarity(masked_emb, original_emb) + λ × compression_ratio

    Where:
    - semantic_similarity = cosine similarity between soft-weighted token
      embedding of compressed prompt vs. original prompt
    - compression_ratio   = fraction of tokens kept (mean of soft_mask)
    - λ (lambda)          = tradeoff coefficient (tune via sweep)

    Minimizing this loss simultaneously:
        → Maximizes cosine similarity  (preserves meaning)
        → Minimizes compression_ratio  (fewer tokens kept)

Why proxy loss?
    The true quality signal requires sending the compressed prompt to Gemini
    and computing BERTScore on the response. This is slow and expensive for
    a per-step training loop. The proxy loss (cosine similarity on embeddings)
    is fast, differentiable, and well-correlated with true quality.

True evaluation (done periodically, not every step):
    Uses Gemini API + BERTScore. See eval_engine.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prompt_optimizer.model.optimizer_model import OptimizerOutput


def proxy_loss(
    output: OptimizerOutput,
    lambda_: float = 0.1,
    min_keep_ratio: float = 0.20,
    target_keep_ratio: float = 0.50,
    emptiness_penalty_weight: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the proxy training loss with collapse prevention.

    Loss = quality_loss + λ × compression_loss + emptiness_penalty

    Where:
      quality_loss      = -cosine_similarity(masked_emb, original_emb)
      compression_loss  = max(0, compression_ratio - target_keep_ratio)
                          (only penalizes keeping MORE than target — pushes toward compression)
      emptiness_penalty = emptiness_penalty_weight × max(0, min_keep_ratio - mean_mask)
                          (heavily penalizes dropping below min_keep_ratio tokens)

    This is computed ENTIRELY from embeddings — no LLM calls needed.
    Fully differentiable — gradients flow to TokenScorer MLP.

    Args:
        output:                  OptimizerOutput from PromptOptimizerModel.forward().
        lambda_:                 Compression penalty coefficient.
        min_keep_ratio:          Minimum fraction of tokens to keep (collapse floor).
        target_keep_ratio:       Fraction of tokens we want the model to aim for.
        emptiness_penalty_weight: Strength of the anti-collapse penalty.

    Returns:
        (loss_tensor, metrics_dict)
    """
    # ----------------------------------------------------------------
    # 1. Semantic similarity term
    # ----------------------------------------------------------------
    # Normalize embeddings to avoid issues with near-zero vectors
    masked_norm = F.normalize(output.masked_embedding, dim=-1, eps=1e-8)
    original_norm = F.normalize(output.original_embedding, dim=-1, eps=1e-8)

    similarity = (masked_norm * original_norm).sum(dim=-1)  # [batch], ∈ [-1, 1]
    quality_term = -similarity.mean()

    # ----------------------------------------------------------------
    # 2. Compression term — only penalize keeping MORE than target
    # (one-sided: we want to compress but not collapse)
    # ----------------------------------------------------------------
    # compression_ratio = fraction of tokens KEPT ∈ (0,1)
    # We want it near target_keep_ratio, so penalize if it's ABOVE target
    compression_term = torch.clamp(
        output.compression_ratio.mean() - target_keep_ratio,
        min=0.0,
    )

    # ----------------------------------------------------------------
    # 3. Emptiness penalty — strongly penalize dropping too many tokens
    # ----------------------------------------------------------------
    mean_mask = output.compression_ratio.mean()
    emptiness_term = torch.clamp(min_keep_ratio - mean_mask, min=0.0)

    # ----------------------------------------------------------------
    # 4. Combined loss
    # ----------------------------------------------------------------
    loss = (
        quality_term
        + lambda_ * compression_term
        + emptiness_penalty_weight * emptiness_term
    )

    # ----------------------------------------------------------------
    # 5. Metrics for logging
    # ----------------------------------------------------------------
    metrics = {
        "loss": loss.item(),
        "quality_term": quality_term.item(),
        "compression_term": compression_term.item(),
        "emptiness_term": emptiness_term.item(),
        "similarity": similarity.mean().item(),
        "compression_ratio": output.compression_ratio.mean().item(),
        "tau": output.tau,
        "lambda": lambda_,
    }

    return loss, metrics


def compression_only_loss(output: OptimizerOutput) -> torch.Tensor:
    """
    Loss that only minimizes compression ratio.
    Useful for sanity check: model should learn to drop all tokens.
    """
    return output.compression_ratio.mean()


def similarity_only_loss(output: OptimizerOutput) -> torch.Tensor:
    """
    Loss that only maximizes semantic similarity.
    Sanity check: model should learn to keep all tokens (identity).
    """
    similarity = F.cosine_similarity(
        output.masked_embedding,
        output.original_embedding,
        dim=-1,
    )
    return -similarity.mean()
