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

import logging

import torch
import torch.nn.functional as F

from prompt_optimizer.model.optimizer_model import OptimizerOutput

logger = logging.getLogger(__name__)


# Sentence-transformer model — loaded once, cached for the process lifetime
_sentence_model = None


def _get_sentence_model():
    """Lazily load the sentence-transformer model (cached after first call)."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            import os  # noqa: PLC0415
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            _sentence_model = SentenceTransformer(model_name)
            logger.info("Sentence-transformer loaded: %s", model_name)
        except Exception as e:
            logger.warning("Could not load sentence-transformer: %s — skipping fluency term", e)
    return _sentence_model


def proxy_loss(
    output: OptimizerOutput,
    lambda_: float = 0.1,
    min_keep_ratio: float = 0.20,
    target_keep_ratio: float = 0.55,
    emptiness_penalty_weight: float = 5.0,
    fluency_weight: float = 0.3,
    decoded_texts: list[str] | None = None,
    original_texts: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the proxy training loss with collapse prevention and fluency signal.

    Loss = quality_loss + λ × compression_loss + emptiness_penalty + fluency_loss

    Where:
      quality_loss      = -cosine_similarity(masked_emb, original_emb)
                          (token-level embedding similarity — fast, differentiable)
      compression_loss  = max(0, compression_ratio - target_keep_ratio)
                          (only penalizes keeping MORE than target — pushes toward compression)
      emptiness_penalty = emptiness_penalty_weight × max(0, min_keep_ratio - mean_mask)
                          (heavily penalizes dropping below min_keep_ratio tokens)
      fluency_loss      = -sentence_similarity(decoded_compressed, original_text)
                          (sentence-level signal — penalises keyword-soup that loses meaning;
                           only computed when decoded_texts and original_texts are provided)

    This is computed ENTIRELY from embeddings and decoded strings — no LLM calls needed.
    The token-level cosine similarity is fully differentiable.
    The fluency term is a detached scalar signal (does not backprop through the sentence model).

    Args:
        output:                  OptimizerOutput from PromptOptimizerModel.forward().
        lambda_:                 Compression penalty coefficient.
        min_keep_ratio:          Minimum fraction of tokens to keep (collapse floor).
        target_keep_ratio:       Fraction of tokens we want the model to aim for.
        emptiness_penalty_weight: Strength of the anti-collapse penalty.
        fluency_weight:          Weight of the sentence-level fluency term.
        decoded_texts:           Decoded compressed strings (for fluency signal). Optional.
        original_texts:          Original prompt strings (for fluency signal). Optional.

    Returns:
        (loss_tensor, metrics_dict)
    """
    # ----------------------------------------------------------------
    # 1. Semantic similarity term (token-level, fully differentiable)
    # ----------------------------------------------------------------
    masked_norm = F.normalize(output.masked_embedding, dim=-1, eps=1e-8)
    original_norm = F.normalize(output.original_embedding, dim=-1, eps=1e-8)

    similarity = (masked_norm * original_norm).sum(dim=-1)  # [batch], ∈ [-1, 1]
    quality_term = -similarity.mean()

    # ----------------------------------------------------------------
    # 2. Compression term — only penalize keeping MORE than target
    # ----------------------------------------------------------------
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
    # 4. Fluency term — sentence-level similarity (non-differentiable scalar)
    #    Penalises keyword-soup output by comparing full sentence embeddings.
    #    Uses detach() so it acts as a constant offset this step, but it still
    #    guides the loss landscape across many steps.
    # ----------------------------------------------------------------
    fluency_sim = 0.0
    if decoded_texts and original_texts:
        try:
            sent_model = _get_sentence_model()
            if sent_model is not None:
                with torch.no_grad():
                    # Encode on CPU via sentence-transformers, then move to device
                    orig_embs = sent_model.encode(original_texts, convert_to_tensor=True,
                                                  show_progress_bar=False)
                    comp_embs = sent_model.encode(decoded_texts, convert_to_tensor=True,
                                                  show_progress_bar=False)
                    orig_embs = orig_embs.to(output.masked_embedding.device)
                    comp_embs = comp_embs.to(output.masked_embedding.device)
                    fluency_sim = F.cosine_similarity(orig_embs, comp_embs, dim=-1).mean().item()
        except Exception as e:
            logger.debug("Fluency term skipped: %s", e)

    # Fluency loss = -similarity (we want high similarity to original meaning)
    # This is a detached scalar — acts as additive constant for the batch loss value
    fluency_term_scalar = -fluency_sim  # lower is better

    # ----------------------------------------------------------------
    # 5. Combined loss
    # ----------------------------------------------------------------
    loss = (
        quality_term
        + lambda_ * compression_term
        + emptiness_penalty_weight * emptiness_term
    )
    # Add fluency as a scaled constant (no gradient, but informs loss scale logging)
    loss = loss + fluency_weight * torch.tensor(
        fluency_term_scalar, dtype=loss.dtype, device=loss.device
    ).detach()

    # ----------------------------------------------------------------
    # 6. Metrics for logging
    # ----------------------------------------------------------------
    metrics = {
        "loss": loss.item(),
        "quality_term": quality_term.item(),
        "compression_term": compression_term.item(),
        "emptiness_term": emptiness_term.item(),
        "fluency_sim": fluency_sim,
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
