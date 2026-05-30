"""
Gumbel-Softmax Binary Selection Gate.

This is the core technique that makes discrete token selection
differentiable during training.

The Problem:
    Choosing which tokens to keep is a discrete (binary) decision.
    Binary decisions have zero gradient almost everywhere — backprop breaks.

The Solution — Gumbel-Softmax (Concrete Distribution):
    1. Add Gumbel noise to logits: logit + Gumbel(0,1)
    2. Apply sigmoid (not softmax — we're doing binary, not categorical)
    3. Divide by temperature τ: sigmoid((logit + noise) / τ)
    4. At training (τ ≈ 1.0): output is soft ∈ (0, 1) — differentiable
    5. At inference (τ → 0): output approaches {0, 1} — discrete

Straight-Through Estimator:
    In the forward pass, we return hard {0,1} masks (via rounding).
    In the backward pass, we use the soft gradient from step 3-4.
    This is done by: hard = round(soft).detach() + soft - soft.detach()

Reference:
    Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017
    Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons", 2013

Usage:
    from prompt_optimizer.model.gumbel_selector import GumbelSelector

    selector = GumbelSelector()
    soft_mask = selector(logits, tau=1.0, hard=False)  # training
    hard_mask = selector(logits, tau=0.1, hard=True)   # inference
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSelector(nn.Module):
    """
    Differentiable binary token selection via Gumbel-Sigmoid.

    Treats each token as an independent Bernoulli variable.
    During training, uses the soft relaxation to allow gradient flow.
    During inference, uses the hard binary mask for actual token selection.

    Args:
        tau_start:  Initial temperature (high = soft, exploratory).
        tau_end:    Final temperature (low = near-discrete).
    """

    def __init__(
        self,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
    ) -> None:
        super().__init__()
        self.tau_start = tau_start
        self.tau_end = tau_end
        self._current_tau = tau_start

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        tau: float | None = None,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Apply Gumbel-Sigmoid to produce a selection mask.

        Args:
            logits: [batch, seq_len] — raw importance scores from TokenScorer.
            tau:    Temperature. If None, uses self._current_tau.
            hard:   If True, returns hard {0,1} mask via straight-through estimator.
                    If False, returns soft ∈ (0,1) mask.

        Returns:
            mask: [batch, seq_len] ∈ {0,1} (hard) or (0,1) (soft).
        """
        tau = tau if tau is not None else self._current_tau

        if self.training or not hard:
            # Add Gumbel noise: sample u ~ Uniform(0,1), then -log(-log(u))
            # This is equivalent to: logits + Gumbel(0,1) noise
            gumbel_noise = self._sample_gumbel(logits.shape, logits.device)
            noisy_logits = logits + gumbel_noise
        else:
            noisy_logits = logits  # no noise at inference

        # Soft mask via sigmoid (Bernoulli relaxation)
        soft_mask = torch.sigmoid(noisy_logits / tau)

        if hard:
            # Straight-through estimator:
            # Forward: hard {0,1} via rounding
            # Backward: gradient flows through soft_mask
            hard_mask = (soft_mask >= 0.5).float()
            mask = hard_mask.detach() + soft_mask - soft_mask.detach()
            return mask
        else:
            return soft_mask

    # ------------------------------------------------------------------
    # Temperature annealing
    # ------------------------------------------------------------------

    def anneal_tau(self, step: int, total_anneal_steps: int) -> float:
        """
        Linearly anneal temperature from tau_start → tau_end.

        Call this at every training step.

        Args:
            step:               Current training step (0-indexed).
            total_anneal_steps: Step at which tau reaches tau_end.

        Returns:
            Current temperature value.
        """
        progress = min(step / max(total_anneal_steps, 1), 1.0)
        self._current_tau = self.tau_start + progress * (self.tau_end - self.tau_start)
        return self._current_tau

    @property
    def current_tau(self) -> float:
        return self._current_tau

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_gumbel(shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Sample from Gumbel(0, 1) distribution.
        g = -log(-log(u))  where u ~ Uniform(0, 1)
        """
        u = torch.rand(shape, device=device).clamp(min=1e-9, max=1 - 1e-9)
        return -torch.log(-torch.log(u))


def apply_mask_to_tokens(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
    threshold: float = 0.5,
) -> list[str]:
    """
    Apply a selection mask to token IDs and decode back to strings.

    Handles subword tokens correctly: if any subword of a word is selected,
    we keep all subwords of that word (avoids 'blockn' artifacts from splitting
    'blockchain' into 'block' + '##n' and only keeping one piece).

    Args:
        input_ids:  [batch, seq_len] — tokenized prompt IDs.
        mask:       [batch, seq_len] — binary mask (1=keep, 0=drop).
        tokenizer:  HuggingFace tokenizer (for decoding).
        threshold:  Threshold for soft→hard if mask is still soft.

    Returns:
        List of decoded compressed prompt strings, one per batch item.
    """
    compressed_texts = []
    binary_mask = (mask >= threshold).bool()  # ensure binary

    for i in range(input_ids.shape[0]):
        ids = input_ids[i].tolist()
        sel = binary_mask[i].tolist()

        # Decode all tokens to check for subword continuation markers (##)
        tokens = tokenizer.convert_ids_to_tokens(ids)

        # Word-boundary alignment:
        # If a continuation subword (##xxx) is selected, also keep its word-start.
        # If a word-start is selected, also keep all its continuation subwords.
        aligned = list(sel)
        for j, tok in enumerate(tokens):
            if tok is None:
                continue
            if tok.startswith("##"):   # continuation subword
                if aligned[j]:         # this subword was selected → keep its word-start
                    # walk back to find the word-starting token
                    k = j - 1
                    while k >= 0 and tokens[k] is not None and tokens[k].startswith("##"):
                        k -= 1
                    if k >= 0:
                        aligned[k] = True
                elif j > 0 and aligned[j - 1]:  # word-start was selected → keep continuation
                    aligned[j] = True

        kept_ids = [tid for tid, keep in zip(ids, aligned) if keep]
        text = tokenizer.decode(kept_ids, skip_special_tokens=True)
        compressed_texts.append(text.strip())

    return compressed_texts
