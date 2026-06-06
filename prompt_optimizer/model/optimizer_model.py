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
        ner_boost: float = 8.0,
        content_boost: float = 5.0,
    ) -> list[str]:
        """
        Compress prompts at inference time. Returns decoded strings.

        Two semantic-preservation boosters are applied to logits before masking:

        1. **NER boost** (`ner_boost`): Named entities (people, places, events,
           organisations) detected via spaCy (+ capitalisation heuristic fallback)
           receive a large logit bonus so they are never dropped.

        2. **Content boost** (`content_boost`): Nouns, main verbs, adjectives and
           cardinal numbers identified via NLTK POS tagging receive a moderate
           bonus. This prevents the common failure mode where the model's MLP
           collapses (all logits ≈ 0) and critical words like "code", "provide",
           "binary", "search" are dropped in favour of filler words.

        Args:
            prompts:        List of raw prompt strings.
            task_types:     List of task type strings.
            threshold:      Sigmoid threshold for the hard mask (default 0.5).
            min_keep_ratio: Minimum fraction of real tokens to always keep.
                            Prevents keyword-soup over-compression.
            max_keep_ratio: Maximum fraction to keep (caps very mild compression).
            ner_boost:      Logit bonus for named-entity tokens. Set 0 to disable.
            content_boost:  Logit bonus for content-word tokens (nouns, verbs,
                            adjectives). Set 0 to disable.

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

        # ----------------------------------------------------------------
        # Semantic logit boosts (applied before hard mask)
        # ----------------------------------------------------------------
        # 1. Content-word boost (POS tagging) — applied first so NER can
        #    stack on top for entities that are also content words.
        if content_boost > 0:
            logits = self._apply_content_boost(
                prompts=prompts,
                logits=logits,
                input_ids=input_ids,
                boost=content_boost,
                device=device,
            )

        # 2. Named-entity boost (spaCy + capitalisation heuristic)
        if ner_boost > 0:
            logits = self._apply_ner_boost(
                prompts=prompts,
                logits=logits,
                input_ids=input_ids,
                boost=ner_boost,
                device=device,
            )

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
                real_logits = logits[i][real_indices]
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
    # Content-word boost helper (POS tagging via NLTK)
    # ------------------------------------------------------------------

    def _apply_content_boost(
        self,
        prompts: list[str],
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        boost: float,
        device,
    ) -> torch.Tensor:
        """
        Add `boost` to the logit of every sub-token whose word is a content
        word — i.e. noun, main verb, adjective, or cardinal number — as
        determined by NLTK's POS tagger.

        This guards against the MLP-collapse failure mode where all logits are
        near zero, causing the min_keep topk to select tokens arbitrarily.
        By boosting content words the topk reliably picks meaningful words
        over filler (pronouns, prepositions, courtesy adverbs like "please").

        Args:
            prompts:   Original prompt strings.
            logits:    [batch, seq_len] logits tensor (cloned before modification).
            input_ids: [batch, seq_len] token IDs.
            boost:     Additive bonus applied to content-word positions.
            device:    Target device.

        Returns:
            New logits tensor with content-word positions boosted.
        """
        boosted = logits.clone()
        tokenizer = self.token_scorer.tokenizer

        # POS tags that indicate a content word worth protecting
        CONTENT_TAGS = {
            # --- Nouns (the most important content words) ---
            "NN", "NNS", "NNP", "NNPS",
            # --- Main verbs (all inflected forms — auxiliaries excluded via FILLER_WORDS below) ---
            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
            # --- Adjectives ---
            "JJ", "JJR", "JJS",
            # --- Cardinal numbers (e.g. "top 5", "in 2024") ---
            "CD",
            # --- Question / interrogative words (these define the INTENT of a prompt) ---
            # Without "What" in "What is the capital", the prompt loses all meaning.
            "WP",    # what, who, whom
            "WP$",   # whose
            "WRB",   # how, why, when, where
            "WDT",   # which, that (as interrogative determiner)
            # --- Foreign words ---
            "FW",
        }

        # Words that may carry a verb/noun POS tag but are semantically empty
        # in the context of prompt compression.  These are EXCLUDED from the
        # content boost so they lose to true content words during top-k selection.
        FILLER_WORDS = {
            # Courtesy / hedge adverbs (often mis-tagged as VB)
            "please", "kindly", "just", "simply", "really", "very",
            "actually", "basically", "honestly", "literally",
            # Verbose padding words (common in over-polite prompts)
            "wondering", "appreciate", "possibly", "kind",
            "maybe", "perhaps",
            # Copula / linking verbs — carry zero semantic content
            "is", "are", "was", "were", "be", "been", "being", "am",
            # Auxiliary / helping verbs — grammar glue, not meaning
            "do", "does", "did",
            "has", "have", "had",
            "will", "would", "could", "should", "shall", "might",
            "ca",   # NLTK tokenises "can't" → ["ca", "n't"]
            # Common low-value verbs that typically serve as filler in prompts
            # e.g. "Could you *help* me ..." — "help" adds no info vs the task
            "help", "let", "give", "tell", "get", "make", "need", "want",
            "know", "able", "try",
            # Demonstratives / pronouns that add no search value
            "this", "that", "these", "those",
        }

        # ---- Negation & constraint words (ALWAYS boosted, regardless of POS tag) ----
        # Dropping a negation INVERTS the meaning of the entire prompt:
        #   "does not use recursion" → "use recursion" is the OPPOSITE intent.
        # These are tagged as RB (adverb) or IN (preposition) by NLTK, which
        # are normally not in CONTENT_TAGS, so we need a special-case set.
        CRITICAL_WORDS = {
            # Negation — flips meaning
            "not", "n't", "no", "never", "neither", "nor", "none",
            # Exclusion / constraint — scopes or restricts the task
            "without", "except", "only", "exclusively", "unless",
            "instead", "rather", "versus", "vs",
            # Comparison — critical for comparison prompts
            "between", "compared", "vs.",
            # Inclusion scope
            "each", "every", "all", "both",
        }

        # Lazily import and set up NLTK (already a project dependency)
        try:
            import nltk  # noqa: PLC0415
            for resource in (
                "taggers/averaged_perceptron_tagger_eng",
                "tokenizers/punkt_tab",
            ):
                try:
                    nltk.data.find(resource)
                except LookupError:
                    nltk.download(resource.split("/")[-1], quiet=True)
        except ImportError:
            logger.debug("NLTK not available; content boost skipped.")
            return boosted

        for i, prompt in enumerate(prompts):
            try:
                words = nltk.word_tokenize(prompt)
                pos_tags = nltk.pos_tag(words)
            except Exception as exc:
                logger.debug("NLTK POS tagging failed for batch %d: %s", i, exc)
                continue

            encoding = tokenizer(
                prompt,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            offsets = encoding["offset_mapping"]
            protect = [False] * len(offsets)

            search_from = 0
            for word, tag in pos_tags:
                # Always advance search_from past this word
                word_start = prompt.find(word, search_from)
                if word_start == -1:
                    continue
                word_end = word_start + len(word)
                search_from = word_end

                # Decide whether this word should be protected
                word_lower = word.lower()
                is_critical = word_lower in CRITICAL_WORDS
                is_content = tag in CONTENT_TAGS and word_lower not in FILLER_WORDS

                if not is_critical and not is_content:
                    continue

                # Mark all sub-tokens overlapping this word's character span
                for tok_idx, (start, end) in enumerate(offsets):
                    if start is None or end is None or (start == 0 and end == 0):
                        continue
                    if start < word_end and end > word_start:
                        protect[tok_idx] = True

            for tok_idx, should_protect in enumerate(protect):
                if should_protect and tok_idx < boosted.shape[1]:
                    boosted[i, tok_idx] += boost

        return boosted

    # ------------------------------------------------------------------
    # NER boost helper
    # ------------------------------------------------------------------

    def _apply_ner_boost(
        self,
        prompts: list[str],
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        boost: float,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Add `boost` to the logit of every sub-token that belongs to a named
        entity span detected by spaCy.  Also boosts capitalised words that
        are NOT at the start of the sentence (a cheap heuristic for proper
        nouns when spaCy is unavailable or misses an entity).

        Args:
            prompts:   Original prompt strings (one per batch item).
            logits:    [batch, seq_len] logits tensor (will not be mutated).
            input_ids: [batch, seq_len] token IDs.
            boost:     Additive bonus to apply.
            device:    Target device for the returned tensor.

        Returns:
            New logits tensor with entity positions boosted.
        """
        boosted = logits.clone()
        tokenizer = self.token_scorer.tokenizer

        # --- Try spaCy first ---
        nlp = None
        try:
            import spacy  # noqa: PLC0415
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.debug("spaCy model 'en_core_web_sm' not found; using capitalisation heuristic.")
        except ImportError:
            logger.debug("spaCy not installed; using capitalisation heuristic for NER boost.")

        for i, prompt in enumerate(prompts):
            ids = input_ids[i].tolist()
            tokens = tokenizer.convert_ids_to_tokens(ids)

            # Build a boolean protection mask (True = must keep)
            protect = [False] * len(tokens)

            if nlp is not None:
                # ---- spaCy-based entity detection ----
                doc = nlp(prompt)
                for ent in doc.ents:
                    # Map entity character span → token positions via
                    # the tokenizer's char-to-token offset mapping.
                    encoding = tokenizer(
                        prompt,
                        return_offsets_mapping=True,
                        add_special_tokens=True,
                    )
                    offsets = encoding["offset_mapping"]
                    for tok_idx, (start, end) in enumerate(offsets):
                        if start is None or end is None:
                            continue
                        # Token overlaps with the entity span
                        if start < ent.end_char and end > ent.start_char:
                            if tok_idx < len(protect):
                                protect[tok_idx] = True

            # ---- Capitalisation heuristic (always applied as a safety net) ----
            # Words after the first that start with a capital letter are likely
            # proper nouns.  We re-tokenize word-by-word to map them.
            words = prompt.split()
            char_pos = 0
            for w_idx, word in enumerate(words):
                # Find where this word starts in the original string
                char_pos = prompt.find(word, char_pos)
                word_end = char_pos + len(word)
                is_mid_sentence_capital = (
                    w_idx > 0
                    and word[:1].isupper()
                    and not word.isupper()  # skip ALL-CAPS abbreviations (handled separately)
                )
                is_all_caps_abbrev = len(word.strip('.,;:!?')) >= 2 and word.strip('.,;:!?').isupper()
                if is_mid_sentence_capital or is_all_caps_abbrev:
                    # Mark any token that overlaps this word
                    enc = tokenizer(
                        prompt,
                        return_offsets_mapping=True,
                        add_special_tokens=True,
                    )
                    for tok_idx, (start, end) in enumerate(enc["offset_mapping"]):
                        if start is None or end is None:
                            continue
                        if start < word_end and end > char_pos:
                            if tok_idx < len(protect):
                                protect[tok_idx] = True
                char_pos = word_end

            # Apply boost wherever protect is True
            for tok_idx, should_protect in enumerate(protect):
                if should_protect and tok_idx < boosted.shape[1]:
                    boosted[i, tok_idx] += boost

        return boosted

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
