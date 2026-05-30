"""
TF-IDF Token Selection Baseline.

Scores each token by its TF-IDF weight and keeps only the top-k%.
This is a smarter baseline than stopword removal because it accounts
for how distinctive each word is in context of the whole dataset.

Expected performance:
    compression_ratio: ~25–40%
    quality: ~75–85%
"""

from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter

from prompt_optimizer.baselines.base import BaseCompressor

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, strip punctuation."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


class TFIDFCompressor(BaseCompressor):
    """
    Keep only the top-k% most distinctive tokens by TF-IDF score.

    The IDF component is computed across the entire dataset corpus
    (fit on the dataset before compressing). If not fitted, falls back
    to pure TF (term frequency within the prompt).

    Args:
        keep_ratio:  Fraction of tokens to keep (0.6 = keep 60%).
        min_tokens:  Minimum tokens to keep (prevents over-compression).
    """

    name = "tfidf_selector"

    def __init__(
        self,
        keep_ratio: float = 0.6,
        min_tokens: int = 5,
    ) -> None:
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self._idf: dict[str, float] = {}  # word → IDF score (filled by fit())

    # ------------------------------------------------------------------
    # Fitting (call this on your dataset before evaluating)
    # ------------------------------------------------------------------

    def fit(self, corpus: list[str]) -> "TFIDFCompressor":
        """
        Compute IDF scores from a corpus of prompts.

        Args:
            corpus: List of raw prompt strings.

        Returns:
            self (for chaining)
        """
        n_docs = len(corpus)
        if n_docs == 0:
            return self

        doc_freq: Counter = Counter()
        for text in corpus:
            tokens = set(_tokenize(text))
            doc_freq.update(tokens)

        self._idf = {
            word: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for word, freq in doc_freq.items()
        }
        logger.info("TF-IDF fitted on %d documents, %d unique terms", n_docs, len(self._idf))
        return self

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, prompt: str) -> str:
        """
        Score each word by TF-IDF and keep the top keep_ratio fraction.

        Preserves original word order (only removes low-scoring words).
        """
        words = prompt.split()
        if not words:
            return prompt

        # TF: count within this prompt
        tokens = _tokenize(prompt)
        tf = Counter(tokens)
        total_tokens = len(tokens)

        # Score each original word
        scored: list[tuple[int, float, str]] = []
        for idx, word in enumerate(words):
            clean = re.sub(r"[^a-zA-Z]", "", word).lower()
            if not clean:
                # Keep punctuation-only tokens always
                scored.append((idx, float("inf"), word))
                continue

            tf_score = tf.get(clean, 0) / max(total_tokens, 1)
            idf_score = self._idf.get(clean, 1.0)  # default IDF = 1.0 (unseen words)
            tfidf = tf_score * idf_score
            scored.append((idx, tfidf, word))

        # Determine cutoff
        n_keep = max(self.min_tokens, int(len(words) * self.keep_ratio))
        n_keep = min(n_keep, len(words))  # never keep more than we have

        # Sort by score descending, take top n_keep, restore original order
        top_indices = set(
            idx
            for idx, score, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:n_keep]
        )

        compressed_words = [word for idx, _, word in scored if idx in top_indices]
        return " ".join(compressed_words)
