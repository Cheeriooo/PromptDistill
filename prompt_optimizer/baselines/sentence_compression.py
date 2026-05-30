"""
Sentence Compression Baseline.

Uses extractive compression: splits the prompt into sentences and keeps
only the most "important" ones based on a simple scoring heuristic
(sentence length + presence of key task-relevant terms).

This is a stronger baseline than stopword removal or TF-IDF because
it preserves grammatical coherence at the sentence level.

Expected performance:
    compression_ratio: ~30–50%
    quality: ~70–85%
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter

from prompt_optimizer.baselines.base import BaseCompressor

logger = logging.getLogger(__name__)

# Task-type keywords that boost sentence importance scores
_TASK_KEYWORDS: dict[str, set[str]] = {
    "qa": {"explain", "what", "how", "why", "define", "describe", "difference", "compare"},
    "summarization": {"summarize", "summary", "main", "key", "points", "brief", "concise", "overview"},
    "code": {"write", "implement", "function", "class", "code", "script", "create", "build"},
    "creative": {"write", "story", "poem", "character", "describe", "imagine", "create"},
    "general": set(),
}

# Polite/filler phrases that add tokens but no information
_FILLER_PHRASES = [
    r"could you please\s*",
    r"would you be able to\s*",
    r"i would really appreciate it if you could\s*",
    r"i would like you to\s*",
    r"i would appreciate it if you\s*",
    r"i am trying to understand\s*",
    r"i have been wondering\s*",
    r"i need you to\s*",
    r"please\s+",
    r"i would like\s*",
    r"can you\s*",
    r"i want you to\s*",
]

_FILLER_RE = re.compile(
    "|".join(_FILLER_PHRASES),
    flags=re.IGNORECASE,
)


class SentenceCompressionCompressor(BaseCompressor):
    """
    Extractive sentence-level compression.

    Strategy:
        1. Strip leading filler phrases (politeness → imperative)
        2. Split into sentences
        3. Score sentences by: length bonus + keyword overlap + position
        4. Keep top sentences up to keep_ratio
        5. Preserve original sentence order

    Args:
        keep_ratio:        Fraction of sentences to keep (0.6 = keep 60%).
        min_sentences:     Minimum sentences to always keep.
        task_type:         If known, use task-specific keywords for scoring.
        strip_fillers:     Remove common filler/politeness phrases.
    """

    name = "sentence_compression"

    def __init__(
        self,
        keep_ratio: float = 0.6,
        min_sentences: int = 1,
        task_type: str = "general",
        strip_fillers: bool = True,
    ) -> None:
        self.keep_ratio = keep_ratio
        self.min_sentences = min_sentences
        self.task_type = task_type
        self.strip_fillers = strip_fillers
        self._keywords = _TASK_KEYWORDS.get(task_type, set())

    def compress(self, prompt: str) -> str:
        """
        Compress by stripping fillers and extracting key sentences.
        """
        text = prompt.strip()

        # Step 1: Strip filler phrases from the beginning
        if self.strip_fillers:
            text = _FILLER_RE.sub("", text)
            # Capitalize first letter after stripping
            text = text.strip()
            if text:
                text = text[0].upper() + text[1:]

        # Step 2: Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            # Single sentence — just return stripped version
            return text

        # Step 3: Score each sentence
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences))
            scored.append((i, score, sent))

        # Step 4: Keep top-k sentences
        n_keep = max(self.min_sentences, int(len(sentences) * self.keep_ratio))
        n_keep = min(n_keep, len(sentences))

        top_indices = set(
            idx
            for idx, score, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:n_keep]
        )

        # Step 5: Restore original order
        kept = [sent for idx, _, sent in scored if idx in top_indices]
        return " ".join(kept)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using basic punctuation rules."""
        # Split on . ! ? followed by whitespace or end of string
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        """
        Score a sentence for importance.

        Factors:
            - Normalized length (longer = more content, up to a point)
            - Keyword overlap with task-type keywords
            - Position bonus (first sentence is usually most important)
        """
        words = sentence.lower().split()
        if not words:
            return 0.0

        # Length score: peak at 15-20 words, penalize very short or very long
        length_score = min(len(words) / 15.0, 1.0)

        # Keyword overlap
        word_set = set(w.strip(string.punctuation) for w in words)
        keyword_overlap = len(word_set & self._keywords) / max(len(self._keywords), 1)

        # Position bonus: first sentence is most important
        position_score = 1.0 if position == 0 else max(0.0, 1.0 - position / total)

        return 0.4 * length_score + 0.4 * keyword_overlap + 0.2 * position_score
