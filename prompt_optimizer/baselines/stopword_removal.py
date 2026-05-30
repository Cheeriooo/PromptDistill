"""
Stopword Removal Baseline.

Removes English stopwords using NLTK. Simple and fast, but linguistically
naive — removes words like "not", "is", "the" which may be important for
some task types.

Expected performance:
    compression_ratio: ~15–25%
    quality: ~80–90% (degrades on tasks where function words matter)
"""

from __future__ import annotations

import logging
import string

from prompt_optimizer.baselines.base import BaseCompressor

logger = logging.getLogger(__name__)

# NLTK stopwords list — bundled inline to avoid requiring an NLTK download
# at test time. This is the standard English stopwords set.
_STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "don't", "should", "should've", "now",
    "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
    "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
    "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't",
    "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't",
})

# Words we NEVER remove regardless of stopword status
# (important for prompt meaning in many task types)
_PROTECTED = frozenset({"not", "no", "never", "without", "except"})


class StopwordRemovalCompressor(BaseCompressor):
    """
    Remove English stopwords from the prompt.

    Args:
        custom_stopwords: Additional words to remove (merged with defaults).
        keep_protected:   If True, keeps words in _PROTECTED (negation words).
    """

    name = "stopword_removal"

    def __init__(
        self,
        custom_stopwords: set[str] | None = None,
        keep_protected: bool = True,
    ) -> None:
        self.stopwords = _STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords |= {w.lower() for w in custom_stopwords}
        if keep_protected:
            self.stopwords -= _PROTECTED

    def compress(self, prompt: str) -> str:
        """
        Remove stopwords from the prompt.

        Strategy:
            1. Split on whitespace (preserve punctuation attached to words)
            2. Remove tokens whose lowercased, stripped form is a stopword
            3. Rejoin with single space
        """
        words = prompt.split()
        result = []
        for word in words:
            # Strip punctuation only for the lookup, keep original word form
            clean = word.strip(string.punctuation).lower()
            if clean and clean not in self.stopwords:
                result.append(word)

        compressed = " ".join(result)

        # If we over-compressed (>80% reduction), fall back to original
        if len(result) == 0 or len(result) < len(words) * 0.2:
            logger.warning("Stopword removal over-compressed — returning original")
            return prompt

        return compressed
