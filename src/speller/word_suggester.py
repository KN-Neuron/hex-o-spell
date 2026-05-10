"""Word suggestion engine — frequency-weighted prefix matching.

Given the current "in-progress" word (the partial prefix typed since the
last space), returns the top-N most likely full-word completions from a
precomputed word list.

Algorithm
---------
Linear scan with prefix filter. With ~500 words this is fast enough at
under a microsecond per query — no need for a trie. If the list grows to
~50k+ in the future, swap in a marisa_trie or radix tree.

The word list is loaded from data/language/english_words.json (built by
the one-off script under scripts/), structured as::

    {"words": [{"word": "the", "count": 100}, ...]}

Counts double as ranking weights — higher count means earlier in the
suggestion list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DEFAULT_WORDS_PATH = Path(__file__).resolve().parents[2] / "data" / "language" / "english_words.json"


class WordSuggester:
    """Look up the top-N most frequent words starting with a given prefix."""

    def __init__(self, words_path: Optional[str | Path] = None):
        path = Path(words_path) if words_path else DEFAULT_WORDS_PATH
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Pre-sort by count desc so we can stop at the first n_results matches.
        self._words: list[tuple[str, int]] = sorted(
            ((entry["word"], entry["count"]) for entry in data["words"]),
            key=lambda x: (-x[1], x[0]),
        )

    def suggest(self, prefix: str, n_results: int = 3) -> list[str]:
        """Return up to n_results full words starting with `prefix`.

        Empty prefix returns the top-N globally-frequent words.
        """
        if not prefix:
            return [w for w, _ in self._words[:n_results]]

        prefix = prefix.lower()
        out: list[str] = []
        for word, _count in self._words:
            if word.startswith(prefix):
                out.append(word)
                if len(out) >= n_results:
                    break
        return out

    def __len__(self) -> int:
        return len(self._words)
