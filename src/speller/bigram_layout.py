"""BigramAdaptiveLayout — Markov-1 conditional ring keyboard.

Architecture
------------
After each committed letter, the layout re-orders all 36 ring positions by
P(next | last_letter). The 6×6 = 36 positions split into 6 sectors of 6
letters each, sector 0 holding the most likely 6 continuations, sector 5
the least likely.

The user sees only the *current sector*'s contents at any time (per the
Ring-O-Spell screens, where 6 letters appear in a "fan" above the ring).
Moving L/R rotates which sector is highlighted; selecting (blink) opens
the sector for letter selection.

This means typing a high-frequency continuation costs:
    ~1-2 sector navigations + 1 blink + ~1-2 letter navigations + 1 blink
≈ 5-6 actions average, vs ~3.5-4 for an idealized binary tree (Dembol's
discussion). The ring sacrifices some efficiency for UX legibility — users
see all candidates simultaneously, no mental tree-walking.

Cold start
----------
Before any letter has been committed, the layout uses the *unigram*
distribution. So the first letter typed will be biased toward the most
common letters in Polish (space, a, o, e, i, ...).

Fallback
--------
If the bigram table has no entry for the current `last_letter` (rare, e.g.
if the previous letter was a backspace token), we fall back to unigrams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.speller.layout import SpellerLayout


N_SECTORS = 6
LETTERS_PER_SECTOR = 6
RING_SIZE = N_SECTORS * LETTERS_PER_SECTOR  # 36

# Where to find the precomputed bigram table by default.
DEFAULT_BIGRAMS_PATH = Path(__file__).resolve().parents[2] / "data" / "language" / "polish_bigrams.json"


class BigramAdaptiveLayout:
    """Conditional 6×6 ring keyboard.

    Parameters
    ----------
    bigrams_path : str | Path | None
        Path to a JSON file with shape::

            {
              "alphabet": [...],
              "unigram": {char: prob, ...},
              "bigram":  {prev_char: {next_char: prob, ...}, ...}
            }

        Defaults to data/language/polish_bigrams.json shipped with the repo.
    initial_letter : str | None
        Optional letter to seed the layout with. Defaults to ' ' (space) which
        gives "first letter of a word" frequencies — the most useful prior
        for typing a fresh word.
    """

    def __init__(
        self,
        bigrams_path: Optional[str | Path] = None,
        initial_letter: str | None = " ",
    ):
        path = Path(bigrams_path) if bigrams_path else DEFAULT_BIGRAMS_PATH
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._alphabet: list[str] = data["alphabet"]
        if len(self._alphabet) > RING_SIZE:
            # Keep only the top-RING_SIZE most common ones (by unigram).
            unigram = data["unigram"]
            self._alphabet = sorted(
                self._alphabet, key=lambda c: -unigram.get(c, 0.0)
            )[:RING_SIZE]

        # Pad the alphabet to RING_SIZE with placeholders so we always have
        # exactly 6×6 positions. The placeholder is rendered as '·' but
        # selecting it commits an empty string (no-op).
        while len(self._alphabet) < RING_SIZE:
            self._alphabet.append("·")

        self._unigram: dict[str, float] = data["unigram"]
        self._bigram: dict[str, dict[str, float]] = data["bigram"]
        self._initial_letter = initial_letter
        self._last_letter: str | None = initial_letter

        self._sectors: list[list[str]] = []
        self._rebuild_sectors()

    # ─── SpellerLayout protocol ─────────────────────────────────────────────

    @property
    def n_sectors(self) -> int:
        return N_SECTORS

    def letters_in_sector(self, sector_idx: int) -> list[str]:
        if sector_idx < 0 or sector_idx >= N_SECTORS:
            raise IndexError(f"sector_idx {sector_idx} out of range [0, {N_SECTORS})")
        return list(self._sectors[sector_idx])

    def on_letter_committed(self, letter: str) -> None:
        # Treat punctuation/space the same as any other letter — they're
        # legitimate "previous letters" for bigram lookup. The placeholder '·'
        # however should NOT update context.
        if letter == "·":
            return
        self._last_letter = letter
        self._rebuild_sectors()

    def reset(self) -> None:
        self._last_letter = self._initial_letter
        self._rebuild_sectors()

    # ─── Internal: rank letters by conditional probability ──────────────────

    def _rebuild_sectors(self) -> None:
        """Sort the alphabet by P(letter | last_letter), then chunk into 6×6."""
        ranked = self._rank_letters()
        self._sectors = [
            ranked[i * LETTERS_PER_SECTOR : (i + 1) * LETTERS_PER_SECTOR]
            for i in range(N_SECTORS)
        ]

    def _rank_letters(self) -> list[str]:
        """Return self._alphabet sorted by descending probability under the
        current context. Letters with no observed conditional probability fall
        back to their unigram probability, then to 0."""
        if self._last_letter is not None and self._last_letter in self._bigram:
            cond = self._bigram[self._last_letter]
        else:
            cond = {}

        def score(letter: str) -> float:
            # Conditional probability if available, else unigram, else 0.
            if letter in cond:
                # Boost so any positive bigram beats any pure unigram fallback.
                return 1.0 + cond[letter]
            return self._unigram.get(letter, 0.0)

        # Stable sort: ties broken by alphabet order (deterministic for tests).
        return sorted(self._alphabet, key=lambda c: (-score(c), c))


# Sanity check at import: this class satisfies SpellerLayout.
_check: SpellerLayout = BigramAdaptiveLayout()
del _check
