"""Pluggable speller layouts.

A `SpellerLayout` decides:
  1. How many sectors are on the outer ring.
  2. What letters live in each sector right now (may be context-dependent).
  3. What happens after a letter is committed (e.g. a bigram-adaptive layout
     re-computes its sectors based on the new "previous letter" context).

This separates *navigation mechanics* (Speller state machine: Idle → Writing
→ SectorNavigation → LetterNavigation → ...) from *content selection*
(which letters are reachable from where).

Two implementations:
  - StaticGridLayout: 5 sectors × 6 letters, no context. The original
    BrainBoard layout, kept for backwards compatibility and as the simplest
    fallback.
  - BigramAdaptiveLayout: 6 sectors, dynamically re-populated from a Polish
    bigram table after each letter commit. Implements the architecture
    discussed in the BrainBoard design notes (Markov-1 conditional ring).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpellerLayout(Protocol):
    """Interface every speller layout must satisfy."""

    @property
    def n_sectors(self) -> int:
        """Number of sectors on the outer ring at the current moment.

        For static layouts this is constant; bigram layouts may keep it
        constant too (just re-populate contents) or vary it — both are fine.
        """
        ...

    def letters_in_sector(self, sector_idx: int) -> list[str]:
        """Letters inside the given sector right now.

        Returns a list (order matters — it's how the cursor traverses the
        sector during LetterNavigation).
        """
        ...

    def on_letter_committed(self, letter: str) -> None:
        """Called by the Speller after a letter is selected.

        Static layouts ignore this; adaptive layouts use it to update
        their internal context (e.g. "what was the last letter typed").
        """
        ...

    def reset(self) -> None:
        """Reset any context. Called on Speller back-to-Idle transitions."""
        ...
