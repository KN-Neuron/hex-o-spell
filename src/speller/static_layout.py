"""StaticGridLayout — the original 5×6 BrainBoard layout (no context)."""

from __future__ import annotations

from src.speller.layout import SpellerLayout


# Original layout from src/speller/speller.py — 5 sectors × 6 letters.
# Polish letters intentionally absent in this fallback layout; switch to
# BigramAdaptiveLayout for a Polish-aware ring.
DEFAULT_GRID = {
    0: ["a", "b", "c", "d", "e", "f"],
    1: ["g", "h", "i", "j", "k", "l"],
    2: ["m", "n", "o", "p", "q", "r"],
    3: ["s", "t", "u", "v", "w", "x"],
    4: ["y", "z", " ", ".", ",", "?"],
}


class StaticGridLayout:
    """Constant grid layout. Doesn't react to typed context."""

    def __init__(self, grid: dict[int, list[str]] | None = None):
        self._grid = dict(grid if grid is not None else DEFAULT_GRID)

    @property
    def n_sectors(self) -> int:
        return len(self._grid)

    def letters_in_sector(self, sector_idx: int) -> list[str]:
        return list(self._grid[sector_idx])

    def on_letter_committed(self, letter: str) -> None:
        # Static — nothing to update.
        pass

    def reset(self) -> None:
        pass


# Sanity check: this class satisfies SpellerLayout.
_check: SpellerLayout = StaticGridLayout()
del _check
