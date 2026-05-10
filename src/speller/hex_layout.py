"""HexLayout — Hex-O-Spell-style 6×6 layout.

Letters are partitioned into 6 hexagons, each containing 6 letters. The
partition is *static* (does not depend on previously typed letters) but
the contents are arranged so that letters with similar frequency are
distributed across hexagons (load-balancing).

Default partition (English):
    Hex 0: e t a o i n     (top-6 most frequent)
    Hex 1: s h r d l c
    Hex 2: u m w f g y
    Hex 3: p b v k j x
    Hex 4: q z space . ,
    Hex 5: ? ' (rest filled with placeholders)

Why static, not bigram-adaptive?
    Hex-O-Spell is the "classic" Blankertz-style layout from BBCI, which
    uses static partitioning. The bigram-adaptive version is implemented
    separately as BigramAdaptiveLayout (Ring-O-Spell). Having both lets
    you compare them empirically — same speller mechanics, different
    content allocation strategy.
"""

from __future__ import annotations

from src.speller.layout import SpellerLayout


# Default English allocation. 6 hexagons × 6 letters = 36 slots.
# Top-30 letters by frequency, then padded with placeholders.
DEFAULT_HEX_GRID_ENGLISH: list[list[str]] = [
    list("etaoin"),
    list("shrdlc"),
    list("umwfgy"),
    list("pbvkjx"),
    list("qz .,?"),
    list("'·····"),  # apostrophe + 5 placeholders
]


class HexLayout:
    """Static 6×6 hex-O-spell layout.

    Parameters
    ----------
    grid : list[list[str]] or None
        Custom 6×6 partition. Defaults to DEFAULT_HEX_GRID_ENGLISH.
        Each inner list must have exactly 6 elements.
    """

    def __init__(self, grid: list[list[str]] | None = None):
        self._grid = [list(row) for row in (grid if grid is not None else DEFAULT_HEX_GRID_ENGLISH)]
        if len(self._grid) != 6:
            raise ValueError(f"HexLayout requires 6 hexagons, got {len(self._grid)}")
        for i, row in enumerate(self._grid):
            if len(row) != 6:
                raise ValueError(f"Hexagon {i} must have 6 letters, got {len(row)}")

    @property
    def n_sectors(self) -> int:
        return 6

    def letters_in_sector(self, sector_idx: int) -> list[str]:
        return list(self._grid[sector_idx])

    def on_letter_committed(self, letter: str) -> None:
        # Static layout — no context update.
        pass

    def reset(self) -> None:
        pass


_check: SpellerLayout = HexLayout()
del _check
