"""TreeSpeller — Huffman-style binary tree keyboard.

Each L/R signal moves down one level in a binary tree. A leaf node commits
its letter and resets the cursor to the root. Letters are arranged so that
the most frequent ones have the shortest paths (Huffman coding).

This speller does NOT use the SpellerLayout / SpellerState abstractions —
it has different navigation mechanics (depth-first instead of sector→letter)
and embedding it into that protocol would muddy the model. Instead, it
exposes the same external API (`move(direction)`, `select()`, `back()`)
so BCIPipeline can drive it identically.

Design choices
--------------
- Build tree from unigrams via standard Huffman coding (priority queue,
  merge two least-frequent symbols).
- Tree depth is unbounded in principle but with 31 symbols (26 letters +
  space + 4 punctuation) and a balanced-ish frequency distribution,
  expect depths 4-7 for most letters, ~3-4 for top letters like 'e' 't'.
- `select` (blink) is *not* required to commit a letter — letter is
  committed automatically when a leaf is reached. `select` resets cursor
  to root (useful if the user got lost and wants to start over).
  Alternative: `select` could mean "I'm done navigating, pick whichever
  letter I'm on", but that's ambiguous at internal nodes.

Caveat: Huffman tree is not necessarily ergonomic for BCI. Frequent letters
have short paths but require precise L/R sequences. Mistakes accumulate.
For comparison studies vs Hex/Ring, this is fine; for production use,
consider depth-balancing tweaks.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.speller.state import Direction


DEFAULT_BIGRAMS_PATH_EN = (
    Path(__file__).resolve().parents[2] / "data" / "language" / "english_bigrams.json"
)


@dataclass
class TreeNode:
    """Internal node (left+right children) or leaf (letter+frequency)."""

    letter: Optional[str] = None  # set iff leaf
    freq: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    # Used for stable Huffman ordering when frequencies tie.
    _id: int = field(default=0, compare=False)

    @property
    def is_leaf(self) -> bool:
        return self.letter is not None


def build_huffman_tree(letter_frequencies: dict[str, float]) -> TreeNode:
    """Standard Huffman coding: merge two lowest-frequency nodes until one remains."""
    counter = 0
    heap: list[tuple[float, int, TreeNode]] = []
    for letter, freq in letter_frequencies.items():
        heapq.heappush(heap, (freq, counter, TreeNode(letter=letter, freq=freq, _id=counter)))
        counter += 1

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = TreeNode(letter=None, freq=f1 + f2, left=n1, right=n2, _id=counter)
        heapq.heappush(heap, (merged.freq, counter, merged))
        counter += 1

    return heap[0][2]


def collect_leaves(node: TreeNode) -> list[str]:
    """Return all letters under this subtree, in left-to-right traversal order."""
    if node.is_leaf:
        return [node.letter]  # type: ignore[list-item]
    out: list[str] = []
    if node.left:
        out.extend(collect_leaves(node.left))
    if node.right:
        out.extend(collect_leaves(node.right))
    return out


def path_to_letter(root: TreeNode, target: str) -> Optional[list[Direction]]:
    """Find the L/R sequence that reaches `target`, for visualization or testing."""
    def _find(node: TreeNode, path: list[Direction]) -> Optional[list[Direction]]:
        if node.is_leaf:
            return path if node.letter == target else None
        if node.left:
            r = _find(node.left, path + [Direction.LEFT])
            if r is not None:
                return r
        if node.right:
            r = _find(node.right, path + [Direction.RIGHT])
            if r is not None:
                return r
        return None
    return _find(root, [])


class TreeSpeller:
    """Binary-tree BCI speller with the same external API as Speller.

    Public methods (intentionally identical to Speller):
        move(direction)  — descend left or right
        select()         — reset cursor to root (escape-hatch)
        back()           — same as select; kept for API parity

    Letter commit is automatic when a leaf is reached after move().

    Parameters
    ----------
    letter_frequencies : dict[str, float] or None
        Letter → relative frequency mapping. Must sum to 1 (or close to).
        Defaults to English unigrams from the bigram table.
    on_letter_select : callable, optional
        Called whenever a letter is committed.
    """

    def __init__(
        self,
        letter_frequencies: Optional[dict[str, float]] = None,
        on_letter_select: Optional[Callable[[str], None]] = None,
    ):
        if letter_frequencies is None:
            with open(DEFAULT_BIGRAMS_PATH_EN, encoding="utf-8") as f:
                data = json.load(f)
            letter_frequencies = data["unigram"]

        # Huffman doesn't handle zero-frequency well — give every letter a tiny floor.
        letter_frequencies = {l: max(f, 1e-6) for l, f in letter_frequencies.items()}

        self.root: TreeNode = build_huffman_tree(letter_frequencies)
        self.cursor: TreeNode = self.root

        self.on_letter_select_hook = on_letter_select or (lambda c: print(f"Selected: {c}"))

    # ─── External API parity with Speller ───────────────────────────────────

    def move(self, direction: Direction) -> None:
        """Descend left or right. If we land on a leaf, auto-commit that letter."""
        if self.cursor.is_leaf:
            # Already at a leaf; ignore further movement until select() resets.
            return
        nxt = self.cursor.left if direction == Direction.LEFT else self.cursor.right
        if nxt is None:
            # Defensive: tree is malformed if we hit this. Treat as no-op.
            return
        self.cursor = nxt
        if self.cursor.is_leaf:
            self._commit_letter(self.cursor.letter)  # type: ignore[arg-type]

    def select(self) -> None:
        """Reset cursor to root (escape-hatch if user got lost)."""
        self.cursor = self.root

    def back(self) -> None:
        """Alias for select(): reset to root."""
        self.cursor = self.root

    # ─── Helpers for GUI rendering ──────────────────────────────────────────

    def left_subtree_letters(self) -> list[str]:
        return collect_leaves(self.cursor.left) if (self.cursor.left and not self.cursor.is_leaf) else []

    def right_subtree_letters(self) -> list[str]:
        return collect_leaves(self.cursor.right) if (self.cursor.right and not self.cursor.is_leaf) else []

    def _commit_letter(self, letter: str) -> None:
        self.on_letter_select_hook(letter)
        self.cursor = self.root  # auto-reset for next letter
