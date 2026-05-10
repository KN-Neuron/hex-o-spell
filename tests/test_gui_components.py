"""Tests for the new GUI-side modules: WordSuggester, HexLayout, TreeSpeller."""

import json
from pathlib import Path

import pytest

from src.speller.hex_layout import HexLayout, DEFAULT_HEX_GRID_ENGLISH
from src.speller.layout import SpellerLayout
from src.speller.state import Direction
from src.speller.tree_speller import (
    TreeSpeller,
    build_huffman_tree,
    collect_leaves,
    path_to_letter,
)
from src.speller.word_suggester import WordSuggester


# ─── HexLayout ───────────────────────────────────────────────────────────────────


def test_hex_layout_has_6_sectors() -> None:
    layout = HexLayout()
    assert layout.n_sectors == 6


def test_hex_layout_each_sector_has_6_letters() -> None:
    layout = HexLayout()
    for i in range(6):
        assert len(layout.letters_in_sector(i)) == 6


def test_hex_layout_top_sector_has_top_letters() -> None:
    """The first hexagon should hold the most common English letters."""
    layout = HexLayout()
    sector_0 = set(layout.letters_in_sector(0))
    # Top-6 most frequent English letters
    assert sector_0 == set("etaoin")


def test_hex_layout_static_does_not_change() -> None:
    layout = HexLayout()
    before = layout.letters_in_sector(0)
    layout.on_letter_committed("z")
    layout.on_letter_committed("e")
    layout.reset()
    assert layout.letters_in_sector(0) == before


def test_hex_layout_satisfies_protocol() -> None:
    assert isinstance(HexLayout(), SpellerLayout)


def test_hex_layout_custom_grid() -> None:
    custom = [list("abcdef"), list("ghijkl"), list("mnopqr"),
              list("stuvwx"), list("yz   ?"), list("..,..,")]
    layout = HexLayout(grid=custom)
    assert layout.letters_in_sector(0) == list("abcdef")


def test_hex_layout_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        HexLayout(grid=[list("abcdef")] * 5)  # only 5 hexagons
    with pytest.raises(ValueError):
        HexLayout(grid=[list("abc")] * 6)  # 3 letters per hexagon


# ─── TreeSpeller ────────────────────────────────────────────────────────────────


def test_huffman_tree_assigns_shorter_paths_to_frequent_letters() -> None:
    """Higher frequency → shallower in the tree."""
    freqs = {"a": 0.5, "b": 0.25, "c": 0.125, "d": 0.125}
    root = build_huffman_tree(freqs)
    depths = {l: len(path_to_letter(root, l)) for l in freqs}
    assert depths["a"] <= depths["b"]
    assert depths["b"] <= depths["c"]
    assert depths["b"] <= depths["d"]


def test_collect_leaves_returns_all_letters() -> None:
    freqs = {"a": 0.5, "b": 0.3, "c": 0.2}
    root = build_huffman_tree(freqs)
    assert set(collect_leaves(root)) == {"a", "b", "c"}


def test_tree_speller_navigates_to_letter() -> None:
    """Navigating along the path to a letter commits that letter."""
    typed: list[str] = []
    speller = TreeSpeller(on_letter_select=lambda c: typed.append(c))
    # Find path to 'e' (most common letter, should have shortest path).
    path = path_to_letter(speller.root, "e")
    assert path is not None
    for direction in path:
        speller.move(direction)
    assert typed == ["e"]


def test_tree_speller_resets_to_root_after_commit() -> None:
    """After committing, cursor is back at root for the next letter."""
    speller = TreeSpeller()
    path = path_to_letter(speller.root, "e")
    for d in path:
        speller.move(d)
    assert speller.cursor is speller.root


def test_tree_speller_select_resets_cursor() -> None:
    """select() during navigation resets to root (escape hatch)."""
    speller = TreeSpeller()
    path = path_to_letter(speller.root, "e")
    speller.move(path[0])  # first step
    assert speller.cursor is not speller.root
    speller.select()
    assert speller.cursor is speller.root


def test_tree_speller_back_resets_cursor() -> None:
    """back() is alias for select() — both reset to root."""
    speller = TreeSpeller()
    path = path_to_letter(speller.root, "e")
    speller.move(path[0])
    speller.back()
    assert speller.cursor is speller.root


def test_tree_speller_subtree_letters() -> None:
    """left_subtree_letters / right_subtree_letters return non-overlapping letter sets."""
    speller = TreeSpeller()
    left = set(speller.left_subtree_letters())
    right = set(speller.right_subtree_letters())
    assert left & right == set()
    assert len(left) + len(right) >= 25  # all 26+ letters distributed


# ─── WordSuggester ───────────────────────────────────────────────────────────────


@pytest.fixture
def suggester() -> WordSuggester:
    return WordSuggester()


def test_word_suggester_top_global(suggester: WordSuggester) -> None:
    """Empty prefix returns the most frequent words."""
    top = suggester.suggest("", n_results=3)
    assert "the" in top  # always in top-3 of any English word list


def test_word_suggester_prefix_match(suggester: WordSuggester) -> None:
    """A specific prefix returns only words starting with it."""
    out = suggester.suggest("th", n_results=5)
    assert all(w.startswith("th") for w in out)


def test_word_suggester_no_match(suggester: WordSuggester) -> None:
    """A nonsense prefix returns an empty list, not an error."""
    out = suggester.suggest("zzzqxqx", n_results=3)
    assert out == []


def test_word_suggester_n_results_respected(suggester: WordSuggester) -> None:
    """Returns at most n_results items even when many words match."""
    out = suggester.suggest("a", n_results=2)
    assert len(out) <= 2


def test_word_suggester_custom_path(tmp_path: Path) -> None:
    """Loads from a custom path correctly."""
    fake = {"words": [{"word": "alpha", "count": 100}, {"word": "beta", "count": 50}]}
    p = tmp_path / "fake_words.json"
    p.write_text(json.dumps(fake))
    s = WordSuggester(words_path=p)
    assert s.suggest("a", 1) == ["alpha"]
    assert s.suggest("b", 1) == ["beta"]
