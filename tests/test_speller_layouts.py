"""Tests for SpellerLayout implementations and Speller integration."""

import json
from pathlib import Path

import pytest

from src.speller import Speller
from src.speller.bigram_layout import BigramAdaptiveLayout, N_SECTORS, LETTERS_PER_SECTOR, RING_SIZE
from src.speller.layout import SpellerLayout
from src.speller.state import (
    Direction,
    SpellerStateIdle,
    SpellerStateLetterNavigation,
    SpellerStateSectorNavigation,
    SpellerStateWriting,
)
from src.speller.static_layout import StaticGridLayout


# ─── StaticGridLayout ───────────────────────────────────────────────────────────


def test_static_grid_default_has_5_sectors() -> None:
    layout = StaticGridLayout()
    assert layout.n_sectors == 5


def test_static_grid_letters_in_sector() -> None:
    layout = StaticGridLayout()
    assert layout.letters_in_sector(0) == ["a", "b", "c", "d", "e", "f"]


def test_static_grid_committing_letter_does_not_change_layout() -> None:
    layout = StaticGridLayout()
    before = layout.letters_in_sector(0)
    layout.on_letter_committed("z")
    assert layout.letters_in_sector(0) == before


def test_static_grid_satisfies_protocol() -> None:
    assert isinstance(StaticGridLayout(), SpellerLayout)


# ─── BigramAdaptiveLayout ────────────────────────────────────────────────────────


@pytest.fixture
def bigram_layout() -> BigramAdaptiveLayout:
    return BigramAdaptiveLayout()


def test_bigram_layout_has_6x6_structure(bigram_layout: BigramAdaptiveLayout) -> None:
    assert bigram_layout.n_sectors == N_SECTORS
    for i in range(N_SECTORS):
        assert len(bigram_layout.letters_in_sector(i)) == LETTERS_PER_SECTOR


def test_bigram_layout_total_positions_equal_36(bigram_layout: BigramAdaptiveLayout) -> None:
    all_letters = sum((bigram_layout.letters_in_sector(i) for i in range(N_SECTORS)), start=[])
    assert len(all_letters) == RING_SIZE


def test_bigram_layout_starts_with_first_word_distribution(bigram_layout: BigramAdaptiveLayout) -> None:
    """Cold start uses ' ' as previous letter → sector 0 holds typical Polish word-starters."""
    sector_0 = bigram_layout.letters_in_sector(0)
    # 'p' / 's' / 'n' / 'w' are extremely common Polish word starts. At least 2 should be in sector 0.
    common_starters = {"p", "s", "n", "w"}
    assert len(common_starters.intersection(sector_0)) >= 2


def test_bigram_layout_adapts_to_committed_letter(bigram_layout: BigramAdaptiveLayout) -> None:
    """After 'k', 'o' should jump up high (top 6 — 'ko' is a very common Polish bigram)."""
    sector_0_before = bigram_layout.letters_in_sector(0)
    bigram_layout.on_letter_committed("k")
    sector_0_after = bigram_layout.letters_in_sector(0)
    assert sector_0_before != sector_0_after
    assert "o" in sector_0_after


def test_bigram_layout_reset_returns_to_initial(bigram_layout: BigramAdaptiveLayout) -> None:
    initial = [bigram_layout.letters_in_sector(i) for i in range(N_SECTORS)]
    bigram_layout.on_letter_committed("k")
    bigram_layout.on_letter_committed("a")
    bigram_layout.reset()
    after_reset = [bigram_layout.letters_in_sector(i) for i in range(N_SECTORS)]
    assert initial == after_reset


def test_bigram_layout_placeholder_does_not_update_context() -> None:
    layout = BigramAdaptiveLayout()
    snapshot_before = [layout.letters_in_sector(i) for i in range(N_SECTORS)]
    layout.on_letter_committed("·")
    snapshot_after = [layout.letters_in_sector(i) for i in range(N_SECTORS)]
    assert snapshot_before == snapshot_after


def test_bigram_layout_satisfies_protocol() -> None:
    assert isinstance(BigramAdaptiveLayout(), SpellerLayout)


def test_bigram_layout_unknown_previous_letter_falls_back_to_unigram(tmp_path: Path) -> None:
    """If the bigram table has no entry for the last_letter, unigram is used."""
    fake_data = {
        "alphabet": ["a", "b", "c"],
        "unigram": {"a": 0.7, "b": 0.2, "c": 0.1},
        "bigram": {"a": {"b": 1.0}},  # only 'a' has continuations
    }
    path = tmp_path / "fake_bigrams.json"
    path.write_text(json.dumps(fake_data))

    layout = BigramAdaptiveLayout(bigrams_path=path, initial_letter="x")  # 'x' not in table
    sector_0 = layout.letters_in_sector(0)
    # Without bigram fallback to unigram, 'a' should be on top.
    assert sector_0[0] == "a"


# ─── Speller + Layout integration ────────────────────────────────────────────────


def test_speller_uses_static_grid_by_default() -> None:
    s = Speller()
    assert isinstance(s.layout, StaticGridLayout)


def test_speller_letter_commit_updates_bigram_layout() -> None:
    """After committing 'p' via the speller, layout reflects post-'p' bigrams."""
    typed: list[str] = []
    speller = Speller(layout=BigramAdaptiveLayout(), on_letter_select=lambda c: typed.append(c))

    # Idle -> Writing -> SectorNavigation -> LetterNavigation(sector 0, cursor 0).
    speller.select()
    speller.select()
    initial_sector_0 = speller.layout.letters_in_sector(0)
    first_letter = initial_sector_0[0]  # whatever bigram layout puts at top

    speller.select()  # enter sector 0
    speller.select()  # commit cursor 0 = first_letter

    assert typed == [first_letter]
    # After commit, layout has updated → sector 0 contents have shifted.
    assert speller.layout.letters_in_sector(0) != initial_sector_0


def test_speller_back_to_idle_resets_bigram_context() -> None:
    """Going back to Idle resets bigram context to initial (cold-start) state."""
    speller = Speller(layout=BigramAdaptiveLayout())
    # Commit 'k' to shift context.
    speller.select()  # Idle -> Writing
    speller.select()  # Writing -> SectorNavigation
    # find sector containing 'k' and select it
    target_sector = next(
        i for i in range(N_SECTORS)
        if "k" in speller.layout.letters_in_sector(i)
    )
    for _ in range(target_sector):
        speller.move(Direction.RIGHT)
    speller.select()  # enter sector
    cursor_pos = speller.layout.letters_in_sector(target_sector).index("k")
    for _ in range(cursor_pos):
        speller.move(Direction.RIGHT)
    speller.select()  # commit 'k'

    sectors_after_k = [speller.layout.letters_in_sector(i) for i in range(N_SECTORS)]

    # Now back fully to Idle.
    speller.back()  # Writing -> Idle
    sectors_after_reset = [speller.layout.letters_in_sector(i) for i in range(N_SECTORS)]

    assert isinstance(speller.state, SpellerStateIdle)
    assert sectors_after_k != sectors_after_reset, "context should have been reset"
