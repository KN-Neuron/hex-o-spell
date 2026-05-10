"""Integration-ish tests for BCIPipeline: signal sequencing → speller transitions.

These tests don't run a real model — they inject deterministic predictions
and blink events to verify the pipeline correctly drives the speller's
state machine and respects debounce.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.eeg_headset.blink_detector import BlinkDetector, BlinkSource
from src.inference.pipeline import BCIPipeline, PipelineConfig
from src.speller import Speller, Direction
from src.speller.state import (
    SpellerStateIdle,
    SpellerStateLetterNavigation,
    SpellerStateSectorNavigation,
    SpellerStateWriting,
)


# ─── Test doubles ───────────────────────────────────────────────────────────────


class _FakeModel(torch.nn.Module):
    """Tiny torch module that returns a fixed logit vector regardless of input."""

    def __init__(self, n_classes: int = 2, time_points: int = 640, chans: int = 64):
        super().__init__()
        self.fc = torch.nn.Linear(8, n_classes)  # only out_features is read by pipeline
        # Stub state_dict layers so derive_hparams() works:
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (1, 80), padding="same", bias=False),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, (chans, 1), groups=16, bias=False),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, (1, 16), groups=32, padding="same", bias=False),
            torch.nn.Conv2d(32, 32, 1, bias=False),
        )
        # Adjust fc dim to match time_points/(4*8) * f2 = time_points/32 * 32 = time_points
        # But our test's fake model doesn't actually compute through these blocks, so:
        self.fc = torch.nn.Linear((time_points // 32) * 32, n_classes)
        self._next_logits = torch.zeros(1, n_classes)

    def set_next(self, label_idx: int, confidence: float = 0.99) -> None:
        """Make the next forward() return a softmax peaked at label_idx with given confidence."""
        n = self.fc.out_features
        # Spread (1 - confidence) uniformly over the other classes.
        # Then take logs to construct logits whose softmax equals that distribution exactly.
        probs = np.full(n, (1.0 - confidence) / max(n - 1, 1))
        probs[label_idx] = confidence
        # Numerically safe log; softmax is invariant to additive constants.
        logits = torch.log(torch.tensor(probs, dtype=torch.float32) + 1e-12).unsqueeze(0)
        self._next_logits = logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._next_logits


class _ScriptedBlink:
    """A blink-detector stand-in with a pre-set return queue."""

    def __init__(self, schedule: list[bool]) -> None:
        self.schedule = list(schedule)
        self.calls = 0

    def detect(self, window: np.ndarray, sfreq: float) -> bool:
        if self.calls < len(self.schedule):
            ret = self.schedule[self.calls]
        else:
            ret = False
        self.calls += 1
        return ret


class _StaticHeadset:
    """An EEGHeadset stand-in returning a constant epoch of zeros."""

    def __init__(self, n_channels: int = 64, sfreq: float = 160.0) -> None:
        self._n = n_channels
        self.sample_rate = sfreq
        self.connected = False
        self.streamed = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def start(self) -> None:
        self.streamed = True

    def stop(self) -> None:
        self.streamed = False

    def annotate(self, label: str) -> None:
        pass

    def poll(self) -> None:
        pass

    def get_output(self, seconds: int = 1) -> np.ndarray:
        return np.zeros((self._n, int(seconds * self.sample_rate)), dtype=np.float32)


# ─── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def speller() -> Speller:
    s = Speller()
    s.state = SpellerStateIdle()
    return s


@pytest.fixture
def fast_config() -> PipelineConfig:
    """Skip waiting and shrink polling to make tests fast."""
    return PipelineConfig(
        smoothing_window=3,
        min_votes_required=2,
        epoch_seconds=4,
        poll_interval_s=0.0,
        inter_epoch_pause_s=0.0,
        min_blink_interval_s=0.0,  # disable debounce for most tests
    )


# ─── Tests ───────────────────────────────────────────────────────────────────────


def test_blink_advances_idle_to_writing(speller, fast_config):
    model = _FakeModel(n_classes=2)
    headset = _StaticHeadset()
    blink = _ScriptedBlink([True])

    pipeline = BCIPipeline(model, headset, speller, blink, config=fast_config)
    assert isinstance(speller.state, SpellerStateIdle)

    pipeline.step()
    assert isinstance(speller.state, SpellerStateWriting)


def test_two_blinks_reach_sector_navigation(speller, fast_config):
    """Idle → Writing → SectorNavigation in two blinks."""
    model = _FakeModel(n_classes=2)
    headset = _StaticHeadset()
    blink = _ScriptedBlink([True, True])

    pipeline = BCIPipeline(model, headset, speller, blink, config=fast_config)

    pipeline.step()
    pipeline.step()
    assert isinstance(speller.state, SpellerStateSectorNavigation)


def test_lr_quorum_advances_cursor_in_sector_navigation(speller, fast_config):
    """In SectorNavigation, after quorum of right_hand predictions cursor moves."""
    speller.state = SpellerStateSectorNavigation()
    model = _FakeModel(n_classes=2)
    headset = _StaticHeadset()
    blink = _ScriptedBlink([False, False, False])

    pipeline = BCIPipeline(model, headset, speller, blink, config=fast_config)

    # 3 right_hand predictions in a window of 3 with quorum 2 → action fires.
    model.set_next(label_idx=1, confidence=0.99)  # 1 = right_hand for binary labels
    pipeline.step()
    pipeline.step()
    pipeline.step()

    assert isinstance(speller.state, SpellerStateSectorNavigation)
    assert speller.state.cursor == 1


def test_blink_in_sector_navigation_selects_letter_navigation(speller, fast_config):
    """Blink in SectorNavigation → LetterNavigation with correct sector."""
    speller.state = SpellerStateSectorNavigation()
    speller.state.cursor = 2
    model = _FakeModel(n_classes=2)
    headset = _StaticHeadset()
    blink = _ScriptedBlink([True])

    pipeline = BCIPipeline(model, headset, speller, blink, config=fast_config)

    pipeline.step()
    assert isinstance(speller.state, SpellerStateLetterNavigation)
    assert speller.state.selected_sector == 2


def test_blink_debounce_suppresses_double_select(speller):
    """With debounce enabled, two consecutive blinks within interval count as one."""
    config = PipelineConfig(
        smoothing_window=3,
        min_votes_required=2,
        epoch_seconds=4,
        poll_interval_s=0.0,
        inter_epoch_pause_s=0.0,
        min_blink_interval_s=10.0,  # long interval → second blink suppressed
    )
    model = _FakeModel(n_classes=2)
    headset = _StaticHeadset()
    blink = _ScriptedBlink([True, True])

    pipeline = BCIPipeline(model, headset, speller, blink, config=config)

    pipeline.step()  # accepted  Idle → Writing
    pipeline.step()  # debounced — should NOT advance further
    assert isinstance(speller.state, SpellerStateWriting)


def test_letter_selection_fires_callback(fast_config):
    """In LetterNavigation, a blink commits the letter via on_letter_select callback."""
    selected: list[str] = []
    speller = Speller(on_letter_select=lambda c: selected.append(c))
    speller.state = SpellerStateLetterNavigation(selected_sector=0)
    speller.state.cursor = 0  # 'a'

    model = _FakeModel(n_classes=2)
    headset = _StaticHeadset()
    blink = _ScriptedBlink([True])

    pipeline = BCIPipeline(model, headset, speller, blink, config=fast_config)
    pipeline.step()

    assert selected == ["a"]
    # State returns to Writing after letter commit (per state.py contract).
    assert isinstance(speller.state, SpellerStateWriting)


def test_low_confidence_predictions_do_not_commit(speller, fast_config):
    """If confidence < cutoff, predictions count as 'uncertain' and don't fire moves."""
    speller.state = SpellerStateSectorNavigation()
    model = _FakeModel(n_classes=2)
    headset = _StaticHeadset()
    blink = _ScriptedBlink([False, False, False])

    pipeline = BCIPipeline(model, headset, speller, blink, config=fast_config)

    # Below cutoff → all entries land in history as "uncertain".
    model.set_next(label_idx=1, confidence=0.51)  # below default 0.55
    pipeline.step()
    pipeline.step()
    pipeline.step()

    # Cursor stayed at initial position.
    assert speller.state.cursor == 0
