"""BCI inference pipeline: EEG stream → MI classifier + blink detector → Speller.

Architectural notes
-------------------
Three signals drive the speller, mapped onto the keyboard's state machine:

    LEFT motor imagery   →  speller.move(LEFT)    — navigate cursor left
    RIGHT motor imagery  →  speller.move(RIGHT)   — navigate cursor right
    Blink (deliberate)   →  speller.select()      — confirm / commit

The MI classifier produces noisy per-epoch predictions (~80% accuracy at best),
so we don't act on a single epoch. We accumulate predictions in a sliding
window and require a quorum (`min_votes_required` out of `smoothing_window`)
agreeing on the same label before issuing a `move()`. Below quorum, no action.

The blink detector returns binary deliberate-blink events from the same EEG
stream (or a separate EOG stream — see `eog_headset`). We debounce blinks
(`min_blink_interval_s`) so one physical blink can't fire two `select()`
actions across overlapping epochs.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from src.eeg_headset.blink_detector import BlinkDetector, BlinkSource
from src.eeg_headset.eeg_headset import EEGHeadset
from src.eeg_headset.drivers import HeadsetDriver, MockDriver, PlaybackDriver
from src.eeg_headset.headset_config import HeadsetConfig, HeadsetModel
from src.inference.starter_bci import (
    LABELS_BINARY,
    LABELS_THREEWAY,
    derive_hparams,
    load_model,
    preprocess,
)
from src.speller import Speller, Direction
from src.speller.state import SpellerStateIdle


@dataclass
class PipelineConfig:
    """All tunable parameters of the pipeline in one place."""

    # MI classifier
    confidence_cutoff: float = 0.55
    smoothing_window: int = 5
    min_votes_required: int = 3
    epoch_seconds: int = 4

    # Preprocessing — must match the model's training distribution.
    # Defaults match `new_full_binary_all_channels_4way` from motor-imagery-AI.
    preprocess_sfreq: float = 160.0
    preprocess_bandpass_low: float = 0.0
    preprocess_bandpass_high: float = 49.0

    # Blink debounce: minimum seconds between two consecutive accepted blinks.
    min_blink_interval_s: float = 1.5

    # Loop pacing
    poll_interval_s: float = 0.1
    inter_epoch_pause_s: float = 1.0


class BCIPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        headset: EEGHeadset,
        speller: Speller,
        blink_detector: BlinkDetector,
        config: Optional[PipelineConfig] = None,
        # Inject a BioAmp headset here (Etap 5) when EOG comes from a separate stream.
        # If None, blink_detector reads from the main EEG headset.
        eog_headset: Optional[EEGHeadset] = None,
        on_action: Optional[Callable[[str], None]] = None,
    ):
        self.model = model
        self.headset = headset
        self.speller = speller
        self.blink_detector = blink_detector
        self.eog_headset = eog_headset
        self.config = config or PipelineConfig()
        self.on_action = on_action or (lambda action: None)

        self._prediction_history: deque[str] = deque(maxlen=self.config.smoothing_window)
        self._last_blink_ts: float = 0.0

        # Pick label map by output dim — robust to swapping 2-class vs 3-class checkpoints.
        n_classes = model.fc.out_features
        if n_classes == 2:
            self._labels = LABELS_BINARY
        elif n_classes == 3:
            self._labels = LABELS_THREEWAY
        else:
            raise ValueError(
                f"Model has {n_classes} output classes; pipeline expects 2 or 3."
            )

        # Cache derived time_points to detect shape mismatches early.
        hparams = derive_hparams(model.state_dict())
        self._expected_time_points = hparams["time_points"]
        self._expected_chans = hparams["chans"]

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self.headset.connect()
        self.headset.start()
        if self.eog_headset is not None and self.eog_headset is not self.headset:
            self.eog_headset.connect()
            self.eog_headset.start()

    def stop(self) -> None:
        self.headset.stop()
        self.headset.disconnect()
        if self.eog_headset is not None and self.eog_headset is not self.headset:
            self.eog_headset.stop()
            self.eog_headset.disconnect()

    # ─── Single epoch step ──────────────────────────────────────────────────

    def step(self) -> None:
        """Acquire one epoch, check blink, run MI classifier, update speller."""
        self.headset.annotate("predict")
        if self.eog_headset is not None and self.eog_headset is not self.headset:
            self.eog_headset.annotate("predict")

        # Spin-poll until we have ~epoch_seconds worth of data.
        # The EEGHeadset.get_output() loop already handles edge cases, but a few
        # explicit polls reduce the chance we hit the fallback zero-padding.
        if self.config.poll_interval_s > 0:
            n_polls = int(self.config.epoch_seconds / self.config.poll_interval_s)
            for _ in range(n_polls):
                self.headset.poll()
                if self.eog_headset is not None and self.eog_headset is not self.headset:
                    self.eog_headset.poll()
                time.sleep(self.config.poll_interval_s)

        epoch_data = self.headset.get_output(seconds=self.config.epoch_seconds)
        epoch_data = self._fit_to_model(epoch_data)

        # ─── 1. Blink (deliberate) → select ─────────────────────────────────
        blink_data = self._get_blink_window()
        sfreq = (
            self.eog_headset.sample_rate
            if self.eog_headset is not None
            else self.headset.sample_rate
        )
        if self.blink_detector.detect(blink_data, sfreq=sfreq):
            now = time.monotonic()
            if now - self._last_blink_ts >= self.config.min_blink_interval_s:
                self._last_blink_ts = now
                print("👁️  Deliberate blink → select")
                self._safe_select()
                # Don't run MI classification on the same epoch — selection already happened.
                return
            else:
                print("👁️  Blink suppressed (debounce)")

        # ─── 2. MI classification → move ────────────────────────────────────
        try:
            tensor = preprocess(
                epoch_data,
                sfreq=self.config.preprocess_sfreq,
                bandpass_low=self.config.preprocess_bandpass_low,
                bandpass_high=self.config.preprocess_bandpass_high,
            )
            with torch.no_grad():
                probs = self.model(tensor).softmax(dim=1)[0]
                pred_idx = int(probs.argmax().item())
                confidence = float(probs[pred_idx].item())

            label = self._labels[pred_idx]
            print(f"MI: {label} (conf {confidence:.0%})")

            if confidence >= self.config.confidence_cutoff:
                self._prediction_history.append(label)
            else:
                self._prediction_history.append("uncertain")

            self._maybe_commit_action()

        except Exception as e:
            print(f"Prediction failed: {e}")

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _fit_to_model(self, epoch: np.ndarray) -> np.ndarray:
        """Pad/truncate the epoch to (expected_chans, expected_time_points).

        Channels: warns if mismatched. Padding channels with zeros distorts the
        spatial filter — for a real 16-ch → 64-ch fit you need spherical
        interpolation, not zero-pad.
        Time: edge-padded if too short, truncated if too long.
        """
        if epoch.shape[0] != self._expected_chans:
            print(
                f"Warning: model expects {self._expected_chans} channels, "
                f"got {epoch.shape[0]}. Spatial filter will be miscalibrated."
            )
            # Crude fallback: tile/truncate channels. This is wrong for real signal
            # but lets the pipeline run end-to-end on mismatched mocks.
            if epoch.shape[0] < self._expected_chans:
                reps = (self._expected_chans + epoch.shape[0] - 1) // epoch.shape[0]
                epoch = np.tile(epoch, (reps, 1))[: self._expected_chans, :]
            else:
                epoch = epoch[: self._expected_chans, :]

        T = self._expected_time_points
        if epoch.shape[1] < T:
            epoch = np.pad(epoch, ((0, 0), (0, T - epoch.shape[1])), mode="edge")
        elif epoch.shape[1] > T:
            epoch = epoch[:, :T]
        return epoch

    def _get_blink_window(self) -> np.ndarray:
        """Pull the most-recent epoch from whichever stream feeds the blink detector."""
        source = self.eog_headset if self.eog_headset is not None else self.headset
        return source.get_output(seconds=self.config.epoch_seconds)

    def _maybe_commit_action(self) -> None:
        """Commit a `move()` action if the smoothing window agrees by quorum."""
        if len(self._prediction_history) < self.config.smoothing_window:
            return

        most_common, count = Counter(self._prediction_history).most_common(1)[0]
        if count < self.config.min_votes_required:
            return

        print(
            f"→ Action confirmed: {most_common} "
            f"({count}/{self.config.smoothing_window})"
        )

        match most_common:
            case "left_hand":
                self._safe_move(Direction.LEFT)
            case "right_hand":
                self._safe_move(Direction.RIGHT)
            case "rest" | "uncertain":
                pass  # No movement, but consume the buffer below.

        self._prediction_history.clear()

    def _safe_move(self, direction: Direction) -> None:
        try:
            self.speller.move(direction)
            self.on_action(f"move:{direction.name}")
        except Exception as e:
            print(
                f"Move({direction.name}) ignored in state "
                f"{type(self.speller.state).__name__}: {e}"
            )

    def _safe_select(self) -> None:
        try:
            self.speller.select()
            self.on_action(f"select:{type(self.speller.state).__name__}")
        except Exception as e:
            print(f"Select ignored in state {type(self.speller.state).__name__}: {e}")

    # ─── Main loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        print("BCI pipeline started. Press Ctrl+C to stop.")
        try:
            while True:
                print("\n--- Starting new epoch ---")
                self.step()
                time.sleep(self.config.inter_epoch_pause_s)
        except KeyboardInterrupt:
            print("\nStopping BCI workflow...")
        finally:
            self.stop()
            print("Disconnected.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — minimal demo wiring for development with PlaybackDriver/MockDriver.
# For real BrainAccess + BioAmp wiring, see src/eeg_headset/cmd/run_keyboard.py
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("Loading EEGNet model...")
    model = load_model()

    print("Initializing demo driver (PlaybackDriver from data/X.npy)...")
    config = HeadsetConfig(model=HeadsetModel.SAMPLE_64CH)
    driver: HeadsetDriver
    try:
        driver = PlaybackDriver(config=config, source="data/X.npy", loop=True)
    except FileNotFoundError:
        print("data/X.npy not found, falling back to MockDriver.")
        driver = MockDriver(config=config)

    headset = EEGHeadset(driver, buffer_size_seconds=10)
    speller = Speller()
    # Start in Idle. Pipeline will issue the first `select()` itself when a blink
    # arrives, advancing through Idle → Writing → SectorNavigation → ... naturally.
    speller.state = SpellerStateIdle()

    blink = BlinkDetector(source=BlinkSource.PREFRONTAL_EEG, frontal_channel_idx=[0, 1])
    pipeline = BCIPipeline(model, headset, speller, blink)

    pipeline.start()
    pipeline.run()


if __name__ == "__main__":
    main()
