"""ScriptedDriver — synthesize EEG that should classify as a target intent.

Use case: end-to-end regression test from a scripted intent sequence
[LEFT, RIGHT, BLINK, ...] all the way through the MI classifier and blink
detector into the Speller's state machine.

Why not just inject high-level signals at the speller level?
    Because then we'd skip half the system. The point of having a "scripted"
    mode is to verify the *entire chain* — including epoch buffering, the
    classifier's softmax cutoff, the smoothing window, and blink debounce —
    not just the speller. To inject at the speller, just call
    speller.move()/select() directly; you don't need a driver.

How it works
------------
The driver takes a sequence of `Intent` enums and a `seconds_per_intent`
budget. It generates `n_channels × seconds_per_intent × sample_rate` of
synthetic EEG that loosely mimics the spectral signatures of the intents:

  - LEFT motor imagery: alpha desynchronization (8-12 Hz) on right-hemisphere
    motor channels (channels indexed by `right_motor_channels`).
  - RIGHT motor imagery: alpha desynchronization on left-hemisphere motor
    channels.
  - REST: pink-ish background noise on all channels.
  - BLINK: large slow positive deflection on prefrontal channels (Fp1, Fp2),
    duration 250 ms (within the deliberate-blink window).

The synthesis is intentionally simple — it's a regression test, not a
calibration. Don't expect 100% classifier accuracy on this data; expect
"high enough confidence often enough that the smoothing window commits the
right action within a few epochs."

This driver is NOT a substitute for real data. Don't draw conclusions about
classifier accuracy from it.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.eeg_headset.headset_config import HeadsetConfig


class Intent(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    REST = "rest"
    BLINK = "blink"


@dataclass
class ScriptedDriverConfig:
    """Tunables for the synthetic EEG generator."""

    seconds_per_intent: float = 4.0
    sample_rate_hz: int = 160  # match PhysioNet / final_best.pth
    n_channels: int = 64

    # Channels (indices) where left-hand MI desynchronizes alpha.
    # PhysioNet 64-channel layout: C4, CP4, FC4 are around indices 11, 50, 18.
    # Using a few rough indices is fine for synthesis purposes.
    right_motor_channels: tuple[int, ...] = (11, 13, 50)
    # Channels where right-hand MI desynchronizes alpha (left hemisphere).
    left_motor_channels: tuple[int, ...] = (9, 8, 48)
    # Prefrontal channels for blink artifacts (Fp1, Fp2 in PhysioNet layout).
    prefrontal_channels: tuple[int, ...] = (21, 22)

    background_amplitude_uv: float = 5.0
    alpha_amplitude_uv: float = 15.0  # baseline alpha rhythm
    blink_amplitude_uv: float = 200.0  # deliberate blink

    rng_seed: int = 1337


class ScriptedDriver:
    """A HeadsetDriver that emits scripted-intent EEG when polled.

    The intent queue is consumed lazily: each `read_available_samples` call
    advances real-time elapsed by the wall clock and returns however many
    samples the configured intent sequence would have produced in that
    interval. So at 160 Hz with a 4s/intent budget, polling at 10 Hz gets
    you ~16 samples per call, all from the current intent's synthesis.
    """

    def __init__(
        self,
        config: HeadsetConfig,
        intent_sequence: list[Intent],
        scripted_config: Optional[ScriptedDriverConfig] = None,
    ):
        self._config = config
        self._scripted = scripted_config or ScriptedDriverConfig(
            n_channels=config.n_channels,
            sample_rate_hz=int(config.sample_rate_hz),
        )
        self._sequence = list(intent_sequence)

        self._is_connected = False
        self._is_streaming = False

        self._stream_start_time: Optional[float] = None
        self._samples_emitted = 0

        self._rng = np.random.default_rng(self._scripted.rng_seed)
        # Pre-render the entire sequence at construction time. Trades memory
        # for simplicity: at 160 Hz × 64 ch × 4s/intent × float32 ≈ 160 KB per
        # intent, so a 50-step sequence is ~8 MB. Fine.
        self._rendered = self._render_sequence()

    # ─── HeadsetDriver protocol ─────────────────────────────────────────────

    @property
    def sampling_rate(self) -> int:
        return self._scripted.sample_rate_hz

    @property
    def channel_count(self) -> int:
        return self._scripted.n_channels

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def config(self) -> HeadsetConfig:
        return self._config

    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False
        self._is_streaming = False

    def start_stream(self) -> None:
        if not self._is_connected:
            raise RuntimeError("ScriptedDriver: connect() before start_stream().")
        self._stream_start_time = time.monotonic()
        self._samples_emitted = 0
        self._is_streaming = True

    def stop_stream(self) -> None:
        self._is_streaming = False

    def annotate(self, text: str) -> None:
        if not (self._is_connected and self._is_streaming):
            raise RuntimeError("ScriptedDriver: not actively streaming.")

    def read_available_samples(self) -> np.ndarray:
        if not (self._is_connected and self._is_streaming):
            raise RuntimeError("ScriptedDriver: not actively streaming.")

        # Compute how many samples we should have emitted by now.
        assert self._stream_start_time is not None
        elapsed = time.monotonic() - self._stream_start_time
        target_total = int(elapsed * self._scripted.sample_rate_hz)
        target_total = min(target_total, self._rendered.shape[1])

        new_samples = target_total - self._samples_emitted
        if new_samples <= 0:
            return np.empty((self._scripted.n_channels, 0), dtype=np.float32)

        chunk = self._rendered[
            :, self._samples_emitted : self._samples_emitted + new_samples
        ]
        self._samples_emitted = target_total
        return chunk

    # ─── Synthesis ──────────────────────────────────────────────────────────

    def _render_sequence(self) -> np.ndarray:
        """Render the entire intent sequence into a single (channels, samples) array."""
        sps = self._scripted.sample_rate_hz
        spi = int(self._scripted.seconds_per_intent * sps)
        chans = self._scripted.n_channels
        n_total = spi * len(self._sequence)

        out = np.empty((chans, n_total), dtype=np.float32)
        for i, intent in enumerate(self._sequence):
            block = self._render_intent(intent, n_samples=spi)
            out[:, i * spi : (i + 1) * spi] = block
        return out

    def _render_intent(self, intent: Intent, n_samples: int) -> np.ndarray:
        """Render one intent's worth of EEG samples."""
        chans = self._scripted.n_channels
        sps = self._scripted.sample_rate_hz
        t = np.arange(n_samples) / sps

        # Background pinkish noise on all channels.
        block = self._rng.normal(0, self._scripted.background_amplitude_uv, (chans, n_samples)).astype(np.float32)

        # Default: a baseline 10 Hz alpha rhythm on motor channels (resting alpha).
        for ch in self._scripted.left_motor_channels + self._scripted.right_motor_channels:
            if ch < chans:
                block[ch] += (self._scripted.alpha_amplitude_uv * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

        if intent == Intent.LEFT:
            # Left-hand MI desynchronizes alpha on right-hemisphere motor channels (contralateral).
            for ch in self._scripted.right_motor_channels:
                if ch < chans:
                    block[ch] -= (self._scripted.alpha_amplitude_uv * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
                    # Net effect: alpha cancellation. Add a small high-freq beta-band burst.
                    block[ch] += (5.0 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)

        elif intent == Intent.RIGHT:
            for ch in self._scripted.left_motor_channels:
                if ch < chans:
                    block[ch] -= (self._scripted.alpha_amplitude_uv * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
                    block[ch] += (5.0 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)

        elif intent == Intent.BLINK:
            # Single deliberate blink: Gaussian bump at 1.5s, ~350 ms wide, on Fp1/Fp2.
            # Width chosen to ensure the bandpass-filtered run stays above threshold
            # for ≥200 ms (the BlinkDetector's min_duration_s) — empirically a 4σ window
            # of 0.35s yields ~210-230 ms above threshold for amplitudes ≥150 µV.
            blink_center_s = 1.5
            blink_width_s = 0.35 / 4
            bump = self._scripted.blink_amplitude_uv * np.exp(
                -((t - blink_center_s) ** 2) / (2 * blink_width_s**2)
            ).astype(np.float32)
            for ch in self._scripted.prefrontal_channels:
                if ch < chans:
                    block[ch] += bump

        # Intent.REST: just background noise + baseline alpha. No-op beyond the default.

        return block
