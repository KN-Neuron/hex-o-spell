"""Unit tests for BlinkDetector.

Tests cover:
- Deliberate blink (200-400 ms) → detected.
- Spontaneous blink (~150 ms) → rejected.
- Pure noise → no false positives.
- Saturation / huge DC drift → rejected (duration > max_duration_s).
- Very short epochs → graceful degradation (returns False, no crash).
"""

import numpy as np
import pytest

from src.eeg_headset.blink_detector import (
    BlinkDetector,
    BlinkDetectorConfig,
    BlinkSource,
    synthesize_blink_signal,
)


SFREQ = 250.0  # BrainAccess MIDI sample rate.


@pytest.fixture
def detector() -> BlinkDetector:
    return BlinkDetector(
        source=BlinkSource.DEDICATED_EOG,
        frontal_channel_idx=[0],
    )


def test_detects_deliberate_blink(detector: BlinkDetector) -> None:
    """A 300 ms blink at amplitude 150 µV should fire detection."""
    signal = synthesize_blink_signal(
        sfreq=SFREQ,
        duration_s=4.0,
        blink_at_s=1.0,
        blink_duration_s=0.3,
        blink_amplitude_uv=150.0,
    )
    assert detector.detect(signal, sfreq=SFREQ) is True


def test_rejects_spontaneous_blink(detector: BlinkDetector) -> None:
    """A 100 ms blink (spontaneous) should NOT fire detection."""
    signal = synthesize_blink_signal(
        sfreq=SFREQ,
        duration_s=4.0,
        blink_at_s=1.0,
        blink_duration_s=0.1,
        blink_amplitude_uv=150.0,
    )
    assert detector.detect(signal, sfreq=SFREQ) is False


def test_rejects_pure_noise(detector: BlinkDetector) -> None:
    """White noise at 5 µV should not produce detections."""
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 5.0, int(4.0 * SFREQ))[np.newaxis, :]
    assert detector.detect(signal, sfreq=SFREQ) is False


def test_rejects_dc_drift(detector: BlinkDetector) -> None:
    """A long DC excursion (e.g. electrode movement) should be rejected as too long."""
    n = int(4.0 * SFREQ)
    t = np.arange(n) / SFREQ
    # 2-second-long ramp: well above max_duration_s of 1.0s.
    signal = (200 * np.where((t > 0.5) & (t < 2.5), 1.0, 0.0)).reshape(1, -1)
    assert detector.detect(signal, sfreq=SFREQ) is False


def test_short_window_returns_false(detector: BlinkDetector) -> None:
    """An epoch shorter than min_samples_for_filter should not crash."""
    short = np.random.randn(1, 50)
    assert detector.detect(short, sfreq=SFREQ) is False


def test_empty_window_returns_false(detector: BlinkDetector) -> None:
    """Zero-sized epoch must not crash."""
    empty = np.empty((1, 0))
    assert detector.detect(empty, sfreq=SFREQ) is False


def test_uses_only_specified_channels() -> None:
    """When frontal_channel_idx selects ch 0 only, noise on ch 1-2 must not trigger."""
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 5.0, (3, int(4.0 * SFREQ)))
    # Plant a deliberate blink on channel 1, but configure detector to read ch 0.
    blink = synthesize_blink_signal(
        sfreq=SFREQ, duration_s=4.0, blink_at_s=1.0, blink_duration_s=0.3,
        blink_amplitude_uv=200.0,
    )
    signal[1, :] = blink[0, :]

    detector_ch0 = BlinkDetector(
        source=BlinkSource.DEDICATED_EOG, frontal_channel_idx=[0]
    )
    assert detector_ch0.detect(signal, sfreq=SFREQ) is False

    # Sanity check: pointing it at ch 1 should detect.
    detector_ch1 = BlinkDetector(
        source=BlinkSource.DEDICATED_EOG, frontal_channel_idx=[1]
    )
    assert detector_ch1.detect(signal, sfreq=SFREQ) is True


def test_higher_threshold_rejects_smaller_blinks() -> None:
    """A blink at 75 µV should be detected with default k, but rejected with k=15."""
    signal = synthesize_blink_signal(
        sfreq=SFREQ, duration_s=4.0, blink_at_s=1.0, blink_duration_s=0.3,
        blink_amplitude_uv=75.0, noise_amplitude_uv=2.0,
    )
    permissive = BlinkDetector(
        source=BlinkSource.DEDICATED_EOG,
        frontal_channel_idx=[0],
        # Lower threshold_min_uv so the test exercises threshold_k rather than the floor.
        config=BlinkDetectorConfig(threshold_k=3.0, threshold_min_uv=10.0),
    )
    strict = BlinkDetector(
        source=BlinkSource.DEDICATED_EOG,
        frontal_channel_idx=[0],
        config=BlinkDetectorConfig(threshold_k=15.0, threshold_min_uv=10.0),
    )

    assert permissive.detect(signal, sfreq=SFREQ) is True
    assert strict.detect(signal, sfreq=SFREQ) is False
