"""Tests for ScriptedDriver synthesis quality.

We don't test classifier accuracy on synthetic data — that's not what this
driver is for. We DO test that the blink artifact synthesized for an
Intent.BLINK epoch is recognizable to the BlinkDetector, since that's the
piece that has to work end-to-end for the integration story to hold.
"""

import numpy as np
import pytest

from src.eeg_headset.blink_detector import BlinkDetector, BlinkSource
from src.eeg_headset.drivers.scripted import (
    Intent,
    ScriptedDriver,
    ScriptedDriverConfig,
)
from src.eeg_headset.headset_config import HeadsetConfig, HeadsetModel


@pytest.fixture
def config_64ch() -> HeadsetConfig:
    return HeadsetConfig(model=HeadsetModel.SAMPLE_64CH)


def test_scripted_driver_emits_correct_total_samples(config_64ch: HeadsetConfig) -> None:
    """Driver emits exactly seconds_per_intent × n_intents × sample_rate samples."""
    seq = [Intent.LEFT, Intent.RIGHT, Intent.BLINK]
    cfg = ScriptedDriverConfig(seconds_per_intent=2.0, sample_rate_hz=160, n_channels=64)
    driver = ScriptedDriver(config_64ch, seq, scripted_config=cfg)

    expected_total = int(cfg.seconds_per_intent * cfg.sample_rate_hz * len(seq))
    # Render is precomputed; we can poke directly at it.
    assert driver._rendered.shape == (cfg.n_channels, expected_total)


def test_scripted_blink_is_detected_by_blink_detector(config_64ch: HeadsetConfig) -> None:
    """A BLINK intent epoch should be recognized as a deliberate blink."""
    cfg = ScriptedDriverConfig(seconds_per_intent=4.0, sample_rate_hz=160, n_channels=64)
    driver = ScriptedDriver(config_64ch, [Intent.BLINK], scripted_config=cfg)

    blink_epoch = driver._render_intent(Intent.BLINK, n_samples=int(4.0 * 160))

    # BlinkDetector with PREFRONTAL_EEG mode looking at Fp1/Fp2 (channels 21, 22).
    detector = BlinkDetector(
        source=BlinkSource.PREFRONTAL_EEG,
        frontal_channel_idx=list(cfg.prefrontal_channels),
    )
    assert detector.detect(blink_epoch, sfreq=cfg.sample_rate_hz) is True


def test_scripted_rest_is_NOT_detected_as_blink(config_64ch: HeadsetConfig) -> None:
    """A REST intent epoch should NOT trip the blink detector."""
    cfg = ScriptedDriverConfig(seconds_per_intent=4.0, sample_rate_hz=160, n_channels=64)
    driver = ScriptedDriver(config_64ch, [Intent.REST], scripted_config=cfg)

    rest_epoch = driver._render_intent(Intent.REST, n_samples=int(4.0 * 160))

    detector = BlinkDetector(
        source=BlinkSource.PREFRONTAL_EEG,
        frontal_channel_idx=list(cfg.prefrontal_channels),
    )
    assert detector.detect(rest_epoch, sfreq=cfg.sample_rate_hz) is False


def test_left_intent_desynchronizes_right_hemisphere(config_64ch: HeadsetConfig) -> None:
    """LEFT MI: right-hemisphere motor channels (contralateral) should have lower
    alpha-band power than left-hemisphere motor channels."""
    cfg = ScriptedDriverConfig(seconds_per_intent=4.0, sample_rate_hz=160, n_channels=64)
    driver = ScriptedDriver(config_64ch, [Intent.LEFT], scripted_config=cfg)
    epoch = driver._render_intent(Intent.LEFT, n_samples=int(4.0 * 160))

    # Crude alpha power estimate: variance of the signal in time domain.
    # For LEFT MI, right-hemisphere channels should have LESS variance (alpha
    # cancelled by the contralateral desync term), left-hemisphere channels
    # should retain their baseline alpha.
    right_motor_var = np.mean([np.var(epoch[ch]) for ch in cfg.right_motor_channels])
    left_motor_var = np.mean([np.var(epoch[ch]) for ch in cfg.left_motor_channels])

    assert right_motor_var < left_motor_var, (
        "LEFT MI should suppress alpha on right (contralateral) motor channels"
    )


def test_streaming_emits_samples_progressively(config_64ch: HeadsetConfig) -> None:
    """read_available_samples should drain the precomputed render gradually as time passes."""
    import time

    cfg = ScriptedDriverConfig(seconds_per_intent=0.5, sample_rate_hz=160, n_channels=64)
    driver = ScriptedDriver(config_64ch, [Intent.LEFT, Intent.RIGHT], scripted_config=cfg)
    driver.connect()
    driver.start_stream()

    # First read right after start — should have at most a handful of samples.
    first = driver.read_available_samples()
    assert first.shape[1] <= 5, "Should not have many samples immediately after start"

    # Sleep enough to consume an entire intent's worth (0.5s).
    time.sleep(0.6)
    second = driver.read_available_samples()
    # Should have ~0.6s worth = ~96 samples.
    assert second.shape[1] >= 50

    driver.stop_stream()
    driver.disconnect()
