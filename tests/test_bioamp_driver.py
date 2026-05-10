"""Tests for BioAmpEXGDriver using a stubbed serial.Serial.

We don't talk to real hardware — we patch `serial.Serial` to feed canned bytes
back to the driver and verify it parses ASCII-int lines correctly, handles
garbage lines gracefully, and respects the streaming lifecycle.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the entire module if pyserial isn't installed.
serial = pytest.importorskip("serial")

from src.eeg_headset.drivers.bioamp import BioAmpConfig, BioAmpEXGDriver


class _FakeSerial:
    """Minimal stand-in for serial.Serial that returns lines from a queue."""

    def __init__(self, lines: list[bytes]) -> None:
        self._queue = list(lines)
        self.is_open = True

    def readline(self) -> bytes:
        if self._queue:
            return self._queue.pop(0)
        # Real Serial returns empty bytes on timeout — emulate that to give the
        # reader thread a chance to notice stop_event.
        time.sleep(0.01)
        return b""

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


def _make_driver_with_lines(lines: list[bytes]) -> BioAmpEXGDriver:
    """Build a driver whose serial port returns the given lines."""
    cfg = BioAmpConfig(port="/dev/null-test", baudrate=115200)
    driver = BioAmpEXGDriver(cfg)
    fake = _FakeSerial(lines)

    # Bypass the real serial.Serial constructor.
    with patch.object(serial, "Serial", return_value=fake):
        driver.connect()
    return driver


def test_parses_valid_lines() -> None:
    driver = _make_driver_with_lines([b"100\n", b"200\n", b"300\n"])
    driver.start_stream()
    time.sleep(0.1)  # give reader thread time to drain the queue
    samples = driver.read_available_samples()
    driver.stop_stream()
    driver.disconnect()

    assert samples.shape == (1, 3)
    assert list(samples[0]) == [100.0, 200.0, 300.0]


def test_ignores_garbage_lines() -> None:
    driver = _make_driver_with_lines(
        [b"BIOAMP READY\r\n", b"42\n", b"junk!\n", b"99\n"]
    )
    driver.start_stream()
    time.sleep(0.1)
    samples = driver.read_available_samples()
    driver.stop_stream()
    driver.disconnect()

    assert samples.shape == (1, 2)
    assert list(samples[0]) == [42.0, 99.0]


def test_adc_to_uv_scale_applied() -> None:
    cfg = BioAmpConfig(port="/dev/null-test", adc_to_uv_scale=2.5)
    driver = BioAmpEXGDriver(cfg)
    fake = _FakeSerial([b"10\n", b"20\n"])
    with patch.object(serial, "Serial", return_value=fake):
        driver.connect()
    driver.start_stream()
    time.sleep(0.1)
    samples = driver.read_available_samples()
    driver.stop_stream()
    driver.disconnect()

    assert list(samples[0]) == [25.0, 50.0]


def test_lifecycle_idempotent() -> None:
    driver = _make_driver_with_lines([])
    driver.connect()  # second connect is a no-op
    assert driver.is_connected
    driver.disconnect()
    assert not driver.is_connected
    driver.disconnect()  # idempotent


def test_read_when_not_streaming_raises() -> None:
    driver = _make_driver_with_lines([])
    with pytest.raises(RuntimeError):
        driver.read_available_samples()


def test_returns_empty_when_no_data() -> None:
    driver = _make_driver_with_lines([])
    driver.start_stream()
    time.sleep(0.05)
    samples = driver.read_available_samples()
    driver.stop_stream()
    driver.disconnect()

    assert samples.shape == (1, 0)
