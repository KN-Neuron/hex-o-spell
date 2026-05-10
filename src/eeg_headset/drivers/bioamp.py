"""Driver for BioAmp EXG Pill connected via a microcontroller (ESP32/Arduino/Maker Uno).

Hardware data flow
------------------
    ┌──────────┐   electrodes      ┌──────────────┐   analog 0-3.3V
    │  Eye/    │──────────────────▶│ BioAmp EXG   │────────────────┐
    │  Skin    │                    │ Pill (AFE)   │                │
    └──────────┘                    └──────────────┘                ▼
                                                            ┌────────────────┐
                                                            │ MCU ADC (12-bit)│
                                                            │  ESP32/Arduino │
                                                            └────────┬───────┘
                                                                     │ USB serial
                                                                     │ ASCII int/line
                                                                     ▼
                                                            ┌────────────────┐
                                                            │  BioAmpEXGDriver│
                                                            └────────────────┘

Wire protocol
-------------
The MCU firmware MUST emit one ADC sample per line as a decimal integer in
microvolts (or in raw ADC counts; see `adc_to_uv_scale`). Newline-terminated.
Example reference firmware: Upside Down Labs `BioAmp_EXG_Pill` example sketch
modified to print samples instead of plotting, e.g.::

    void loop() {
        int raw = analogRead(INPUT_PIN);
        // simple bandpass via biquad would happen here
        Serial.println(raw);
        delayMicroseconds(2000);  // 500 Hz
    }

Why ASCII over binary
---------------------
- Trivial to debug with `cat /dev/ttyUSB0`.
- Resilient to dropped bytes (newline resyncs the next sample).
- 500 Hz × ~6 chars/line ≈ 3000 B/s, well below 115200 baud capacity (~14400 B/s).
At higher rates (>2 kHz) you'd switch to a packed binary protocol.

Why a separate thread for serial reads
--------------------------------------
The pipeline poll() is invoked from the main thread at ~10 Hz. Serial reads
must not block the pipeline if the MCU stops sending; we keep a background
reader thread that fills an internal queue, and `read_available_samples`
just drains the queue non-blockingly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# pyserial is optional — only required when this driver is actually used.
try:
    import serial  # type: ignore
except ImportError:
    serial = None  # type: ignore


@dataclass
class BioAmpConfig:
    """Configuration for the BioAmp EXG Pill driver.

    The defaults match a stock BioAmp EXG Pill + ESP32 sketch sampling at 500 Hz
    with 12-bit ADC range 0-4095 and a typical gain that makes 1 ADC count ≈ 1 µV
    when the pill is configured for EOG. Tune `adc_to_uv_scale` empirically by
    holding the electrodes still and confirming RMS noise is roughly the
    expected 5-15 µV.
    """

    port: str
    baudrate: int = 115200
    sample_rate_hz: int = 500
    adc_to_uv_scale: float = 1.0  # multiply incoming int by this to get µV
    n_channels: int = 1  # BioAmp EXG Pill is single-channel
    read_timeout_s: float = 1.0


class BioAmpEXGDriver:
    """Implements the HeadsetDriver Protocol for a serial-attached BioAmp EXG Pill.

    Conforms to the same interface as MockDriver/PlaybackDriver/BrainAccessDriver,
    so it can be wired into EEGHeadset and the rest of the pipeline transparently.
    """

    def __init__(self, config: BioAmpConfig) -> None:
        if serial is None:
            raise ImportError(
                "pyserial is required for BioAmpEXGDriver. "
                "Install with: pip install pyserial"
            )
        self._cfg = config
        self._serial: Optional["serial.Serial"] = None

        self._is_connected = False
        self._is_streaming = False

        # Background reader fills this buffer; read_available_samples drains it.
        self._buffer_lock = threading.Lock()
        self._sample_buffer: list[float] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # A minimal HeadsetConfig-like shim so the rest of the system (which uses
        # `driver.config.channel_map` etc.) doesn't crash. The BioAmp doesn't have
        # a 10-20 channel layout, so the map is just {0: "EOG"}.
        self._config_shim = _BioAmpConfigShim(
            n_channels=config.n_channels,
            sample_rate_hz=config.sample_rate_hz,
        )

    # ─── HeadsetDriver protocol ─────────────────────────────────────────────

    @property
    def sampling_rate(self) -> int:
        return self._cfg.sample_rate_hz

    @property
    def channel_count(self) -> int:
        return self._cfg.n_channels

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def config(self):
        return self._config_shim

    def connect(self) -> None:
        if self._is_connected:
            return
        self._serial = serial.Serial(
            self._cfg.port,
            baudrate=self._cfg.baudrate,
            timeout=self._cfg.read_timeout_s,
        )
        # Flush junk left by the MCU bootloader.
        self._serial.reset_input_buffer()
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self.stop_stream()
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        self._is_connected = False

    def start_stream(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot start stream: BioAmp not connected.")
        if self._is_streaming:
            return

        self._stop_event.clear()
        with self._buffer_lock:
            self._sample_buffer.clear()

        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="BioAmpReader"
        )
        self._reader_thread.start()
        self._is_streaming = True

    def stop_stream(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot stop stream: BioAmp not connected.")
        if not self._is_streaming:
            return

        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        self._is_streaming = False

    def annotate(self, text: str) -> None:
        # BioAmp doesn't have hardware annotation; the upstream EEGHeadset
        # tracks annotation indices in its own buffer instead.
        if not self._is_connected or not self._is_streaming:
            raise RuntimeError("Cannot annotate: BioAmp not actively streaming.")

    def read_available_samples(self) -> np.ndarray:
        if not self._is_connected or not self._is_streaming:
            raise RuntimeError("Cannot read samples: BioAmp not actively streaming.")

        with self._buffer_lock:
            samples = self._sample_buffer
            self._sample_buffer = []

        if not samples:
            return np.empty((self.channel_count, 0), dtype=float)

        # BioAmp EXG Pill is single-channel: shape (1, n_samples).
        return np.asarray(samples, dtype=float).reshape(1, -1)

    # ─── Internal: serial reader thread ─────────────────────────────────────

    def _read_loop(self) -> None:
        """Background thread reading lines from serial and pushing to the buffer."""
        assert self._serial is not None
        scale = self._cfg.adc_to_uv_scale

        while not self._stop_event.is_set():
            try:
                # readline() blocks up to read_timeout_s; if MCU stops sending,
                # we just loop and re-check stop_event.
                raw = self._serial.readline()
            except Exception as e:
                # Hardware disconnected mid-stream, or USB error.
                # Propagating up would crash the thread silently — log and stop.
                print(f"[BioAmpEXGDriver] serial read error: {e}")
                break

            if not raw:
                continue

            try:
                # Whitespace-tolerant; accept e.g. "  1234\r\n".
                value = int(raw.decode("ascii", errors="ignore").strip())
            except ValueError:
                # Garbage line (e.g. firmware bootup banner). Ignore.
                continue

            uv = value * scale
            with self._buffer_lock:
                self._sample_buffer.append(uv)


class _BioAmpConfigShim:
    """Minimal `config` object so `EEGHeadset.channel_labels` works for BioAmp.

    The full HeadsetConfig pulls fields from headsets.yaml, but BioAmp doesn't
    have a 10-20 layout. We expose the few fields the rest of the codebase
    actually reads.
    """

    def __init__(self, n_channels: int, sample_rate_hz: int) -> None:
        self.n_channels = n_channels
        self.sample_rate_hz = sample_rate_hz
        self.device_name = "BioAmp EXG Pill"
        self.channel_map = {i: ("EOG" if i == 0 else f"AUX{i}") for i in range(n_channels)}
