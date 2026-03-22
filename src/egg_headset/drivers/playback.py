import time

import numpy as np

from src.egg_headset.model import HeadsetConfiguration


class PlaybackDriver:
    """
    A driver that simulates a headset by reading from a pre-recorded data file.
    The data file should be a NumPy array (.npy) with shape (n_channels, n_samples)
    and should match the configuration specified in the HeadsetConfiguration.
    """

    def __init__(self, config: HeadsetConfiguration, source: str, loop: bool = False):
        self._data: np.ndarray = np.load(source)

        if self._data.ndim != 2:
            raise ValueError(
                f"Data array must be 2D (channels, samples). Got {self._data.ndim}D."
            )

        if self._data.shape[0] != config.n_channels:
            raise ValueError(
                f"""Data channel count of {self._data.shape[0]} does
not match config channel count of {config.n_channels}."""
            )

        self._config = config
        self._loop = loop

        self._is_connected = False
        self._is_streaming = False
        self._stream_start_time: None | int = None
        self._samples_read_so_far: int = 0
        self._last_annotation_time: None | int = None

    @property
    def sampling_rate(self) -> int:
        return self._config.sample_rate_hz

    @property
    def channel_count(self) -> int:
        return self._config.n_channels

    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_streaming = False
        self._is_connected = False

    def start_stream(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot start stream: Headset not connected.")

        if self._is_streaming:
            return

        self._stream_start_time = time.time_ns()
        self._samples_read_so_far = 0
        self._is_streaming = True
        return

    def stop_stream(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot stop stream: Headset not connected.")

        if not self._is_streaming:
            return

        self._stream_start_time = None
        self._samples_read_so_far = 0
        self._is_streaming = False
        return

    def annotate(self, text: str) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot annotate: Headset not connected.")
        if not self._is_streaming:
            raise RuntimeError("Cannot annotate: Headset not streaming.")

        self._last_annotation_time = time.time_ns()
        return

    def read_available_samples(self) -> np.ndarray:
        if not self._is_connected:
            raise RuntimeError("Cannot read samples: Headset not connected.")
        if not self._is_streaming:
            raise RuntimeError("Cannot read samples: Headset not streaming.")

        current_time = time.time_ns()

        assert self._stream_start_time is not None
        elapsed_sec_since_start = (current_time - self._stream_start_time) / 1e9
        total_samples_expected = int(elapsed_sec_since_start * self.sampling_rate)

        samples_to_read = total_samples_expected - self._samples_read_so_far

        if samples_to_read <= 0:
            return np.empty((self.channel_count, 0))

        total_samples_available = self._data.shape[1]

        if self._loop:
            indices = (
                np.arange(
                    self._samples_read_so_far,
                    self._samples_read_so_far + samples_to_read,
                )
                % total_samples_available
            )
            eeg_only_data = self._data[:, indices]
            self._samples_read_so_far += samples_to_read
        else:
            start_offset = min(self._samples_read_so_far, total_samples_available)
            end_offset = min(start_offset + samples_to_read, total_samples_available)
            eeg_only_data = self._data[:, start_offset:end_offset]
            self._samples_read_so_far += samples_to_read

        return eeg_only_data
