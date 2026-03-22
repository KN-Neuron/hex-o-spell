import numpy as np
from brainaccess.utils import acquisition

from src.egg_headset.model import HeadsetConfiguration
from brainaccess.core.eeg_manager import EEGManager


class BrainAccessDriver:
    """
    BrainAccess EEG driver for interfacing with the BrainAccess SDK.\n
    Quirks:
    - EEG mode "accumulate" retains all data forever, and
    "roll" only keeps n latest seconds,
    so we have to manually clear the buffer after reading samples
    to achieve dynamic rolling window behavior.
    (the "roll" functionality is achieved later in the RingBuffer, which
    1. is more efficient and 2. keeps the protocol contract)
    """

    def __init__(self, config: HeadsetConfiguration):
        self._config = config
        self._eeg = acquisition.EEG(mode="accumulate")
        self._mgr = EEGManager()

        self._is_connected = False
        self._is_streaming = False

    @property
    def sampling_rate(self) -> int:
        return self._config.sample_rate_hz

    @property
    def channel_count(self) -> int:
        return self._config.n_channels

    def connect(self) -> None:
        if self._is_connected:
            return

        self._eeg.setup(
            self._mgr,
            device_name=self._config.device_name,
            cap=self._config.channel_map,
            sfreq=self._config.sample_rate_hz,
        )
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        self.stop_stream()
        self._mgr.disconnect()

    def start_stream(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot start stream: Headset not connected.")

        if self._is_streaming:
            return

        self._eeg.start_acquisition()
        self._is_streaming = True
        return

    def stop_stream(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot stop stream: Headset not connected.")

        if not self._is_streaming:
            return

        self._eeg.stop_acquisition()
        self._is_streaming = False
        return

    def annotate(self, text: str) -> None:
        if not self._is_connected:
            raise RuntimeError("Cannot annotate: Headset not connected.")
        if not self._is_streaming:
            raise RuntimeError("Cannot annotate: Headset not streaming.")

        self._eeg.annotate(text)
        return

    def read_available_samples(self) -> np.ndarray:
        if not self._is_connected:
            raise RuntimeError("Cannot annotate: Headset not connected.")
        if not self._is_streaming:
            raise RuntimeError("Cannot annotate: Headset not streaming.")

        # safely acquire chunks and clear the buffer
        with self._eeg.data.lock:
            chunks = self._eeg.data.data[:]
            self._eeg.data.data.clear()

        if not chunks:
            return np.empty((self.channel_count, 0))

        raw_hw_data = np.concatenate(chunks, axis=1)

        eeg_row_indices = [
            row_idx
            for hw_id, row_idx in self._eeg.channels_indexes.items()
            if self._eeg.channels_type[hw_id] == "EEG"
        ]

        eeg_only_data = raw_hw_data[eeg_row_indices, :]

        return eeg_only_data
