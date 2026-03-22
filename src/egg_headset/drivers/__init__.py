from typing import Protocol
import numpy as np

from src.egg_headset.drivers.brainaccess import BrainAccessDriver
from src.egg_headset.drivers.mock import MockDriver
from src.egg_headset.drivers.playback import PlaybackDriver


class HeadsetDriver(Protocol):
    """Interface for EEG headset drivers."""

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of the headset in Hz."""
        ...

    @property
    def channel_count(self) -> int:
        """Number of EEG channels."""
        ...

    def connect(self) -> None:
        """
        Connect to the headset.
        Idempotent: safe to call multiple times.
        """
        ...

    def disconnect(self) -> None:
        """
        Disconnect from the headset.
        Idempotent: safe to call multiple times.
        """
        ...

    def start_stream(self) -> None:
        """
        Start streaming data from the headset.
        Requires connection. Throws if not connected.
        Idempotent: safe to call multiple times.
        """
        ...

    def stop_stream(self) -> None:
        """
        Stop streaming data from the headset.
        Requires connection. Throws if not connected.
        Idempotent: safe to call multiple times.
        """
        ...

    def annotate(self, text: str) -> None:
        """
        Add an annotation/marker to the data stream.
        Requires connection and active streaming.
        Throws if not connected or not streaming.

        Args:
            text: Annotation text to record.
        """
        ...

    def read_available_samples(self) -> np.ndarray:
        """
        Read all samples captured since the last call.
        Requires connection and active streaming.
        Throws if not connected or not streaming.

        Returns:
            np.ndarray: Shape (n_channels, n_samples).
            Returns empty array if no data available.
        """
        ...


__all__ = ["BrainAccessDriver", "MockDriver", "HeadsetDriver", "PlaybackDriver"]
