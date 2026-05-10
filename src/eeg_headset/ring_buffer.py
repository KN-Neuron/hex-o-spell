import numpy as np


class RingBuffer:
    def __init__(self, n_channels: int, capacity_seconds: int, sample_rate_hz: int):
        self.n_channels = n_channels
        self.sample_rate = sample_rate_hz
        self.capacity = capacity_seconds * sample_rate_hz
        self._buffer = np.zeros((self.n_channels, self.capacity), dtype=float)

        # Total number of samples that have ever been written to the buffer
        self.total_samples = 0

        # Index in the circular buffer where the next incoming sample will be written
        self._write_idx = 0

    def append(self, new_data: np.ndarray) -> None:
        """Writes new chunks into the circular buffer."""
        if new_data.size == 0:
            return

        if new_data.ndim != 2 or new_data.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected new_data shape (n_channels, n_samples), got {new_data.shape}"
            )

        n_new = new_data.shape[1]

        # Edge case: chunk is bigger than our entire buffer capacity
        if n_new >= self.capacity:
            self._buffer[:, :] = new_data[:, -self.capacity :]
            self._write_idx = 0
            self.total_samples += n_new
            return

        end_idx = self._write_idx + n_new
        if end_idx <= self.capacity:
            # Fits before wrapping around
            self._buffer[:, self._write_idx : end_idx] = new_data
        else:
            overflow = end_idx - self.capacity
            first_part = n_new - overflow

            self._buffer[:, self._write_idx :] = new_data[:, :first_part]
            self._buffer[:, :overflow] = new_data[:, first_part:]

        self._write_idx = (self._write_idx + n_new) % self.capacity
        self.total_samples += n_new

    def get_slice(
        self, start_idx: int, end_idx: int, pad_zeros: bool = True
    ) -> np.ndarray:
        """
        Gets a window of data based on absolute sample indices.
        Args:
            start_idx: Absolute index of the first sample (0-based).
            end_idx: Absolute index of the last sample (exclusive).
            pad_zeros:
            If True, pads with zeros for any requested samples
            that are not available.
            If False, returns only the available portion without padding.
        Returns:
            np.ndarray: Shape (n_channels, requested_length) if pad_zeros=True,
            otherwise shape (n_channels, available_length).
        """
        requested_length = end_idx - start_idx
        out = np.zeros((self.n_channels, requested_length), dtype=float)

        # Check if the requested data is completely in the future
        if start_idx >= self.total_samples:
            return out if pad_zeros else np.empty((self.n_channels, 0))

        # Check if the requested data was evicted
        oldest_available = max(0, self.total_samples - self.capacity)
        actual_start = max(start_idx, oldest_available)

        # Check if the requested data hasn't fully arrived yet
        actual_end = min(end_idx, self.total_samples)

        available_length = actual_end - actual_start
        if available_length <= 0:
            return out if pad_zeros else np.empty((self.n_channels, 0))

        # Calculate where actual_start is located in our ring buffer
        read_offset = self.total_samples - actual_start
        read_start_idx = (self._write_idx - read_offset) % self.capacity

        # Calculate where to put it in the output array
        out_start = actual_start - start_idx
        out_end = out_start + available_length

        # Read from the ring buffer
        read_end = read_start_idx + available_length
        if read_end <= self.capacity:
            # Continuous read
            out[:, out_start:out_end] = self._buffer[:, read_start_idx:read_end]
        else:
            # Split read across the wrap-around boundary
            first_part = self.capacity - read_start_idx
            out[:, out_start : out_start + first_part] = self._buffer[
                :, read_start_idx:
            ]
            out[:, out_start + first_part : out_end] = self._buffer[
                :, : available_length - first_part
            ]

        if not pad_zeros:
            return out[:, out_start:out_end]

        return out
