from typing import Callable, List
import numpy as np
from typing import Optional
from src.egg_headset.drivers import HeadsetDriver
from src.egg_headset.ring_buffer import RingBuffer

EegSubscriberCallback = Callable[[np.ndarray], None]


class EggHeadset:
    """
    Interfejs do opaski EEG.
    Umożliwia połączenie z opaską, rozpoczęcie i zatrzymanie strumienia danych,
    dodawanie annotacji oraz pobieranie próbek EEG od ostatniej annotacji.
    """

    def __init__(self, driver: HeadsetDriver, buffer_size_seconds: int = 60) -> None:
        self._driver = driver
        self._buffer = RingBuffer(
            self._driver.channel_count,
            buffer_size_seconds * self._driver.sampling_rate,
            self._driver.sampling_rate,
        )
        self._subscribers: List[EegSubscriberCallback] = []
        self._last_annotation_index: Optional[int] = None

    def connect(self) -> None:
        self._driver.connect()

    def disconnect(self) -> None:
        self._driver.disconnect()

    def start(self) -> None:
        self._driver.start_stream()

    def stop(self) -> None:
        self._driver.stop_stream()

    def add_subscriber(self, callback: EegSubscriberCallback) -> None:
        """Subskrybuje callback, który będzie wywoływany przy każdym wywołaniu poll()"""
        self._subscribers.append(callback)

    def annotate(self, label: str) -> None:
        self._last_annotation_index = self._buffer.total_samples
        self._driver.annotate(label)

    def poll(self) -> None:
        new_data = self._driver.read_available_samples()
        if new_data.size > 0:
            self._buffer.append(new_data)

            for callback in self._subscribers:
                callback(new_data)

    def get_output(self, seconds: int = 1) -> np.ndarray:
        """Zwraca `seconds` sekund próbek EEG o kształcie
        `(channel_count, sample_rate*seconds)` od startu ostatniej annotacji.
        Jeśli brakuje próbek, dopełnia zerami.
        """

        if self._last_annotation_index is None:
            raise RuntimeError(
                "Brak annotacji — wywołaj annotate() przed get_output()."
            )

        start_idx = self._last_annotation_index
        end_idx = start_idx + self._driver.sampling_rate * seconds

        # Logic ported over from the original implementation.
        # Suboptimal, since reading samples includes locking, moving, clearing buffers.
        # Delete this in the future
        max_iters = 20
        while self._buffer.total_samples < end_idx and max_iters > 0:
            self.poll()
            max_iters -= 1

        return self._buffer.get_slice(start_idx, end_idx, pad_zeros=True)
