import threading
import time

from numpy import ndarray
from egg_headset.drivers.playback import PlaybackDriver
from egg_headset.model import HeadsetConfiguration, HeadsetModel
from . import EggHeadset
import numpy as np


driver_config = HeadsetConfiguration(
    model=HeadsetModel.MIDI_16CH_BASE, config_path="brainaccess_headsets_config.yaml"
)
# driver = BrainAccessDriver(driver_config)
# driver = MockDriver(driver_config)
driver = PlaybackDriver(
    driver_config, source="data/example_16ch_250samples.npy", loop=True
)
eeg = EggHeadset(driver)

eeg.connect()
eeg.start()
eeg.annotate("start")

samples = ndarray(shape=(driver_config.n_channels, 0), dtype=float)


def poll_continuously() -> None:
    while True:
        eeg.poll()
        time.sleep(0.05)  # 20 Hz polling rate


threading.Thread(target=poll_continuously, daemon=True).start()

while True:
    x = input("Wprowadź adnotacje (exit aby zakończyć): ")

    if x == "exit":
        break

    output = eeg.get_output(seconds=1)
    samples = np.concatenate((samples, output), axis=1)

    eeg.annotate(x)

eeg.stop()


print(samples)
