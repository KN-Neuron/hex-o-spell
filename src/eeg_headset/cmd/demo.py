import threading
import time

import numpy as np
from ..drivers import MockDriver
from ..headset_config import HeadsetConfig, HeadsetModel
from ..eeg_headset import EEGHeadset


def main() -> None:
    driver_config = HeadsetConfig(
        model=HeadsetModel.MIDI_16CH_BASE, 
        config_path="brainaccess_headsets_config.yaml"
    )
    # driver = BrainAccessDriver(driver_config)
    driver = MockDriver(driver_config)
    eeg = EEGHeadset(driver)

    eeg.connect()
    eeg.start()
    eeg.annotate("start")

    samples = np.ndarray(shape=(driver_config.n_channels, 0), dtype=float)

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

if __name__ == "__main__":
    main()
