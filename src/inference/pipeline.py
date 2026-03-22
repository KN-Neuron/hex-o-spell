import time

import numpy as np
import torch
from src.egg_headset import EggHeadset
from src.egg_headset.drivers.playback import PlaybackDriver
from src.egg_headset.model import HeadsetConfiguration, HeadsetModel
from src.inference.starter_bci import LABELS, load_model, preprocess
from src.speller import Speller, Direction


# A live example of the model interacting with the headset and speller.
# This script simulates a real-time BCI pipeline. It continuously polls the headset
# for new data, preprocesses it, and feeds it into the model to get predictions.
# Based on the predicted labels, it updates the speller's state accordingly.
# The pipeline runs until the user interrupts it (e.g., by pressing Ctrl+C).
# Note: The model is only 2-class, so we handle left and right hand predictions.
# The speller state machine also supports select and back actions, but we haven't
# mapped any predictions to those yet.
# Example of the speller working correctly is executable in speller/speller.py


# How confident does the model need to be before we consider a prediction valid?
CONFIDENCE_CUTOFF = 0.55


def main() -> None:
    print("Loading EEGNet model...")
    model = load_model()

    print("Initializing Fake EEG Driver...")
    config = HeadsetConfiguration(model=HeadsetModel.SAMPLE_64CH)
    driver = PlaybackDriver(
        config=config, source="data/raw/example_64ch_250samples.npy", loop=True
    )
    headset = EggHeadset(driver, buffer_size_seconds=60)
    speller: Speller = Speller()

    print("Connecting to headset...")
    headset.connect()
    headset.start()

    print("BCI Pipeline started. Press Ctrl+C to stop.")

    try:
        while True:
            print("\n--- Starting new epoch ---")

            # 1. Mark the start of an epoch
            headset.annotate("predict")

            for _ in range(40):
                headset.poll()
                time.sleep(0.1)

            # 3. Retrieve the slice
            epoch_data = headset.get_output(seconds=4)

            expected_samples = 641
            if epoch_data.shape[1] < expected_samples:
                pad_width = expected_samples - epoch_data.shape[1]
                epoch_data = np.pad(epoch_data, ((0, 0), (0, pad_width)), mode="edge")
            elif epoch_data.shape[1] > expected_samples:
                epoch_data = epoch_data[:, :expected_samples]

            if epoch_data.shape[0] != 64:
                print(
                    (
                        f"Warning: Expected 64 channels, got {epoch_data.shape[0]}. "
                        "Ensure driver fits the model."
                    )
                )

            try:
                tensor = preprocess(epoch_data)

                with torch.no_grad():
                    probs = model(tensor).softmax(dim=1)[0]
                    pred = probs.argmax().item()

                print(f"Predicted: {LABELS[pred]} (confidence: {probs[pred]:.0%})")

                if probs[pred] >= CONFIDENCE_CUTOFF:
                    label = LABELS[pred]

                    if label == "left_hand":
                        speller.move(Direction.LEFT)
                    elif label == "right_hand":
                        speller.move(Direction.RIGHT)

            except Exception as e:
                print(f"Prediction failed: {e}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping BCI workflow...")
    finally:
        headset.stop()
        headset.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
