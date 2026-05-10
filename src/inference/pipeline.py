import time
from collections import deque, Counter
from src.speller.state import SpellerStateSectorNavigation

import numpy as np
import torch

from src.eeg_headset.eeg_headset import EEGHeadset
from src.eeg_headset.drivers.playback import PlaybackDriver
from src.eeg_headset.headset_config import HeadsetConfig, HeadsetModel
from src.inference.starter_bci import LABELS, load_model, preprocess
from src.speller import Speller, Direction

CONFIDENCE_CUTOFF = 0.55
SMOOTHING_WINDOW_SIZE = 5  # Liczba ostatnich predykcji branych pod uwagę
MIN_VOTES_REQUIRED = 3  # Wymagana liczba spójnych głosów do wykonania akcji

def detect_blink(window: np.ndarray, threshold: float = 100.0, frontal_channels: list[int] = None) -> bool:
    """
    Sprawdza, czy w oknie sygnału wystąpiło mrugnięcie okiem.
    
    window: macierz (channels, samples)
    threshold: próg napięcia (w mikrowoltach), powyżej którego uznajemy sygnał za mrugnięcie.
    frontal_channels: indeksy elektrod czołowych (np. Fp1, Fp2). 
                      Jeśli None, sprawdzamy wszystkie kanały.
    """
    if frontal_channels is not None:
        data_to_check = window[frontal_channels, :]
    else:
        data_to_check = window
        
    max_amplitude = np.max(np.abs(data_to_check))
    
    return max_amplitude > threshold

def main() -> None:
    print("Loading EEGNet model...")
    model = load_model()

    print("Initializing Fake EEG Driver...")
    # config = HeadsetConfig(model=HeadsetModel.MIDI_16CH_BASE)
    config = HeadsetConfig(model=HeadsetModel.SAMPLE_64CH)
    driver = PlaybackDriver(
        config=config, source="data/X.npy", loop=True
    )
    headset = EEGHeadset(driver, buffer_size_seconds=60)
    speller: Speller = Speller()
    speller.state = SpellerStateSectorNavigation()

    prediction_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)

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
            print(f"Suma kontrolna danych: {np.sum(epoch_data):.5f}")
            
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

            if detect_blink(epoch_data, threshold=150.0):
                print("👁️ WYKRYTO MRUGNIĘCIE! (Wybór zatwierdzony)")
                
                continue

            try:
                tensor = preprocess(epoch_data)

                with torch.no_grad():
                    probs = model(tensor).softmax(dim=1)[0]
                    pred_idx = probs.argmax().item()
                    confidence = probs[pred_idx].item()

                label = LABELS[pred_idx]
                print(f"Predicted: {label} (confidence: {confidence:.0%})")

                if confidence >= CONFIDENCE_CUTOFF:
                    prediction_history.append(label)
                else:
                    prediction_history.append("uncertain")
                    print("Confidence too low. Ignored.")

                if len(prediction_history) == SMOOTHING_WINDOW_SIZE:
                    vote_counts = Counter(prediction_history)
                    most_common_label, count = vote_counts.most_common(1)[0]

                    if count >= MIN_VOTES_REQUIRED:
                        print(f"--> Action Confirmed: {most_common_label} (votes: {count}/{SMOOTHING_WINDOW_SIZE})")

                        match most_common_label:
                            case "uncertain":
                                print("Decision uncertain. Skipped")
                            case "left_hand":
                                speller.move(Direction.LEFT)
                            case "right_hand":
                                speller.move(Direction.RIGHT)

                        prediction_history.clear()

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