import os
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd


class EEGDataExtractorV2:
    def __init__(
        self,
        data_dir: str,
        lfreq: float = 1,
        hfreq: float = 100,
        notch_filter: Optional[List[float]] = None,
        baseline: Optional[Tuple] = None,
        tmin: float = 0,
        tmax: float = 3,
    ):
        """
        Parameters:
            data_dir (str): Directory with .fif files.
            lfreq (float): Low cutoff frequency for bandpass filtering.
            hfreq (float): High cutoff frequency for bandpass filtering.
            notch_filter (list): Frequencies for notch filtering (e.g. 50Hz noise).
            baseline (tuple): Baseline correction period.
            tmin (float): Start time (in seconds) relative to the event.
            tmax (float): End time (in seconds) relative to the event.
        """
        self.data_dir = data_dir
        self.lfreq = lfreq
        self.hfreq = hfreq
        self.notch_filter = notch_filter or [50]
        self.baseline = baseline
        self.tmin = tmin
        self.tmax = tmax

    def _read_from_dir(self) -> List[str]:
        """Returns a list of .fif files in the data directory."""
        return [f for f in os.listdir(self.data_dir) if f.endswith(".fif")]

    def _load_eeg(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Loads each .fif file, applies filtering and amplitude-based artifact rejection,
        converts units, extracts events and epochs, and maps event codes to labels.
        """
        files = self._read_from_dir()
        eeg_and_events = []
        participants = []

        for file in files:
            participant_id = os.path.splitext(file)[0]
            file_path = os.path.join(self.data_dir, file)

            eeg = self._process_eeg_file(file_path)

            events, event_id = mne.events_from_annotations(eeg)
            if not event_id:
                print(f"No events found in file {file}")
                continue

            id_to_label = {v: k for k, v in event_id.items()}

            epochs = self._create_epochs_with_rejection(eeg, events, event_id)

            numeric_labels = epochs.events[:, -1]
            labels = [
                id_to_label.get(label_id, "unknown") for label_id in numeric_labels
            ]

            eeg_and_events.append(
                {"participant_id": participant_id, "epochs": epochs, "labels": labels}
            )
            participants.append({"participant_id": participant_id, "file": file})

        return eeg_and_events, participants

    def _process_eeg_file(self, file_path: str) -> mne.io.Raw:
        """Load and preprocess an EEG file with channel selection and filtering."""
        eeg = mne.io.read_raw_fif(file_path, preload=True)
        eeg = self._select_eeg_channels(eeg)
        eeg = self._convert_units_to_millivolts(eeg)
        eeg = self._apply_filters(eeg)
        return eeg

    def _select_eeg_channels(self, eeg: mne.io.Raw) -> mne.io.Raw:
        """Select only EEG channels, excluding stimulus and EOG channels."""
        return eeg.pick_types(eeg=True, stim=False, eog=False, exclude="bads")

    def _convert_units_to_millivolts(self, eeg: mne.io.Raw) -> mne.io.Raw:
        """Convert EEG units from microvolts to millivolts."""
        microvolts_to_millivolts_conversion_factor = 10**-6
        return eeg.apply_function(
            lambda x: x * microvolts_to_millivolts_conversion_factor
        )

    def _apply_filters(self, eeg: mne.io.Raw) -> mne.io.Raw:
        """Apply bandpass and notch filtering to the EEG data."""
        eeg.filter(l_freq=self.lfreq, h_freq=self.hfreq)
        if self.notch_filter:
            eeg.notch_filter(self.notch_filter)
        return eeg

    def _create_epochs_with_rejection(
        self, eeg: mne.io.Raw, events: np.ndarray, event_id: Dict[str, int]
    ) -> mne.Epochs:
        """Create epochs from the filtered EEG data with artifact rejection."""
        artifact_rejection_criteria = dict(eeg=150e-6)
        return mne.Epochs(
            raw=eeg,
            events=events,
            event_id=event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            reject=artifact_rejection_criteria,
            preload=True,
        )

    def extract_dataframe(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Iterates over each participant's data, extracts each epoch as a numpy array,
        and returns a DataFrame with columns: participant_id, epoch, label.
        Also returns a list of participants with metadata.
        """

        eeg_and_events, participants = self._load_eeg()
        data = []
        for item in eeg_and_events:
            participant_id = item["participant_id"]
            epochs = item["epochs"]
            labels = item["labels"]
            epoch_data = epochs.get_data()

            for i, label in enumerate(labels):
                single_epoch_data = epoch_data[i]
                data.append(
                    {
                        "participant_id": participant_id,
                        "epoch": single_epoch_data,
                        "label": label,
                    }
                )
        df = pd.DataFrame(data)
        return df, participants

    def extract_erp_dataframe(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Iterates over each participant's data, computes ERP by averaging epochs,
        and returns a DataFrame with columns: participant_id, label, erp.
        Also returns a list of participants with metadata.
        """
        eeg_and_events, participants = self._load_eeg()
        data = []

        for item in eeg_and_events:
            participant_id = item["participant_id"]
            epochs = item["epochs"]
            labels = item["labels"]
            epoch_data = epochs.get_data()

            epochs_by_label: Dict[str, List[np.ndarray]] = {}
            for i, label in enumerate(labels):
                if label not in epochs_by_label:
                    epochs_by_label[label] = []
                epochs_by_label[label].append(epoch_data[i])

            for label, epoch_list in epochs_by_label.items():
                erp_trial_average = np.mean(epoch_list, axis=0)
                erp = np.mean(erp_trial_average, axis=0)
                data.append(
                    {"participant_id": participant_id, "label": label, "erp": erp}
                )

        df = pd.DataFrame(data)
        return df, participants

    def extract_raw_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Loads continuous preprocessed EEG data and returns a list of dictionaries
        containing participant_id and raw_data (n_channels, n_samples).
        Also returns a list of participants with metadata.
        """
        files = self._read_from_dir()
        raw_results = []
        participants = []

        for file in files:
            participant_id = os.path.splitext(file)[0]
            file_path = os.path.join(self.data_dir, file)

            eeg = self._process_eeg_file(file_path)
            # Look at the sampling rate (Hz)
            sfreq = eeg.info["sfreq"]

            # Look at the total duration in seconds
            duration = eeg.times[-1]

            print(f"Sampling rate: {sfreq} Hz, Duration: {duration} seconds")
            # mne.io.Raw.get_data() returns an array (n_channels, n_samples)
            raw_data = eeg.get_data()

            raw_results.append({"participant_id": participant_id, "raw_data": raw_data})
            participants.append({"participant_id": participant_id, "file": file})

        return raw_results, participants


if __name__ == "__main__":
    DATA_DIR = "data/raw"

    extractor = EEGDataExtractorV2(data_dir=DATA_DIR)

    raw_data_list, _ = extractor.extract_raw_data()
    if not raw_data_list:
        print("No raw data extracted.")
        exit(1)

    sample_raw: np.ndarray = raw_data_list[0]["raw_data"]

    # Copy paste channels from participant to achieve (64, n_samples)
    target_channel_count = 64
    composite_raw = np.tile(
        sample_raw, (target_channel_count // sample_raw.shape[0] + 1, 1)
    )[:target_channel_count, :250]

    # save the composite raw data to a .npy file for testing
    np.save(f"{DATA_DIR}/example_64ch_250samples.npy", composite_raw)
