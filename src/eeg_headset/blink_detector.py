"""Deliberate blink detection from EEG/EOG.

Algorithm
---------
A deliberate (intentional) blink is distinguished from a spontaneous one by
*duration*: spontaneous blinks last roughly 100-150 ms while deliberate
("hard" or prolonged) blinks last 200-400+ ms.

Pipeline per epoch:
    1. Select the relevant channel(s) (prefrontal Fp1/Fp2 for EEG-as-EOG,
       or the dedicated EOG channel for BioAmp EXG Pill).
    2. Bandpass 0.5-10 Hz (4th-order Butterworth, zero-phase via filtfilt).
       The blink artifact has most of its spectral energy in 1-3 Hz, plus
       slower DC drift below that. Removing >10 Hz strips muscle and EEG
       cortical activity that we don't want triggering false positives.
    3. Compute |signal| (rectified).
    4. Adaptive threshold = baseline_median + threshold_k * baseline_MAD.
       MAD-based threshold is robust against the blink itself biasing the
       baseline estimate — large excursions are masked by the median.
    5. Find contiguous runs above threshold. A run of duration in
       [min_duration_s, max_duration_s] counts as a deliberate blink.
       Below min: spontaneous, ignored. Above max: probably DC drift or
       movement artifact, also ignored.

References
----------
- Agarwal & Sivakumar, "BLINK: A Fully Automated Unsupervised Algorithm for
  Eye-Blink Detection in EEG Signals", 2019.
- Imperial College thesis on EOG-based BCI control (Spiral repository).
- Upside Down Labs application notes for BioAmp EXG Pill EOG mode.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt


class BlinkSource(enum.Enum):
    """Where the blink signal comes from — affects channel selection only."""

    # EOG signal recorded on prefrontal EEG electrodes (Fp1, Fp2). Cheap fallback
    # when no separate EOG hardware is available; SNR is okay because mrugnięcia
    # generate huge artifacts (~50-200 µV) on Fp electrodes due to their proximity
    # to the eyes.
    PREFRONTAL_EEG = "prefrontal_eeg"

    # Dedicated EOG channel from BioAmp EXG Pill (or similar) connected to
    # electrodes placed above and below one eye. Cleaner signal, but requires
    # extra hardware and a separate driver/stream.
    DEDICATED_EOG = "dedicated_eog"


@dataclass
class BlinkDetectorConfig:
    """All thresholds and timing params in one place; tune empirically per subject."""

    bandpass_low: float = 0.5
    bandpass_high: float = 10.0
    filter_order: int = 4

    # Adaptive threshold = median(|x|) + threshold_k * MAD(|x|).
    # Higher k → fewer false positives, more missed blinks.
    threshold_k: float = 5.0

    # Hard floor in µV for the threshold. If the baseline is silent (e.g. first
    # epoch with all zeros from a freshly-connected MockDriver), the MAD-based
    # threshold collapses to ~0 and any noise triggers a "blink". This floor
    # prevents that pathology.
    threshold_min_uv: float = 30.0

    # Duration constraints (seconds).
    min_duration_s: float = 0.2  # below this: spontaneous blink, ignored.
    max_duration_s: float = 1.0  # above this: drift/movement artifact, ignored.

    # If the epoch is too short for filter stability, skip detection.
    min_samples_for_filter: int = 100


class BlinkDetector:
    """Detect deliberate blinks in a single epoch of EEG/EOG data.

    Designed to be called once per pipeline step on a (channels, samples)
    epoch buffer. Returns True iff at least one deliberate blink was detected
    in the window.

    Caller is responsible for *debouncing* across consecutive calls — that's
    a pipeline concern, not a detector concern.
    """

    def __init__(
        self,
        source: BlinkSource = BlinkSource.PREFRONTAL_EEG,
        frontal_channel_idx: Optional[list[int]] = None,
        config: Optional[BlinkDetectorConfig] = None,
    ):
        self.source = source
        # For PREFRONTAL_EEG: indices of Fp1/Fp2 (or just Fp1) in the headset's
        # channel layout. Defaults to [0, 1] which matches MIDI_16CH_BASE.
        # For DEDICATED_EOG: typically [0] (single channel).
        self.frontal_channel_idx = frontal_channel_idx or [0, 1]
        self.config = config or BlinkDetectorConfig()

    def detect(self, window: np.ndarray, sfreq: float) -> bool:
        """
        Parameters
        ----------
        window : np.ndarray, shape (n_channels, n_samples)
            Raw EEG/EOG epoch. Units expected to be µV.
        sfreq : float
            Sampling frequency of `window`, in Hz.

        Returns
        -------
        bool
            True if at least one contiguous run of |signal| > threshold has
            duration in [min_duration_s, max_duration_s].
        """
        if window.size == 0 or window.shape[1] < self.config.min_samples_for_filter:
            return False

        # ─── 1. Channel selection ───────────────────────────────────────────
        idx = [i for i in self.frontal_channel_idx if i < window.shape[0]]
        if not idx:
            # Defensive: if config asks for channels that don't exist, just use ch 0.
            idx = [0]
        channel_data = window[idx, :]

        # Average across selected frontal channels (Fp1 + Fp2 → bipolar-ish proxy).
        # For DEDICATED_EOG with one channel this is a no-op.
        signal_1d = channel_data.mean(axis=0)

        # ─── 2. Bandpass filter ─────────────────────────────────────────────
        try:
            sos = butter(
                self.config.filter_order,
                [self.config.bandpass_low, self.config.bandpass_high],
                btype="band",
                fs=sfreq,
                output="sos",
            )
            filtered = sosfiltfilt(sos, signal_1d)
        except ValueError:
            # Filter design fails for very short signals — degrade gracefully.
            return False

        # ─── 3. Rectify ─────────────────────────────────────────────────────
        rectified = np.abs(filtered)

        # ─── 4. Adaptive threshold via MAD ──────────────────────────────────
        median = np.median(rectified)
        mad = np.median(np.abs(rectified - median))
        # Convert MAD to standard-deviation-equivalent scale for normal data.
        # The 1.4826 factor makes MAD a consistent estimator of σ for Gaussian noise.
        sigma_est = 1.4826 * mad
        threshold = max(
            median + self.config.threshold_k * sigma_est,
            self.config.threshold_min_uv,
        )

        above = rectified > threshold
        if not above.any():
            return False

        # ─── 5. Find contiguous runs and check duration ─────────────────────
        # Runs of consecutive True values in `above`. Use diff trick: run starts
        # where above goes 0→1, ends where 1→0.
        padded = np.concatenate(([False], above, [False]))
        diffs = np.diff(padded.astype(np.int8))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]

        min_samples = int(self.config.min_duration_s * sfreq)
        max_samples = int(self.config.max_duration_s * sfreq)

        for start, end in zip(run_starts, run_ends):
            duration = end - start
            if min_samples <= duration <= max_samples:
                return True

        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers (used by tests/test_blink_detector.py)
# ═══════════════════════════════════════════════════════════════════════════════


def synthesize_blink_signal(
    sfreq: float = 250.0,
    duration_s: float = 4.0,
    blink_at_s: float = 1.0,
    blink_duration_s: float = 0.3,
    blink_amplitude_uv: float = 150.0,
    noise_amplitude_uv: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a synthetic 1-channel signal with a single blink-like artifact.

    Useful for unit-testing the detector. The blink is modeled as a Gaussian
    bump over `blink_duration_s`, which approximates the typical positive
    deflection seen on Fp1/Fp2 during a blink.

    Returns shape (1, n_samples).
    """
    rng = rng or np.random.default_rng(42)
    n_samples = int(duration_s * sfreq)
    t = np.arange(n_samples) / sfreq

    # Background: pink-ish noise (just white noise + tiny alpha is fine for tests).
    signal = rng.normal(0, noise_amplitude_uv, n_samples)

    # Blink bump.
    center = blink_at_s
    sigma = blink_duration_s / 4  # ≈99% of the bump fits in blink_duration_s
    bump = blink_amplitude_uv * np.exp(-((t - center) ** 2) / (2 * sigma**2))
    signal += bump

    return signal[np.newaxis, :]
