"""
Ring-O-Spell — model + preprocessing + dane
Resztę (driver, pipeline, streaming, UI glue) robisz sam.

Szybki test:
    python starter_bci.py --prepare          # ściąga dataset → data/
    python starter_bci.py --test             # sprawdza czy model działa
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, sosfiltfilt

# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════


class EEGNet(nn.Module):
    def __init__(
        self,
        chans=64,
        classes=3,
        time_points=641,
        temp_kernel=80,
        f1=16,
        f2=64,
        d=4,
        pk1=4,
        pk2=8,
        dropout_rate=0.5,
    ):
        super().__init__()
        linear_size = (time_points // (pk1 * pk2)) * f2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                d * f1, d * f1, (1, 16), groups=d * f1, padding="same", bias=False
            ),
            nn.Conv2d(d * f1, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        return self.fc(x)


def load_model(
    path: str = "data/model/3class_eegnet_best.pth", device: str = "cpu"
) -> nn.Module:
    model = EEGNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING — musi być identyczny jak przy treningu
# ═══════════════════════════════════════════════════════════════
# Input:  surowy sygnał EEG (64, n_samples)
# Output: tensor (1, 1, 64, 641) gotowy do model()
#
# Parametry treningu:
#   sfreq        = 160 Hz
#   bandpass      = 7–30 Hz (butterworth order 5)
#   epoka         = 4.0s → 641 próbek
#   normalizacja  = z-score per kanał

SFREQ = 160
EPOCH_SAMPLES = 641
LABELS = {0: "rest", 1: "left_hand", 2: "right_hand"}


def preprocess(epoch: np.ndarray) -> torch.Tensor:
    """(64, 641) surowego EEG → tensor gotowy do modelu."""
    sos = butter(5, [7.0, 30.0], btype="band", fs=SFREQ, output="sos")
    x = sosfiltfilt(sos, epoch, axis=-1).astype(np.float32)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        model = load_model()
        X = np.load("data/X.npy")
        y = np.load("data/y.npy")
        # 5 losowych epok
        for i in np.random.choice(len(X), 5, replace=False):
            tensor = (
                torch.from_numpy(X[i : i + 1]).unsqueeze(1).float()
            )  # już przetworzone
            probs = model(tensor).softmax(dim=1)[0]
            pred = probs.argmax().item()
            print(
                f"y={LABELS[y[i]]:<12s} pred={LABELS[pred]:<12s} conf={probs[pred]:.0%}"
            )
