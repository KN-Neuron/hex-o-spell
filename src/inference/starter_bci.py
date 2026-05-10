"""
EEGNet model + preprocessing for inference.

Hparams of a checkpoint can be auto-derived from state_dict tensor shapes,
which is critical for swapping checkpoints (e.g. 16-channel BrainAccess MIDI
retrain) without touching code.

The auto-detection formula (assuming pk1, pk2 are fixed structural constants):
    f1          = block1.0.weight.shape[0]
    temp_kernel = block1.0.weight.shape[3]
    chans       = block2.0.weight.shape[2]
    d           = block2.0.weight.shape[0] // f1
    f2          = block3.1.weight.shape[0]
    classes     = fc.weight.shape[0]
    time_points = pk1 * pk2 * (fc.weight.shape[1] // f2)
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
        chans: int = 64,
        classes: int = 3,
        time_points: int = 641,
        temp_kernel: int = 80,
        f1: int = 16,
        f2: int = 64,
        d: int = 4,
        pk1: int = 4,
        pk2: int = 8,
        dropout_rate: float = 0.5,
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


# Pooling kernels are structural — reflect EEGNet topology, not a swap-knob.
# If you ever need to change them per-checkpoint, save them alongside weights.
_PK1 = 4
_PK2 = 8


def derive_hparams(state_dict: dict) -> dict:
    """Reconstruct EEGNet hparams from a state_dict.

    Raises KeyError if the state_dict is not an EEGNet of this topology.
    """
    f1 = state_dict["block1.0.weight"].shape[0]
    temp_kernel = state_dict["block1.0.weight"].shape[3]
    df1 = state_dict["block2.0.weight"].shape[0]
    chans = state_dict["block2.0.weight"].shape[2]
    f2 = state_dict["block3.1.weight"].shape[0]
    classes = state_dict["fc.weight"].shape[0]
    fc_in = state_dict["fc.weight"].shape[1]

    if df1 % f1 != 0:
        raise ValueError(
            f"Inconsistent state_dict: d*f1={df1} not divisible by f1={f1}"
        )
    d = df1 // f1

    if fc_in % f2 != 0:
        raise ValueError(
            f"Inconsistent state_dict: fc_in={fc_in} not divisible by f2={f2}"
        )
    time_points = _PK1 * _PK2 * (fc_in // f2)

    return dict(
        chans=chans,
        classes=classes,
        time_points=time_points,
        temp_kernel=temp_kernel,
        f1=f1,
        d=d,
        f2=f2,
        pk1=_PK1,
        pk2=_PK2,
    )


def load_model(
    path: str = "data/model/final_best.pth",
    device: str = "cpu",
) -> nn.Module:
    """Load an EEGNet checkpoint with auto-detection of hparams.

    Works with any EEGNet checkpoint of this topology — both the original
    PhysioNet 64ch/160Hz model and any future 16ch/250Hz BrainAccess retrain
    will load without code changes, as long as the saved file is a state_dict.
    """
    state_dict = torch.load(path, map_location=device, weights_only=False)
    hparams = derive_hparams(state_dict)
    model = EEGNet(**hparams)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING — musi być identyczny jak przy treningu
# ═══════════════════════════════════════════════════════════════
# Input:  surowy sygnał EEG (chans, n_samples)
# Output: tensor (1, 1, chans, time_points) gotowy do model()
#
# UWAGA: parametry preprocessingu (bandpass, sfreq) MUSZĄ pasować do treningu.
# Domyślne wartości pasują do binarnego checkpointa final_best.pth wytrenowanego
# na PhysioNet EEGMMIDB (160 Hz, 0–49 Hz bandpass per config new_full_binary).
# Dla retrenowanego modelu BrainAccess MIDI: sfreq=250, dostosować bandpass.

# Klasy wybrane przez argmax na wyjściu modelu.
# Kolejność musi pasować do mapowania użytego przy treningu.
# Dla starego 3-klasowego modelu: {0: "rest", 1: "left_hand", 2: "right_hand"}.
# Dla obecnego binarnego (final_best.pth): {0: "left_hand", 1: "right_hand"}
LABELS_BINARY = {0: "left_hand", 1: "right_hand"}
LABELS_THREEWAY = {0: "rest", 1: "left_hand", 2: "right_hand"}


def preprocess(
    epoch: np.ndarray,
    sfreq: float = 160.0,
    bandpass_low: float = 0.0,
    bandpass_high: float = 49.0,
    filter_order: int = 5,
) -> torch.Tensor:
    """(chans, n_samples) surowego EEG → tensor gotowy do modelu.

    Bandpass + z-score per kanał. Domyślne wartości pasują do treningu
    `new_full_binary_all_channels_4way` z motor-imagery-AI.

    Jeśli `bandpass_low <= 0`, używa lowpass (highcut only) — jak w configu
    z bandpass [0.0, 49.0].
    """
    if bandpass_low > 0:
        sos = butter(
            filter_order,
            [bandpass_low, bandpass_high],
            btype="band",
            fs=sfreq,
            output="sos",
        )
    else:
        sos = butter(filter_order, bandpass_high, btype="low", fs=sfreq, output="sos")

    x = sosfiltfilt(sos, epoch, axis=-1).astype(np.float32)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Sanity check: load the model and run inference on data/X.npy if present.
        # X.npy/y.npy generation is the responsibility of motor-imagery-AI repo.
        try:
            model = load_model()
            print(f"Model loaded. Hparams: {derive_hparams(model.state_dict())}")
        except FileNotFoundError as e:
            print(f"Checkpoint not found at default path: {e}")
            print("Place a .pth file at data/model/final_best.pth or pass a path.")
            sys.exit(1)

        try:
            X = np.load("data/X.npy")
            y = np.load("data/y.npy")
        except FileNotFoundError:
            print("data/X.npy or data/y.npy not found.")
            print("Generate them via motor-imagery-AI repo's training pipeline.")
            sys.exit(1)

        labels = LABELS_BINARY if model.fc.out_features == 2 else LABELS_THREEWAY
        for i in np.random.choice(len(X), min(5, len(X)), replace=False):
            tensor = torch.from_numpy(X[i : i + 1]).unsqueeze(1).float()
            probs = model(tensor).softmax(dim=1)[0]
            pred = probs.argmax().item()
            print(
                f"y={labels.get(int(y[i]), '?'):<12s} "
                f"pred={labels[pred]:<12s} conf={probs[pred]:.0%}"
            )
