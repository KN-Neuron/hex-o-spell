

Motor-imagery BCI keyboard. Three signals — left-hand MI, right-hand MI, and
deliberate blink — drive a sector/letter speller via a state machine.

## Architecture

```
   ┌──────────────┐                ┌──────────────────┐
   │ BrainAccess  │   16ch EEG     │                  │   move(L|R)
   │ MIDI / Mock  │───────────────▶│   BCIPipeline    │──────────────▶┌──────────┐
   │ / Playback   │   250 Hz       │                  │               │  Speller │
   └──────────────┘                │  • EEGNet (MI)   │   select()    │  (state  │
                                   │  • BlinkDetector │──────────────▶│  machine)│
   ┌──────────────┐                │  • smoothing     │               └──────────┘
   │ BioAmp EXG   │   1ch EOG      │  • debounce      │
   │ Pill (opt.)  │───────────────▶│                  │
   │ via serial   │   500 Hz       └──────────────────┘
   └──────────────┘
```

Speller state machine: `Idle → Writing → SectorNavigation → LetterNavigation
→ (commit letter) → Writing`. Blink advances/selects, L/R navigates.

## Running

### Demo (no hardware)

```bash
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard --driver mock
```

### Screenshots

#### Hex-o-Spell
<img width="1710" height="1112" alt="image" src="https://github.com/user-attachments/assets/b2d69835-835b-4999-90ed-3d4879f2d874" />

#### Ring-o-spell
<img width="1710" height="1112" alt="image" src="https://github.com/user-attachments/assets/e8938526-6e83-49e2-a227-1976c9782d93" />


#### Tree like implementation (experimental to find fastest way of typing with word suggestions)
<img width="1710" height="1112" alt="image" src="https://github.com/user-attachments/assets/02d4dd9b-76bb-4f0b-9e93-d780403fe969" />



### Interactive REPL — "see what's going on" mode

Bypass the EEG pipeline and drive the speller directly with keyboard input.
This is the easiest way to understand how the keyboard behaves before
plugging in hardware.

```bash
# Bigram-adaptive layout (Polish, context-sensitive)
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.scripted_repl --layout bigram

# Static 5×6 grid (the original BrainBoard layout)
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.scripted_repl --layout static
```

Controls: `L`/`a` = left, `R`/`d` = right, `B`/`w`/`Enter` = select (blink),
`S` = back, `Q` = quit.

### Scripted driver — end-to-end regression test

Synthesizes EEG that should classify as the given intent sequence. Useful
for testing the full pipeline (driver → MI classifier → blink detector →
speller) without hardware:

```bash
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver scripted \
    --scripted-sequence blink,blink,right,right,blink,blink \
    --layout bigram --headset-model SAMPLE_64CH
```

Note: each scripted intent takes `--epoch-seconds` (default 4s) of wall-clock
time, so a 6-intent sequence is a 24s demo.

### With recorded data

```bash
# Requires data/X.npy from the motor-imagery-AI repo's training pipeline.
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver playback --playback-source data/X.npy \
    --headset-model SAMPLE_64CH
```

### Real BrainAccess MIDI (no separate EOG)

```bash
# Sanity check first:
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.brainaccess_sanity \
    --model MIDI_16CH_BASE

# Then run the keyboard. Blink reads from Fp1/Fp2 (channels 0, 1 of MIDI).
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver brainaccess --headset-model MIDI_16CH_BASE
```

### Real BrainAccess MIDI + BioAmp EXG Pill (full setup)

1. Flash `firmware/bioamp_exg_pill.ino` to your ESP32 / Arduino / Maker Uno.
2. Connect electrodes (vertical EOG: above eye, below eye, reference on
   forehead).
3. Find the serial port (`ls /dev/ttyUSB*` on Linux, Device Manager on Windows).
4. Run:

```bash
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver brainaccess --headset-model MIDI_16CH_BASE \
    --bioamp-port /dev/ttyUSB0 --bioamp-baud 115200
```

## Layouts

The speller has a pluggable layout system. Two are shipped:

**StaticGridLayout** (default, `--layout static`): Original 5×6 grid,
hard-coded letter assignment. No context.

**BigramAdaptiveLayout** (`--layout bigram`): 6×6 ring keyboard with
context-sensitive contents. After each letter is committed, all 36 ring
positions are re-ranked by `P(next | last_letter)` from a precomputed
Polish bigram table (`data/language/polish_bigrams.json`). The 6 most likely
continuations sit in sector 0, the next 6 in sector 1, etc. Cold start uses
the unigram distribution conditioned on space (i.e. "first letter of a
word" frequencies).

To plug in your own layout, implement the `SpellerLayout` Protocol from
`src/speller/layout.py` (4 methods: `n_sectors`, `letters_in_sector`,
`on_letter_committed`, `reset`) and pass an instance to `Speller(layout=...)`.

## Model

The default checkpoint at `data/model/final_best.pth` is the binary L/R
PhysioNet model (64 channels, 160 Hz). `load_model()` auto-derives all hparams
from the state_dict — swap in a 16-channel BrainAccess MIDI retrain by just
replacing the `.pth` file. **No code edit required.**

If your retrain uses different preprocessing, override at runtime:

```bash
... --preprocess-sfreq 250 --preprocess-bandpass-low 7.0 --preprocess-bandpass-high 30.0
```

## Tests

```bash
poetry run pytest                        # full suite (~10 s)
poetry run pytest tests/test_blink_detector.py -v
poetry run pytest tests/test_pipeline.py -v
poetry run pytest tests/test_bioamp_driver.py -v
```

## Tuning blink detection

Defaults work for synthetic test signals (150 µV blink, 5 µV noise). For real
subjects you'll likely want to lower `threshold_min_uv` or adjust `threshold_k`
in `BlinkDetectorConfig` after a baseline session. The detector exposes all
thresholds as constructor args:

```python
from src.eeg_headset.blink_detector import BlinkDetector, BlinkDetectorConfig
detector = BlinkDetector(
    config=BlinkDetectorConfig(
        threshold_k=4.0,        # lower → more sensitive
        threshold_min_uv=20.0,  # floor against quiet baselines
        min_duration_s=0.25,    # higher → reject more spontaneous blinks
    )
)
```

## Project layout

```
src/
├── eeg_headset/
│   ├── eeg_headset.py        — high-level streaming wrapper
│   ├── blink_detector.py     — bandpass + MAD + duration check
│   ├── ring_buffer.py        — fixed-capacity sample buffer
│   ├── headset_config.py     — YAML-backed channel/sfreq config
│   ├── drivers/
│   │   ├── headset_driver.py — Protocol all drivers conform to
│   │   ├── mock.py           — synthetic data, no hardware
│   │   ├── playback.py       — replay from .npy
│   │   ├── brainaccess.py    — real BrainAccess SDK
│   │   └── bioamp.py         — serial-attached BioAmp EXG Pill
│   └── cmd/
│       ├── run_keyboard.py       — main entry point
│       ├── brainaccess_sanity.py — connectivity smoke test
│       └── demo.py               — annotation-only legacy demo
├── inference/
│   ├── pipeline.py           — BCIPipeline class wiring everything together
│   └── starter_bci.py        — EEGNet, load_model (auto-hparams), preprocess
└── speller/
    ├── speller.py            — Speller façade
    └── state.py              — Idle / Writing / Sector / Letter states

firmware/
└── bioamp_exg_pill.ino       — reference Arduino sketch matching the driver
```
