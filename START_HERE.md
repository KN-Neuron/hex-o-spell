# START HERE — Jak odpalić BrainBoard

To jest **jeden projekt Pythona**. Klawiaturę wybierasz flagą `--layout` przy uruchamianiu.
Nie ma trzech projektów ani osobnych instalacji.

---

## Krok 1: Rozpakuj ZIP

```bash
unzip BrainBoard-final.zip
cd BrainBoard
```

## Krok 2: Zainstaluj zależności

Masz dwie opcje:

### Opcja A (zalecana, jak miałeś wcześniej): Poetry

```bash
poetry install
```

### Opcja B: zwykły venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Krok 3: Sprawdź że wszystko działa

```bash
PYTHONPATH=. poetry run pytest        # lub: PYTHONPATH=. python -m pytest
```

Powinno wyświetlić **87 passed**. Jeśli tak — wszystko jest OK.

---

## Jak to odpalić — 5 sposobów do wyboru

### 1. **GUI** (najfajniejszy podgląd) ✨

Pełnoprawne okno z 3 układami klawiatury do wyboru, wpisywanym tekstem i sugestiami słów.

```bash
PYTHONPATH=. poetry run python -m src.gui.main_window
```

W okienku:
- **Toolbar u góry**: wybierasz **Hex-O-Spell** / **Ring-O-Spell** / **Tree (Huffman)**
- **Pole tekstowe**: pokazuje co wpisałeś
- **Pasek sugestii**: 3 propozycje słów po angielsku (klikalne myszką lub klawiszem 1/2/3)
- **Klawiatura**: aktualnie wybrany layout

Sterowanie z klawiatury:
- `←` / `a` — w lewo
- `→` / `d` — w prawo
- `Spacja` / `Enter` / `w` — select (blink)
- `Backspace` / `s` — back/cofnij
- `1`/`2`/`3` — wstaw odpowiednią sugestię słowa

### 2. Najprostszy podgląd: REPL z klawiatury

Tylko `Ty + klawiatura komputera`. Naciskasz L/R/B i widzisz jak speller się porusza.
**To jest tryb na pokazy / żeby pokazać znajomemu jak to działa.**

```bash
# Klawiaturka bigram-adaptive (polskie kontynuacje, układ z Ring-O-Spell):
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.scripted_repl --layout bigram

# Klawiaturka 5x6 (oryginalna):
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.scripted_repl --layout static
```

Sterowanie:
- `L` lub `a` — ruch w lewo
- `R` lub `d` — ruch w prawo
- `B` lub `Enter` — blink (zatwierdzenie)
- `S` — back (cofnięcie)
- `Q` — wyjście

### 2. Pełen pipeline na fake danych (bez czepka)

Pełny przepływ: syntetyczny EEG → klasyfikator EEGNet → blink detector → speller.
**To jest tryb żeby zobaczyć cały system w akcji bez sprzętu.**

```bash
# Bigram layout + scripted intencje:
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver scripted \
    --scripted-sequence blink,blink,right,blink,blink \
    --layout bigram \
    --headset-model SAMPLE_64CH

# Albo z losowym mockiem (model zobaczy losowe sinusy):
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver mock --layout bigram --headset-model SAMPLE_64CH
```

### 3. Z prawdziwym czepkiem BrainAccess MIDI

```bash
# Najpierw test połączenia z czepkiem:
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.brainaccess_sanity \
    --model MIDI_16CH_BASE

# Potem odpalenie klawiatury:
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver brainaccess --headset-model MIDI_16CH_BASE \
    --layout bigram
```

### 4. Z czepkiem + dedykowanym EOG (BioAmp EXG Pill)

1. Wgraj `firmware/bioamp_exg_pill.ino` do Arduino/ESP32 przez Arduino IDE.
2. Podłącz elektrody (nad okiem, pod okiem, referencja na czole).
3. Sprawdź port szeregowy (np. `/dev/ttyUSB0` na Linux, COM3 na Windows).
4. Odpal:

```bash
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.run_keyboard \
    --driver brainaccess --headset-model MIDI_16CH_BASE \
    --bioamp-port /dev/ttyUSB0 \
    --layout bigram
```

---

## Wybór klawiatury — najczęstsze pytanie

```bash
--layout static   # oryginalny 5×6, hardkodowany alfabet
--layout bigram   # 6×6 ring, dynamicznie zmienia zawartość po każdej literze
```

To **jedna flaga**, ten sam projekt, te same komendy. Speller to drop-in
replacement — wszystko inne (driver, pipeline, model, blink detector) jest
identyczne.

---

## Jeśli będziesz wymieniał model na 16-kanałowy retrain

Wystarczy że podmienisz plik:

```bash
cp twoj_nowy_model.pth data/model/final_best.pth
```

Plus podasz odpowiedni sample rate (BrainAccess MIDI ma 250 Hz, nie 160):

```bash
... --preprocess-sfreq 250 --preprocess-bandpass-low 7.0 --preprocess-bandpass-high 30.0
```

Parametry modelu (f1, d, f2, liczba kanałów, klas) wykrywają się automatycznie ze
state_dict — nie musisz nic edytować w kodzie.

---

## Co jeśli coś nie działa

### Test "tylko speller, bez niczego"
```bash
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.scripted_repl --layout bigram
```
Jeśli to nie działa — zła instalacja Pythona/poetry.

### Test "wszystko bez sprzętu"
```bash
PYTHONPATH=. poetry run pytest
```
Powinno być 87 passed.

### Test "model się ładuje"
```bash
PYTHONPATH=. poetry run python -c "from src.inference.starter_bci import load_model; m = load_model(); print('OK', m.fc.out_features, 'klasy')"
```
Powinno wypisać `OK 2 klasy`.

### Test "BrainAccess SDK gada z czepkiem"
```bash
PYTHONPATH=. poetry run python -m src.eeg_headset.cmd.brainaccess_sanity --model MIDI_16CH_BASE
```

Jeśli `brainaccess SDK not installed` — musisz go doinstalować osobno (oficjalna wheel z BrainAccess), to jest poza pip'em.

---

## Co gdzie leży

```
BrainBoard/
├── START_HERE.md              ← ten plik
├── README.md                  ← bardziej szczegółowa dokumentacja
├── pyproject.toml             ← poetry, jak dotąd
├── requirements.txt
├── headsets.yaml              ← konfiguracja czepków (BrainAccess MIDI itd.)
├── data/
│   ├── model/final_best.pth   ← Twój model EEGNet, gotowy
│   └── language/polish_bigrams.json   ← tabela bigramów dla layoutu bigram
├── firmware/
│   └── bioamp_exg_pill.ino    ← sketch do flashowania Arduino/ESP32
├── src/
│   ├── eeg_headset/
│   │   ├── cmd/
│   │   │   ├── run_keyboard.py        ← główna komenda
│   │   │   ├── scripted_repl.py       ← REPL z klawiatury
│   │   │   ├── brainaccess_sanity.py  ← test połączenia z BrainAccess
│   │   │   └── demo.py                ← legacy demo
│   │   ├── drivers/                   ← mock, playback, brainaccess, bioamp, scripted
│   │   ├── blink_detector.py
│   │   └── eeg_headset.py
│   ├── inference/
│   │   ├── pipeline.py        ← BCIPipeline łączy wszystko
│   │   └── starter_bci.py     ← EEGNet, load_model auto-detekcja
│   └── speller/
│       ├── speller.py         ← façade
│       ├── state.py           ← state machine (Idle→Writing→Sector→Letter)
│       ├── layout.py          ← protokół layoutu
│       ├── static_layout.py   ← oryginalny 5×6
│       └── bigram_layout.py   ← bigram-adaptive Polski
└── tests/                     ← 87 testów
```
