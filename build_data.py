import os
import numpy as np
from src.data.loading import download_dataset, load_raw_subjects
from src.data.preprocessing import epoch_with_params

def main():
    print("1. Pobieranie listy plików z PhysioNet...")
    subjects_dict = download_dataset(desired_runs=["R04", "R08", "R12"])
    
    subset = {k: subjects_dict[k] for k in list(subjects_dict.keys())[:5]}
    
    print("2. Ładowanie surowych plików EDF (mne.io)...")
    raw_data = load_raw_subjects(subset)
    
    print("3. Preprocessing (Cięcie na okna 4s i filtrowanie)...")
    X, y, subjects = epoch_with_params(
        raw_data=raw_data,
        low_freq=7.0,     # Filtr dolnoprzepustowy
        high_freq=30.0,   # Filtr górnoprzepustowy (pasmo mu/beta)
        tmin=0.0,
        tmax=4.0,         # Długość epoki 4 sekundy
        task_mode="binary" # Dwie klasy, zgodnie z ustawieniami w starter_bci.py
    )
    
    print("4. Zapisywanie na dysk...")
    os.makedirs("data", exist_ok=True)
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)
    print("Gotowe! Możesz wracać do testowania modelu.")

if __name__ == "__main__":
    main()