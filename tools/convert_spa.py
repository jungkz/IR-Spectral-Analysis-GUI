import os
import numpy as np
from spectrochempy import read_omnic
import pandas as pd

def convert_spa_files(file_paths: list[str]):
    os.makedirs('data/samples', exist_ok=True)
    os.makedirs('data/processed_spa', exist_ok=True)

    for path in file_paths:
        filename = os.path.basename(path)
        try:
            dataset = read_omnic(path)

            # Determine whether the data is in transmittance
            try:
                label = str(dataset.units[1]).lower()
            except Exception:
                label = ""

            if 'transmittance' in label or (np.mean(dataset.data) > 1.5):
                dataset.data = -np.log10(dataset.data / 100)
                print(f"[i] Converted to absorbance manually: {filename}")
            else:
                print(f"[i] Already in absorbance or unknown format: {filename}")

            if dataset.x is None or dataset.data is None:
                raise ValueError("Missing spectral x or data values.")
            if np.isnan(dataset.data).any() or np.isinf(dataset.data).any():
                raise ValueError("Invalid absorbance values (NaN or inf) detected.")

            wn = np.ravel(dataset.x.to('1/cm').data)
            ab = np.ravel(dataset.data)
            df = pd.DataFrame({'Wavenumber': wn, 'Absorbance': ab})

            sample_name = os.path.splitext(filename)[0]
            csv_path = os.path.join('data', 'samples', f"{sample_name}.csv")
            df.to_csv(csv_path, index=False, header=False)
            print(f"[✓] Converted {filename} → {csv_path}")

            processed_path = os.path.join('data', 'processed_spa', filename)
            os.rename(path, processed_path)

        except Exception as e:
            print(f"[!] Failed to convert {filename}: {e}")