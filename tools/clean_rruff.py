import os
import pandas as pd

def clean_rruff_files(file_paths: list[str]):
    os.makedirs('data/references', exist_ok=True)
    os.makedirs('data/processed_rruff_txt', exist_ok=True)

    for path in file_paths:
        filename = os.path.basename(path)
        if not filename.endswith('.txt'):
            continue

        with open(path, 'r') as file:
            lines = file.readlines()

        # Extract only the numeric data lines (lines 11 to 1878 if 1-indexed)
        numeric_lines = lines[10:1878]  # Python 0-indexed, so line 11 is index 10

        data = []
        for line in numeric_lines:
            try:
                x, y = map(float, line.strip().split(','))
                data.append((x, y))
            except ValueError:
                continue  # Skip lines that don't have valid float pairs

        # Construct dataframe and write to CSV without headers
        df = pd.DataFrame(data, columns=["Wavenumber", "Absorbance"])
        rock_name = filename.split('__')[0]
        csv_path = os.path.join('data', 'references', f"{rock_name}.csv")
        df.to_csv(csv_path, index=False, header=False)
        print(f"[✓] Cleaned {filename} → {csv_path}")

        # Move original file to archive folder
        processed_path = os.path.join('data', 'processed_rruff_txt', filename)
        os.rename(path, processed_path)