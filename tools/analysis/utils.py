import numpy as np
import pandas as pd
import os
from spectrochempy import NDDataset

def load_csv_as_nddataset(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.to_numpy()
    return NDDataset(data)

def list_csv_files_in_dir(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".csv")
    ]

def save_nddataset_to_csv(nddataset, file_path):
    x = nddataset.x.data
    y = nddataset.data
    np.savetxt(file_path, np.column_stack((x, y)), delimiter=",", header="Wavenumber,Absorbance", comments="")