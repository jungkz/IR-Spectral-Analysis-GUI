import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

CORRELATION_WAVENUMBER_RANGE = [700, 1250]

def load_and_interpolate_csv(file_path, target_wavenumbers=None):
    data = pd.read_csv(file_path, header=None)
    x = data.iloc[:, 0].to_numpy()
    y = data.iloc[:, 1].to_numpy()
    if target_wavenumbers is None:
        return x, y
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
    return target_wavenumbers, f(target_wavenumbers)

def run_pca_analysis(sample_paths, reference_paths=None, n_components=2):
    all_paths = sample_paths + (reference_paths if reference_paths else [])
    if len(all_paths) < 2:
        raise ValueError("At least two spectra are required for PCA.")

    labels = [os.path.basename(p).replace(".csv", "") for p in all_paths]

    base_x, _ = load_and_interpolate_csv(all_paths[0])
    if CORRELATION_WAVENUMBER_RANGE:
        base_x = base_x[(base_x >= CORRELATION_WAVENUMBER_RANGE[0]) & 
                        (base_x <= CORRELATION_WAVENUMBER_RANGE[1])]

    spectra = []
    for path in all_paths:
        _, y_interp = load_and_interpolate_csv(path, base_x)
        spectra.append(y_interp)

    data_matrix = np.array(spectra)
    if np.isnan(data_matrix).any():
        data_matrix = np.nan_to_num(data_matrix)

    if data_matrix.shape[1] == 0:
        raise ValueError("PCA failed: no overlapping wavenumber range.")

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_matrix)

    fig, ax = plt.subplots()
    ax.set_title("PCA Plot")
    ax.scatter(transformed[:, 0], transformed[:, 1])
    for i, label in enumerate(labels):
        ax.annotate(label, (transformed[i, 0], transformed[i, 1]))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()

    return {
        "pca": pca,
        "components": transformed,
        "labels": labels,
        "figure": fig
    }