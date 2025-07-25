import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import os

def run_ica_analysis(file_paths):
    if len(file_paths) < 2:
        raise ValueError("At least two sample files are required for ICA.")

    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        if df.shape[1] != 2:
            raise ValueError(f"File {os.path.basename(path)} does not have 2 columns.")
        df = df.sort_values(df.columns[0])  # sort by wavenumber
        df.set_index(df.columns[0], inplace=True)
        dfs.append(df)

    all_df = pd.concat(dfs, axis=1, join='inner')
    data_matrix = all_df.T.values  # samples x features

    ica = FastICA(n_components=min(5, len(file_paths)), random_state=42, max_iter=1000, tol=0.001)
    S_ = ica.fit_transform(data_matrix)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    fig, axs = plt.subplots(nrows=min(5, S_.shape[1]), figsize=(10, 2.5 * min(5, S_.shape[1])))
    if min(5, S_.shape[1]) == 1:
        axs = [axs]  # Ensure axs is iterable

    base_names = [os.path.basename(f) for f in file_paths]
    source_info = ", ".join(base_names)
    for i, ax in enumerate(axs):
        ax.plot(S_[:, i])
        ax.set_title(f'IC {i + 1} from: {source_info}', fontsize=10)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Signal')

    plt.tight_layout()

    # plt.show()  # Removed interactive blocking call

    return fig