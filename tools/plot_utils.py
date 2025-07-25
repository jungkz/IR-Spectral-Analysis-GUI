import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set folder paths to reflect actual folder structure
compare_output_folder = "data/plot_output"
os.makedirs(compare_output_folder, exist_ok=True)

# Folder containing your processed SPA CSVs (converted to absorbance)

def plot_selected_spectra(sample_files: list[str], reference_files: list[str], save_plot=False, plot_mode='overlay'):
    if plot_mode == 'stacked':
        total_plots = len(sample_files) + len(reference_files)
        fig, axs = plt.subplots(total_plots, 1, figsize=(10, 2 * total_plots), sharex=True)
        if total_plots == 1:
            axs = [axs]

        all_files = [(reference_files, '-'), (sample_files, '--')]
        idx = 0
        for file_group, style in all_files:
            for filepath in file_group:
                if filepath.endswith('.csv') and os.path.exists(filepath):
                    df = pd.read_csv(filepath, header=None, names=["Wavenumber", "Absorbance"])
                    axs[idx].plot(df["Wavenumber"], df["Absorbance"], linestyle=style,
                                  label=os.path.basename(filepath).replace(".csv", "").split("_IR")[0])
                    axs[idx].invert_xaxis()
                    axs[idx].legend(fontsize='small', loc='upper right')
                    axs[idx].set_ylabel("Absorbance")
                    idx += 1
        axs[0].set_title("IR Spectra Comparison")
        axs[-1].set_xlabel("Wavenumber (cm⁻¹)")

        fig.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        for filepath in reference_files:
            if filepath.endswith('.csv') and os.path.exists(filepath):
                df = pd.read_csv(filepath, header=None, names=["Wavenumber", "Absorbance"])
                ax.plot(df["Wavenumber"], df["Absorbance"], label=os.path.basename(filepath).replace(".csv", "").split("_IR")[0])
        for filepath in sample_files:
            if filepath.endswith('.csv') and os.path.exists(filepath):
                df = pd.read_csv(filepath, header=None, names=["Wavenumber", "Absorbance"])
                ax.plot(df["Wavenumber"], df["Absorbance"], linestyle='--', label=os.path.basename(filepath).replace(".csv", "").split("_IR")[0])

        ax.invert_xaxis()
        ax.set_title("IR Spectra Comparison")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Absorbance")
        ax.legend(fontsize='small', loc='upper right')
        fig.tight_layout()

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(compare_output_folder, f"spectra_plot_{timestamp}.png")
        fig.savefig(output_path)
        print(f"[✓] Saved comparison plot to {output_path}")

    return fig