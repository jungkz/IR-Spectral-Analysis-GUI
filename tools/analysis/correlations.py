import numpy as np
from scipy.stats import pearsonr, spearmanr

CORRELATION_WAVENUMBER_RANGE = [700, 1300]  # Set to None to use full range

# Helper to safely compute correlations
def safe_correlation(x, y):
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan, np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    try:
        pearson = pearsonr(x, y)[0]
    except Exception:
        pearson = np.nan
    try:
        spearman = spearmanr(x, y)[0]
    except Exception:
        spearman = np.nan
    return pearson, spearman

def run_correlation_analysis(file1_path, file2_path, interpolate=True):
    try:
        data1 = np.loadtxt(file1_path, delimiter=',', skiprows=1)
        data2 = np.loadtxt(file2_path, delimiter=',', skiprows=1)

        wn1, ab1 = data1[:, 0], data1[:, 1]
        wn2, ab2 = data2[:, 0], data2[:, 1]

        if CORRELATION_WAVENUMBER_RANGE:
            mask1 = (wn1 >= CORRELATION_WAVENUMBER_RANGE[0]) & (wn1 <= CORRELATION_WAVENUMBER_RANGE[1])
            mask2 = (wn2 >= CORRELATION_WAVENUMBER_RANGE[0]) & (wn2 <= CORRELATION_WAVENUMBER_RANGE[1])
            wn1, ab1 = wn1[mask1], ab1[mask1]
            wn2, ab2 = wn2[mask2], ab2[mask2]

        if interpolate:
            # Interpolate file2 absorbance to file1 wavenumber axis
            ab2_aligned = np.interp(wn1, wn2, ab2)
        else:
            # Ensure lengths match for direct comparison
            if len(ab1) != len(ab2):
                raise ValueError("Wavenumber grids do not match and interpolation is disabled.")
            ab2_aligned = ab2

        # Compute Pearson and Spearman correlations
        pearson_r, spearman_rho = safe_correlation(ab1, ab2_aligned)
        return pearson_r, spearman_rho

    except Exception as e:
        print(f"Correlation analysis failed: {e}")
        return None, None