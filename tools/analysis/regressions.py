
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress

smoothing_window_length = 11
smoothing_poly_order = 3
zero_deriv_match_tolerance = 15
regression_wavenumber_min = 500
regression_wavenumber_max = 2000

def run_derivative_regression_analysis(sample_file_path, reference_file_path, comparison_type="reference"):
    df_reference = pd.read_csv(reference_file_path, header=None, names=['Wavenumber', 'Absorbance'])
    df_reference['Wavenumber'] = pd.to_numeric(df_reference['Wavenumber'], errors='coerce')
    df_reference['Absorbance'] = pd.to_numeric(df_reference['Absorbance'], errors='coerce')
    df_reference.dropna(inplace=True)
    df_reference = df_reference.sort_values(by='Wavenumber').reset_index(drop=True)

    df_sample = pd.read_csv(sample_file_path, header=None, names=['Wavenumber', 'Absorbance'])
    df_sample = df_sample.iloc[1:].copy()
    df_sample['Wavenumber'] = pd.to_numeric(df_sample['Wavenumber'], errors='coerce')
    df_sample['Absorbance'] = pd.to_numeric(df_sample['Absorbance'], errors='coerce')
    df_sample.dropna(inplace=True)
    df_sample = df_sample.sort_values(by='Wavenumber').reset_index(drop=True)

    label_sample = "Sample 1" if comparison_type == "sample" else "Sample"
    label_comparison = "Sample 2" if comparison_type == "sample" else "Reference"

    min_common_wavenumber = max(df_sample['Wavenumber'].min(), df_reference['Wavenumber'].min())
    max_common_wavenumber = min(df_sample['Wavenumber'].max(), df_reference['Wavenumber'].max())
    df_reference_filtered = df_reference[
        (df_reference['Wavenumber'] >= min_common_wavenumber) &
        (df_reference['Wavenumber'] <= max_common_wavenumber)
    ].copy()
    df_sample_filtered = df_sample[
        (df_sample['Wavenumber'] >= min_common_wavenumber) &
        (df_sample['Wavenumber'] <= max_common_wavenumber)
    ].copy()
    common_wavenumbers = np.sort(np.unique(np.concatenate([
        df_reference_filtered['Wavenumber'].values,
        df_sample_filtered['Wavenumber'].values
    ])))
    common_wavenumbers = common_wavenumbers[
        (common_wavenumbers >= min_common_wavenumber) &
        (common_wavenumbers <= max_common_wavenumber)
    ]
    from scipy.interpolate import interp1d
    f_reference_interp = interp1d(df_reference_filtered['Wavenumber'], df_reference_filtered['Absorbance'], kind='linear', fill_value="extrapolate")
    f_sample_interp = interp1d(df_sample_filtered['Wavenumber'], df_sample_filtered['Absorbance'], kind='linear', fill_value="extrapolate")
    interpolated_absorbance_reference = f_reference_interp(common_wavenumbers)
    interpolated_absorbance_sample = f_sample_interp(common_wavenumbers)
    df_aligned = pd.DataFrame({
        'Wavenumber': common_wavenumbers,
        'Absorbance_reference': interpolated_absorbance_reference,
        'Absorbance_sample': interpolated_absorbance_sample
    })

    swl = smoothing_window_length
    spo = smoothing_poly_order
    if swl > len(df_aligned['Absorbance_reference']):
        swl = len(df_aligned['Absorbance_reference']) - 1
        if swl % 2 == 0:
            swl -= 1
    if spo >= swl:
        spo = swl - 1
        if spo < 0: spo = 0
    df_aligned['Absorbance_reference_smoothed'] = savgol_filter(
        df_aligned['Absorbance_reference'], swl, spo
    )
    df_aligned['Absorbance_sample_smoothed'] = savgol_filter(
        df_aligned['Absorbance_sample'], swl, spo
    )
    df_aligned = df_aligned[
        df_aligned['Absorbance_reference_smoothed'].notnull() &
        np.isfinite(df_aligned['Absorbance_reference_smoothed']) &
        df_aligned['Absorbance_sample_smoothed'].notnull() &
        np.isfinite(df_aligned['Absorbance_sample_smoothed'])
    ]

    sample_derivative = np.gradient(df_aligned['Absorbance_sample_smoothed'], df_aligned['Wavenumber'])
    reference_derivative = np.gradient(df_aligned['Absorbance_reference_smoothed'], df_aligned['Wavenumber'])

    def find_zero_crossings(wavenumbers, derivative):
        zero_crossings = []
        for i in range(1, len(derivative)):
            if (derivative[i-1] * derivative[i] < 0) or (derivative[i-1] == 0 and derivative[i] != 0):
                x1, y1 = wavenumbers[i-1], derivative[i-1]
                x2, y2 = wavenumbers[i], derivative[i]
                if (y2 - y1) != 0:
                    zero_wavenumber = x1 - y1 * (x2 - x1) / (y2 - y1)
                    zero_crossings.append(zero_wavenumber)
                elif y1 == 0:
                    zero_crossings.append(x1)
        return np.array(zero_crossings)

    sample_zero_derivative_wavenumbers = find_zero_crossings(df_aligned['Wavenumber'].values, sample_derivative)
    reference_zero_derivative_wavenumbers = find_zero_crossings(df_aligned['Wavenumber'].values, reference_derivative)

    # Smoothed plots
    plt.figure(figsize=(14, 6))
    plt.suptitle(f"{label_sample} and {label_comparison} Spectrum Derivative (Smoothed) Plots (Plots 13 & 14):", fontsize=15, y=1.04)
    plt.subplot(1, 2, 1)
    plt.plot(df_aligned['Wavenumber'], sample_derivative, label=f'{label_sample} Derivative (Smoothed)', color='blue')
    plt.plot(sample_zero_derivative_wavenumbers, np.zeros_like(sample_zero_derivative_wavenumbers), 'o', color='darkblue', markersize=6, label='Zero Derivative Points')
    plt.xlabel('Wavenumber ($cm^{-1}$)')
    plt.ylabel('Derivative (dAbsorbance/d$\\nu$)')
    plt.title(f'{label_sample} Spectrum Derivative (Smoothed)')
    plt.gca().invert_xaxis()
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.plot(df_aligned['Wavenumber'], reference_derivative, label=f'{label_comparison} Derivative (Smoothed)', color='red')
    plt.plot(reference_zero_derivative_wavenumbers, np.zeros_like(reference_zero_derivative_wavenumbers), 'x', color='darkred', markersize=6, label='Zero Derivative Points')
    plt.xlabel('Wavenumber ($cm^{-1}$)')
    plt.ylabel('Derivative (dAbsorbance/d$\\nu$)')
    plt.title(f'{label_comparison} Spectrum Derivative (Smoothed)')
    plt.gca().invert_xaxis()
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Linear regressions on zero derivatives
    sample_zeros_filt = sample_zero_derivative_wavenumbers[
        (sample_zero_derivative_wavenumbers >= regression_wavenumber_min) &
        (sample_zero_derivative_wavenumbers <= regression_wavenumber_max)
    ]
    reference_zeros_filt = reference_zero_derivative_wavenumbers[
        (reference_zero_derivative_wavenumbers >= regression_wavenumber_min) &
        (reference_zero_derivative_wavenumbers <= regression_wavenumber_max)
    ]
    # Matching logic: nearest neighbor, one-to-one match, within tolerance
    matched_sample_zeros = []
    matched_reference_zeros = []
    sample_zeros_sorted = np.sort(sample_zeros_filt)
    reference_zeros_sorted = np.sort(reference_zeros_filt)
    temp_reference_zeros = list(reference_zeros_sorted)
    for s_zero in sample_zeros_sorted:
        if not temp_reference_zeros:
            matched_sample_zeros.append(s_zero)
            matched_reference_zeros.append(np.nan)
            continue
        closest_r_zero_idx = np.argmin(np.abs(np.array(temp_reference_zeros) - s_zero))
        closest_r_zero = temp_reference_zeros[closest_r_zero_idx]
        if np.abs(s_zero - closest_r_zero) < zero_deriv_match_tolerance:
            matched_sample_zeros.append(s_zero)
            matched_reference_zeros.append(closest_r_zero)
            temp_reference_zeros.pop(closest_r_zero_idx)
        else:
            matched_sample_zeros.append(s_zero)
            matched_reference_zeros.append(np.nan)
    for r_zero in temp_reference_zeros:
        matched_sample_zeros.append(np.nan)
        matched_reference_zeros.append(r_zero)
    valid_matched_sample_zeros = [z for z in matched_sample_zeros if not np.isnan(z)]
    valid_matched_reference_zeros = [z for z in matched_reference_zeros if not np.isnan(z)]
    if len(valid_matched_sample_zeros) > 1 and len(valid_matched_reference_zeros) > 1:
        min_len_zeros = min(len(valid_matched_sample_zeros), len(valid_matched_reference_zeros))
        valid_matched_sample_zeros = valid_matched_sample_zeros[:min_len_zeros]
        valid_matched_reference_zeros = valid_matched_reference_zeros[:min_len_zeros]
        slope_zero_deriv, intercept_zero_deriv, r_value_zero_deriv, p_value_zero_deriv, std_err_zero_deriv = \
            linregress(valid_matched_reference_zeros, valid_matched_sample_zeros)
        plt.figure(figsize=(8, 6))
        plt.scatter(valid_matched_reference_zeros, valid_matched_sample_zeros, label='Matched Zero Derivative Points', color='green')
        plt.plot(valid_matched_reference_zeros, intercept_zero_deriv + slope_zero_deriv * np.array(valid_matched_reference_zeros),
                 color='purple', linestyle='--', label=f'Linear Regression: y = {slope_zero_deriv:.2f}x + {intercept_zero_deriv:.2f}\n$R^2$ = {r_value_zero_deriv**2:.2f}')
        min_val = min(min(valid_matched_reference_zeros), min(valid_matched_sample_zeros))
        max_val = max(max(valid_matched_reference_zeros), max(valid_matched_sample_zeros))
        plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle=':', label='y = x Line')
        plt.xlabel(f'{label_comparison} Zero Derivative Wavenumber ($cm^{{-1}}$)', fontsize=12)
        plt.ylabel(f'{label_sample} Zero Derivative Wavenumber ($cm^{{-1}}$)', fontsize=12)
        title_suffix = "(Sample vs. Sample)" if comparison_type == "sample" else "(Sample vs. Reference)"
        plt.title(f'Linear Regression of Zero Derivative Wavenumbers {title_suffix}', fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Residuals of the regression ---
    if len(valid_matched_sample_zeros) > 1 and len(valid_matched_reference_zeros) > 1:
        residuals_zero_deriv = np.array(valid_matched_sample_zeros) - np.array(valid_matched_reference_zeros)
        plt.figure(figsize=(12, 7))
        plt.scatter(valid_matched_reference_zeros, residuals_zero_deriv,
                    alpha=0.7, s=30, color='blue', label=f'Residuals ({label_sample} - {label_comparison})')
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero Residuals')
        title_suffix = "(Sample vs. Sample)" if comparison_type == "sample" else "(Sample vs. Reference)"
        plt.title(f"Residuals of Zero Derivative Wavenumbers Regression {title_suffix}")
        plt.xlabel(f"{label_comparison} Zero Derivative Wavenumber ($cm^{{-1}}$)")
        plt.ylabel(f"Residuals ({label_sample} ZD - {label_comparison} ZD) ($cm^{{-1}}$)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_derivative_regression_analysis("sample.csv", "reference.csv", comparison_type="reference")
