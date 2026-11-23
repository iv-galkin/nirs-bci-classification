import numpy as np
import pandas as pd
from scipy.signal import welch

from .preprocessing import preprocess_nirs, split_hbo_hbr


def make_fnirs_bands(fs: float):
    """Frequency bands for exploratory fNIRS spectral diagnostics."""
    nyquist = fs / 2
    return [
        ("very_slow_<0.01", 0.0, 0.01),
        ("mayer_0.01_0.05", 0.01, 0.05),
        ("respiration_0.1_0.5", 0.1, min(0.5, nyquist)),
        ("heart_0.8_2.0", 0.8, min(2.0, nyquist)),
    ]


def compute_band_power(freqs, psd, f_min: float, f_max: float):
    mask = (freqs >= f_min) & (freqs <= f_max)
    if mask.sum() < 2:
        return np.nan
    return np.trapz(psd[mask], freqs[mask])


def band_power_summary_hbo_hbr(day, filters, bands, filter_labels=None, nperseg=None):
    fs = day["sampling_rate"]
    X_raw = day["X"]

    if filter_labels is None:
        filter_labels = {f: f for f in filters}

    if nperseg is None:
        nperseg = min(4096, X_raw.shape[0])

    rows = []

    for mode in filters:
        X = preprocess_nirs(X_raw, fs, mode)
        X_hbo, X_hbr = split_hbo_hbr(X)

        for component_name, X_part in [("HbO", X_hbo), ("HbR", X_hbr)]:
            freqs, psds = welch(
                X_part,
                fs=fs,
                axis=0,
                nperseg=min(nperseg, X_part.shape[0]),
            )
            median_psd = np.nanmedian(psds, axis=1)

            row = {
                "filter": mode,
                "filter_label": filter_labels.get(mode, mode),
                "component": component_name,
            }

            for band_name, f_min, f_max in bands:
                row[band_name] = compute_band_power(freqs, median_psd, f_min, f_max)

            rows.append(row)

    return pd.DataFrame(rows)


def normalize_band_power_to_raw(band_table, band_cols):
    df = band_table.copy()
    raw = df[df["filter"] == "raw"].set_index("component")[band_cols]

    rows = []

    for _, row in df.iterrows():
        component = row["component"]
        new_row = row.copy()

        for col in band_cols:
            raw_value = raw.loc[component, col]
            if pd.isna(raw_value) or raw_value == 0:
                new_row[col + "_rel_to_raw"] = np.nan
            else:
                new_row[col + "_rel_to_raw"] = row[col] / raw_value

        rows.append(new_row)

    return pd.DataFrame(rows)
