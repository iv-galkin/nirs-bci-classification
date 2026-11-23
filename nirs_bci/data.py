import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def read_matlab_string(f: h5py.File, obj: Any) -> str:
    """Read MATLAB char/string stored in HDF5 .mat v7.3 file."""
    if isinstance(obj, h5py.Reference):
        obj = f[obj]

    arr = np.array(obj)

    if arr.dtype == np.uint16:
        return "".join(chr(c) for c in arr.ravel() if c != 0)

    if arr.dtype.kind in {"S", "U"}:
        return "".join(arr.astype(str).ravel())

    return str(arr)


def read_ref_string_array(f: h5py.File, dataset_name: str) -> list[str] | None:
    """Read MATLAB cellstr-like reference array from HDF5 .mat v7.3 file."""
    if dataset_name not in f:
        return None

    ds = f[dataset_name]
    arr = np.array(ds)

    result = []
    for ref in arr.ravel():
        if isinstance(ref, h5py.Reference):
            result.append(read_matlab_string(f, ref))
        else:
            result.append(str(ref))
    return result


def load_nirs_mat_v73(file_path: str | Path) -> dict:
    """
    Load one MATLAB v7.3 file with NIRS data.

    Returned X has shape time x features and feature order:
        [HbO_0, ..., HbO_C-1, HbR_0, ..., HbR_C-1]
    """
    file_path = Path(file_path)

    with h5py.File(file_path, "r") as f:
        HbO = np.array(f["HbO"], dtype=float).T
        Hb = np.array(f["Hb"], dtype=float).T

        wl1 = np.array(f["wl1"], dtype=float).T if "wl1" in f else None
        wl2 = np.array(f["wl2"], dtype=float).T if "wl2" in f else None

        states = np.array(f["States"]).astype(int).ravel()
        sampling_rate = float(np.array(f["sampling_rate"]).ravel()[0])
        day = int(np.array(f["day"]).ravel()[0]) if "day" in f else None
        subject = read_matlab_string(f, f["subject"]) if "subject" in f else None

        channel_labels = read_ref_string_array(f, "channel_labels")
        detector_labels = read_ref_string_array(f, "detector_labels")

        if "emitter_lables" in f:
            emitter_labels = read_ref_string_array(f, "emitter_lables")
        else:
            emitter_labels = read_ref_string_array(f, "emitter_labels")

    if HbO.shape != Hb.shape:
        raise ValueError(f"HbO and HbR have different shapes: {HbO.shape}, {Hb.shape}")

    if HbO.shape[0] != len(states):
        raise ValueError(
            f"Time dimension mismatch: HbO.shape={HbO.shape}, len(states)={len(states)}."
        )

    X = np.concatenate([HbO, Hb], axis=1)

    return {
        "file": str(file_path),
        "file_name": file_path.name,
        "day": day,
        "subject": subject,
        "sampling_rate": sampling_rate,
        "HbO": HbO,
        "Hb": Hb,
        "wl1": wl1,
        "wl2": wl2,
        "States": states,
        "X": X,
        "channel_labels": channel_labels,
        "detector_labels": detector_labels,
        "emitter_labels": emitter_labels,
    }


def day_number(path: str | Path) -> int:
    path = Path(path)
    match = re.search(r"Day_(\d+)\.mat", path.name)
    return int(match.group(1)) if match else 10**9


def load_all_days(root: str | Path, add_segments: bool = True) -> list[dict]:
    """Load all Day_*.mat files from root, sorted by day number."""
    from .segmentation import extract_segments, extract_trials_from_segments

    root = Path(root)
    mat_files = sorted(root.glob("Day_*.mat"), key=day_number)
    days = [load_nirs_mat_v73(p) for p in mat_files]

    if add_segments:
        for day in days:
            day["segments"] = extract_segments(day["States"], day["sampling_rate"])
            day["trials"] = extract_trials_from_segments(day["segments"])

    return days


def summarize_days(days: list[dict]) -> pd.DataFrame:
    rows = []
    for d in days:
        n_time = d["X"].shape[0]
        fs = d["sampling_rate"]
        values, counts = np.unique(d["States"], return_counts=True)
        rows.append({
            "file": d["file_name"],
            "day": d.get("day"),
            "subject": d.get("subject"),
            "n_time": n_time,
            "n_channels": d["HbO"].shape[1],
            "n_features": d["X"].shape[1],
            "fs": fs,
            "duration_sec": n_time / fs,
            "duration_min": n_time / fs / 60,
            "class_counts": dict(zip(values.astype(int), counts.astype(int))),
        })
    return pd.DataFrame(rows)
