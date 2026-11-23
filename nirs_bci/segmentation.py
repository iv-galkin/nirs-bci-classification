from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# 1. States -> segments -> trials
# ============================================================

def extract_segments(labels, fs=None):
    """
    Split a label time series into contiguous segments of equal labels.

    Compatible with data.py:
        extract_segments(labels, fs)
    """
    labels = np.asarray(labels).astype(int)

    if labels.ndim != 1:
        labels = labels.ravel()

    columns = ["label", "start_idx", "end_idx", "duration_samples"]
    if fs is not None:
        columns.append("duration_sec")

    if len(labels) == 0:
        return pd.DataFrame(columns=columns)

    rows = []
    start = 0
    current_label = int(labels[0])

    def make_row(label, start_idx, end_idx):
        duration_samples = end_idx - start_idx
        row = {
            "label": int(label),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "duration_samples": int(duration_samples),
        }
        if fs is not None:
            row["duration_sec"] = duration_samples / float(fs)
        return row

    for i in range(1, len(labels)):
        label = int(labels[i])
        if label != current_label:
            rows.append(make_row(current_label, start, i))
            start = i
            current_label = label

    rows.append(make_row(current_label, start, len(labels)))
    return pd.DataFrame(rows)


def extract_trials_from_segments(segments):
    """
    Extract trials with labels 1/2/3 from segments.

    Convention:
        0 - ignored transition / technical state;
        1 - rest;
        2 - left imagery;
        3 - right imagery.

    First-series block structure:
        8 instructions per block: 4 rest + 2 left + 2 right.
    """
    rows = []
    trial_id = 0

    for _, row in segments.iterrows():
        label = int(row["label"])
        if label not in {1, 2, 3}:
            continue

        block_id = trial_id // 8
        out = {
            "trial_id": int(trial_id),
            "block_id": int(block_id),
            "label": label,
            "start_idx": int(row["start_idx"]),
            "end_idx": int(row["end_idx"]),
            "duration_samples": int(row["duration_samples"]),
        }
        if "duration_sec" in row.index:
            out["duration_sec"] = float(row["duration_sec"])

        rows.append(out)
        trial_id += 1

    return pd.DataFrame(rows)


# ============================================================
# 2. Label transformation
# ============================================================

def transform_labels(y, task):
    """
    Transform original labels 1/2/3 into binary task labels.

    left_vs_right:
        2 -> 0
        3 -> 1
        1 and 0 are discarded

    rest_vs_active:
        1 -> 0
        2/3 -> 1
        0 is discarded
    """
    y = np.asarray(y).astype(int)

    if task == "left_vs_right":
        mask = np.isin(y, [2, 3])
        y_new = y[mask].copy()
        y_new[y_new == 2] = 0
        y_new[y_new == 3] = 1
        return mask, y_new.astype(int)

    if task == "rest_vs_active":
        mask = np.isin(y, [1, 2, 3])
        y_new = y[mask].copy()
        y_new[y_new == 1] = 0
        y_new[np.isin(y_new, [2, 3])] = 1
        return mask, y_new.astype(int)

    raise ValueError(f"Unknown task: {task}")


# ============================================================
# 3. Block-level collection
# ============================================================

def collect_train_timepoints(X, trials, train_blocks, task):
    """Collect training timepoints from selected blocks."""
    X = np.asarray(X, dtype=float)
    train_blocks = set(np.asarray(train_blocks, dtype=int).tolist())

    X_parts = []
    y_parts = []

    for _, row in trials.iterrows():
        if int(row["block_id"]) not in train_blocks:
            continue

        label = int(row["label"])
        start = int(row["start_idx"])
        end = int(row["end_idx"])

        X_seg = X[start:end]
        y_seg = np.full(X_seg.shape[0], label, dtype=int)
        mask, y_new = transform_labels(y_seg, task)

        if len(y_new) == 0:
            continue

        X_parts.append(X_seg[mask])
        y_parts.append(y_new)

    if len(X_parts) == 0:
        return np.empty((0, X.shape[1]), dtype=float), np.empty((0,), dtype=int)

    return np.vstack(X_parts), np.concatenate(y_parts)


def collect_test_windows(X, trials, test_blocks, task, fs, window_sec=1.0):
    """Cut selected blocks into non-overlapping windows."""
    X = np.asarray(X, dtype=float)
    test_blocks = set(np.asarray(test_blocks, dtype=int).tolist())
    window_samples = max(1, int(round(window_sec * fs)))

    windows = []
    labels = []
    trial_ids_out = []
    block_ids_out = []

    for _, row in trials.iterrows():
        block_id = int(row["block_id"])
        if block_id not in test_blocks:
            continue

        label = int(row["label"])
        _, y_tmp = transform_labels(np.array([label]), task)
        if len(y_tmp) == 0:
            continue

        task_label = int(y_tmp[0])
        start = int(row["start_idx"])
        end = int(row["end_idx"])

        pos = start
        while pos + window_samples <= end:
            windows.append(X[pos:pos + window_samples])
            labels.append(task_label)
            trial_ids_out.append(int(row["trial_id"]))
            block_ids_out.append(block_id)
            pos += window_samples

    return (
        windows,
        np.asarray(labels, dtype=int),
        np.asarray(trial_ids_out, dtype=int),
        np.asarray(block_ids_out, dtype=int),
    )


def collect_windows_from_blocks(X, trials, blocks, task, fs, window_sec=1.0):
    """Convenience wrapper returning only windows and labels."""
    windows, y, _, _ = collect_test_windows(
        X=X,
        trials=trials,
        test_blocks=blocks,
        task=task,
        fs=fs,
        window_sec=window_sec,
    )
    return windows, y


def timepoints_to_windows(X, y, window_size):
    """Convert timepoint training data into non-overlapping windows."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)

    windows = []
    labels = []

    for i in range(0, len(X) - window_size + 1, window_size):
        windows.append(X[i:i + window_size])
        labels.append(y[i])

    return windows, np.asarray(labels, dtype=int)


# ============================================================
# 4. Trial-id collection for inner CV in channel selection
# ============================================================

def collect_train_timepoints_by_trial_ids(X, trials, trial_ids, task):
    X = np.asarray(X, dtype=float)
    trial_ids = set(np.asarray(trial_ids, dtype=int).tolist())

    X_parts = []
    y_parts = []

    for _, row in trials.iterrows():
        if int(row["trial_id"]) not in trial_ids:
            continue

        label = int(row["label"])
        start = int(row["start_idx"])
        end = int(row["end_idx"])

        X_seg = X[start:end]
        y_seg = np.full(X_seg.shape[0], label, dtype=int)
        mask, y_new = transform_labels(y_seg, task)

        if len(y_new) == 0:
            continue

        X_parts.append(X_seg[mask])
        y_parts.append(y_new)

    if len(X_parts) == 0:
        return np.empty((0, X.shape[1]), dtype=float), np.empty((0,), dtype=int)

    return np.vstack(X_parts), np.concatenate(y_parts)


def collect_test_windows_by_trial_ids(X, trials, trial_ids, task, fs, window_sec=1.0):
    X = np.asarray(X, dtype=float)
    trial_ids = set(np.asarray(trial_ids, dtype=int).tolist())
    window_samples = max(1, int(round(window_sec * fs)))

    windows = []
    labels = []
    trial_ids_out = []
    block_ids_out = []

    for _, row in trials.iterrows():
        trial_id = int(row["trial_id"])
        if trial_id not in trial_ids:
            continue

        label = int(row["label"])
        _, y_tmp = transform_labels(np.array([label]), task)
        if len(y_tmp) == 0:
            continue

        task_label = int(y_tmp[0])
        start = int(row["start_idx"])
        end = int(row["end_idx"])

        pos = start
        while pos + window_samples <= end:
            windows.append(X[pos:pos + window_samples])
            labels.append(task_label)
            trial_ids_out.append(trial_id)
            block_ids_out.append(int(row["block_id"]))
            pos += window_samples

    return (
        windows,
        np.asarray(labels, dtype=int),
        np.asarray(trial_ids_out, dtype=int),
        np.asarray(block_ids_out, dtype=int),
    )


def make_half_block_validation_splits(trials, outer_train_blocks):
    """Inner CV splits: validation = half block = 1 left + 1 right + 2 rest."""
    outer_train_blocks = np.asarray(outer_train_blocks, dtype=int)
    outer_trials = trials[trials["block_id"].isin(outer_train_blocks)].copy()
    outer_trial_ids = outer_trials["trial_id"].astype(int).values

    splits = []

    for block_id in outer_train_blocks:
        block_trials = trials[trials["block_id"] == block_id].copy()

        rest_ids = block_trials.loc[block_trials["label"] == 1, "trial_id"].astype(int).values
        left_ids = block_trials.loc[block_trials["label"] == 2, "trial_id"].astype(int).values
        right_ids = block_trials.loc[block_trials["label"] == 3, "trial_id"].astype(int).values

        if len(left_ids) != 2 or len(right_ids) != 2 or len(rest_ids) != 4:
            raise ValueError(
                f"Block {block_id}: expected 2 left, 2 right, 4 rest; "
                f"got left={len(left_ids)}, right={len(right_ids)}, rest={len(rest_ids)}"
            )

        for left_id in left_ids:
            for right_id in right_ids:
                for rest_pair in combinations(rest_ids, 2):
                    val_trial_ids = np.array([left_id, right_id, rest_pair[0], rest_pair[1]], dtype=int)
                    train_trial_ids = np.setdiff1d(outer_trial_ids, val_trial_ids)
                    splits.append(
                        {
                            "val_from_block": int(block_id),
                            "inner_val_trial_ids": val_trial_ids,
                            "inner_train_trial_ids": train_trial_ids,
                        }
                    )

    return splits


# ============================================================
# 5. Compatibility helper
# ============================================================

def collect_windows_from_global_blocks(train_items, task, window_sec=1.0):
    """
    Compatibility helper for older code.

    train_items:
        list of tuples (day, X_filtered, block_id, session_idx)
    """
    all_windows = []
    all_y = []

    for day, X, block_id, _ in train_items:
        fs = day["sampling_rate"]
        windows, y, _, _ = collect_test_windows(
            X=X,
            trials=day["trials"],
            test_blocks=[block_id],
            task=task,
            fs=fs,
            window_sec=window_sec,
        )

        if len(y) == 0:
            continue

        all_windows.extend(windows)
        all_y.append(y)

    if len(all_y) == 0:
        return [], np.empty((0,), dtype=int)

    return all_windows, np.concatenate(all_y)
