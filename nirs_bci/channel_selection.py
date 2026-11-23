from itertools import combinations
import numpy as np
import pandas as pd

from .evaluation import dissertation_score_fixed_labels, get_task_classes
from .models import GaussianWindowDiscriminant
from .preprocessing import preprocess_nirs
from .segmentation import collect_train_timepoints, collect_test_windows, collect_train_timepoints_by_trial_ids, collect_test_windows_by_trial_ids


def channel_feature_indices(channel_ids, n_channels: int):
    """Feature indices for X order [HbO all | HbR all]."""
    idx = []
    for ch in channel_ids:
        ch = int(ch)
        idx.extend([ch, ch + n_channels])
    return np.asarray(idx, dtype=int)


def make_half_block_validation_splits(trials, outer_train_blocks):
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
            raise ValueError(f"Block {block_id}: expected 2 left, 2 right, 4 rest")
        for left_id in left_ids:
            for right_id in right_ids:
                for rest_pair in combinations(rest_ids, 2):
                    val_ids = np.array([left_id, right_id, rest_pair[0], rest_pair[1]], dtype=int)
                    train_ids = np.setdiff1d(outer_trial_ids, val_ids)
                    splits.append({"val_from_block": int(block_id), "inner_val_trial_ids": val_ids, "inner_train_trial_ids": train_ids})
    return splits


def score_channel_set_inner_cv(X, trials, outer_train_blocks, task, model_name, channel_ids, fs, window_sec=1.0):
    n_channels = X.shape[1] // 2
    feat_idx = channel_feature_indices(channel_ids, n_channels)
    labels_fixed = get_task_classes(task)
    scores = []
    for split in make_half_block_validation_splits(trials, outer_train_blocks):
        X_train, y_train = collect_train_timepoints_by_trial_ids(X[:, feat_idx], trials, split["inner_train_trial_ids"], task)
        val_windows, y_val, _, _ = collect_test_windows_by_trial_ids(X[:, feat_idx], trials, split["inner_val_trial_ids"], task, fs, window_sec)
        if len(y_train) == 0 or len(y_val) == 0:
            scores.append(np.nan)
            continue
        clf = GaussianWindowDiscriminant(mode=model_name)
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict_windows(val_windows)
            P, _, _ = dissertation_score_fixed_labels(y_val, y_pred, labels_fixed)
            scores.append(P)
        except Exception:
            scores.append(np.nan)
    return np.nan if np.all(pd.isna(scores)) else float(np.nanmean(scores))


def select_best_single_channel(X, trials, outer_train_blocks, task, model_name, fs, candidate_channels=None, window_sec=1.0):
    n_channels = X.shape[1] // 2
    if candidate_channels is None:
        candidate_channels = list(range(n_channels))
    rows = []
    for ch in candidate_channels:
        rows.append({
            "channels": (int(ch),),
            "n_channels": 1,
            "cv_score": score_channel_set_inner_cv(X, trials, outer_train_blocks, task, model_name, [ch], fs, window_sec),
        })
    table = pd.DataFrame(rows).sort_values("cv_score", ascending=False, na_position="last")
    return list(table.iloc[0]["channels"]), float(table.iloc[0]["cv_score"]), table


def select_best_pair_channels(X, trials, outer_train_blocks, task, model_name, fs, candidate_channels=None, window_sec=1.0):
    n_channels = X.shape[1] // 2
    if candidate_channels is None:
        candidate_channels = list(range(n_channels))
    rows = []
    for ch1, ch2 in combinations(candidate_channels, 2):
        rows.append({
            "channels": (int(ch1), int(ch2)),
            "n_channels": 2,
            "cv_score": score_channel_set_inner_cv(X, trials, outer_train_blocks, task, model_name, [ch1, ch2], fs, window_sec),
        })
    table = pd.DataFrame(rows).sort_values("cv_score", ascending=False, na_position="last")
    return list(table.iloc[0]["channels"]), float(table.iloc[0]["cv_score"]), table


def greedy_channel_selection_from_start(X, trials, outer_train_blocks, task, model_name, fs, start_channels, candidate_channels=None, window_sec=1.0):
    n_channels = X.shape[1] // 2
    if candidate_channels is None:
        candidate_channels = list(range(n_channels))
    selected = [int(ch) for ch in start_channels]
    remaining = [int(ch) for ch in candidate_channels if int(ch) not in selected]
    path_rows = []
    start_score = score_channel_set_inner_cv(X, trials, outer_train_blocks, task, model_name, selected, fs, window_sec)
    path_rows.append({"step": len(selected), "added_channel": None, "channels": tuple(selected), "n_channels": len(selected), "cv_score": start_score})
    while len(remaining) > 0:
        candidates = []
        for ch in remaining:
            trial_channels = selected + [ch]
            candidates.append({"candidate_channel": ch, "channels": tuple(trial_channels), "cv_score": score_channel_set_inner_cv(X, trials, outer_train_blocks, task, model_name, trial_channels, fs, window_sec)})
        candidates_df = pd.DataFrame(candidates).sort_values("cv_score", ascending=False, na_position="last")
        best_row = candidates_df.iloc[0]
        best_ch = int(best_row["candidate_channel"])
        selected.append(best_ch)
        remaining.remove(best_ch)
        path_rows.append({"step": len(selected), "added_channel": best_ch, "channels": tuple(selected), "n_channels": len(selected), "cv_score": float(best_row["cv_score"])})
    path = pd.DataFrame(path_rows)
    best_idx = path["cv_score"].astype(float).idxmax()
    return list(path.loc[best_idx, "channels"]), float(path.loc[best_idx, "cv_score"]), path


def run_lobo_one_day_with_channel_selection(day, task, model_name, filter_mode, selection_method, candidate_channels=None, window_sec=1.0):
    valid_methods = {"all_channels", "best_single", "greedy_from_single", "best_pair", "greedy_from_pair"}
    if selection_method not in valid_methods:
        raise ValueError(f"Unknown selection_method: {selection_method}")
    fs = day["sampling_rate"]
    trials = day["trials"]
    X_filtered = preprocess_nirs(day["X"], fs, filter_mode)
    n_channels = X_filtered.shape[1] // 2
    if candidate_channels is None:
        candidate_channels = list(range(n_channels))
    unique_blocks = np.sort(trials["block_id"].unique())
    rows = []
    for train_blocks in combinations(unique_blocks, 2):
        train_blocks = np.asarray(train_blocks, dtype=int)
        test_blocks = np.setdiff1d(unique_blocks, train_blocks)
        start_channels = None
        if selection_method == "all_channels":
            selected_channels = list(candidate_channels); selection_cv_score = np.nan
        elif selection_method == "best_single":
            selected_channels, selection_cv_score, _ = select_best_single_channel(X_filtered, trials, train_blocks, task, model_name, fs, candidate_channels, window_sec)
        elif selection_method == "best_pair":
            selected_channels, selection_cv_score, _ = select_best_pair_channels(X_filtered, trials, train_blocks, task, model_name, fs, candidate_channels, window_sec)
        elif selection_method == "greedy_from_single":
            start_channels, _, _ = select_best_single_channel(X_filtered, trials, train_blocks, task, model_name, fs, candidate_channels, window_sec)
            selected_channels, selection_cv_score, _ = greedy_channel_selection_from_start(X_filtered, trials, train_blocks, task, model_name, fs, start_channels, candidate_channels, window_sec)
        else:
            start_channels, _, _ = select_best_pair_channels(X_filtered, trials, train_blocks, task, model_name, fs, candidate_channels, window_sec)
            selected_channels, selection_cv_score, _ = greedy_channel_selection_from_start(X_filtered, trials, train_blocks, task, model_name, fs, start_channels, candidate_channels, window_sec)

        feat_idx = channel_feature_indices(selected_channels, n_channels)
        X_train, y_train = collect_train_timepoints(X_filtered[:, feat_idx], trials, train_blocks, task)
        test_windows, y_test, _, _ = collect_test_windows(X_filtered[:, feat_idx], trials, test_blocks, task, fs, window_sec)
        base = {
            "file": day["file_name"], "day": day.get("day"), "subject": day.get("subject"),
            "task": task, "model": model_name, "filter": filter_mode, "selection_method": selection_method,
            "train_blocks": tuple(train_blocks.tolist()), "test_blocks": tuple(test_blocks.tolist()),
            "start_channels": tuple(start_channels) if start_channels is not None else None,
            "selected_channels": tuple(selected_channels), "n_selected_channels": len(selected_channels),
            "selection_cv_score": selection_cv_score, "n_train_samples": len(y_train), "n_test_windows": len(y_test),
        }
        if len(y_train) == 0 or len(y_test) == 0:
            rows.append({**base, "score": np.nan, "Q": None, "confusion_matrix": None, "status": "skipped_empty_data", "error": None})
            continue
        clf = GaussianWindowDiscriminant(mode=model_name)
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict_windows(test_windows)
            P, cm, Q = dissertation_score_fixed_labels(y_test, y_pred, labels=get_task_classes(task))
            rows.append({**base, "score": P, "Q": Q, "confusion_matrix": cm.tolist(), "status": "ok", "error": None})
        except Exception as e:
            rows.append({**base, "score": np.nan, "Q": None, "confusion_matrix": None, "status": "failed", "error": repr(e)})
    return pd.DataFrame(rows)
