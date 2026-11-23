from itertools import combinations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .evaluation import dissertation_score
from .features import (
    windows_to_features,
    window_to_grouped_feature,
    n_features_after_grouping,
)
from .models import GaussianWindowDiscriminant, make_classifier
from .preprocessing import preprocess_nirs
from .segmentation import (
    collect_train_timepoints,
    collect_test_windows,
    transform_labels,
    collect_windows_from_blocks,
)


# Compatibility only. Experiment configs should be defined in notebooks.
FEATURE_SET = {}


# ============================================================
# 1. Baseline LOBO within one session: dissertation LDA/QDA
# ============================================================

def run_lobo_one_day(day, task: str, model_name: str, filter_mode: str, window_sec: float = 1.0):
    if model_name not in {"LDA", "QDA"}:
        raise ValueError("model_name must be LDA or QDA")

    fs = day["sampling_rate"]
    trials = day["trials"]
    X_filtered = preprocess_nirs(X=day["X"], fs=fs, mode=filter_mode)

    unique_blocks = np.sort(trials["block_id"].unique())
    if len(unique_blocks) != 3:
        raise ValueError(f"{day['file_name']}: expected 3 blocks, got {len(unique_blocks)}")

    rows = []

    for train_blocks in combinations(unique_blocks, 2):
        train_blocks = np.asarray(train_blocks, dtype=int)
        test_blocks = np.setdiff1d(unique_blocks, train_blocks)

        X_train, y_train = collect_train_timepoints(
            X=X_filtered,
            trials=trials,
            train_blocks=train_blocks,
            task=task,
        )

        test_windows, y_test, _, _ = collect_test_windows(
            X=X_filtered,
            trials=trials,
            test_blocks=test_blocks,
            task=task,
            fs=fs,
            window_sec=window_sec,
        )

        base = {
            "file": day["file_name"],
            "day": day.get("day"),
            "subject": day.get("subject"),
            "task": task,
            "model": model_name,
            "filter": filter_mode,
            "window_sec": window_sec,
            "train_blocks": tuple(train_blocks.tolist()),
            "test_blocks": tuple(test_blocks.tolist()),
            "n_train_samples": len(y_train),
            "n_test_windows": len(y_test),
        }

        if len(y_train) == 0 or len(y_test) == 0:
            rows.append({
                **base,
                "score": np.nan,
                "Q": None,
                "confusion_matrix": None,
                "status": "skipped_empty_data",
                "error": None,
            })
            continue

        clf = GaussianWindowDiscriminant(mode=model_name)

        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict_windows(test_windows)
            P, cm, Q = dissertation_score(y_test, y_pred)
            rows.append({
                **base,
                "score": P,
                "Q": Q,
                "confusion_matrix": cm.tolist(),
                "status": "ok",
                "error": None,
            })
        except Exception as e:
            rows.append({
                **base,
                "score": np.nan,
                "Q": None,
                "confusion_matrix": None,
                "status": "failed",
                "error": repr(e),
            })

    return pd.DataFrame(rows)


def run_subject_filter_experiments(days, task: str, filters, model_name: str = "LDA", window_sec: float = 1.0):
    out = []
    for day in days:
        for filter_mode in filters:
            out.append(
                run_lobo_one_day(
                    day=day,
                    task=task,
                    model_name=model_name,
                    filter_mode=filter_mode,
                    window_sec=window_sec,
                )
            )
    return pd.concat(out, ignore_index=True)


# ============================================================
# 2. Previous blocks / previous sessions experiment
# ============================================================

def fit_and_predict(
    clf_name,
    X_train,
    y_train,
    test_windows,
    y_test,
    train_windows,
    y_train_windows,
    feature_set=None,
):
    """
    Train classifier and return (P_train, P_test).

    LDA:
        dissertation window formula on raw timepoints.

    SVM / CatBoost:
        window-level statistical features specified by feature_set[clf_name].
    """
    clf = make_classifier(clf_name)

    if clf_name == "LDA":
        clf.fit(X_train, y_train)

        y_pred_test = clf.predict_windows(test_windows)
        P_test, _, _ = dissertation_score(y_test, y_pred_test)

        if len(train_windows) == 0:
            P_train = np.nan
        else:
            y_pred_train = clf.predict_windows(train_windows)
            P_train, _, _ = dissertation_score(y_train_windows, y_pred_train)

        return P_train, P_test

    if feature_set is None:
        raise ValueError("For feature-based classifiers pass feature_set explicitly")

    if clf_name not in feature_set:
        raise ValueError(
            f"No features specified for classifier {clf_name}. "
            f"Add it to FEATURE_SET in the notebook."
        )

    features = feature_set[clf_name]
    if features is None:
        raise ValueError(f"Feature-based classifier {clf_name} cannot have features=None")

    if len(train_windows) == 0:
        return np.nan, np.nan

    X_train_features = windows_to_features(train_windows, features)
    X_test_features = windows_to_features(test_windows, features)

    clf.fit(X_train_features, y_train_windows)

    y_pred_train = clf.predict(X_train_features)
    P_train, _, _ = dissertation_score(y_train_windows, y_pred_train)

    y_pred_test = clf.predict(X_test_features)
    P_test, _, _ = dissertation_score(y_test, y_pred_test)

    return P_train, P_test


def run_one_permutation(
    perm_order,
    task,
    clf_name,
    filter_mode,
    preprocessed,
    window_sec=1.0,
    feature_set=None,
):
    """
    Sessions are concatenated into one block sequence according to perm_order.

    exp1:
        train on previous blocks from the same session only.

    exp2:
        train on all previous blocks, including previous sessions.
    """
    days_ordered = [preprocessed[filter_mode][i] for i in perm_order]

    global_blocks = []
    for session_idx, (day, X) in enumerate(days_ordered):
        unique_blocks = np.sort(day["trials"]["block_id"].unique())
        for block_id in unique_blocks:
            global_blocks.append((day, X, int(block_id), int(session_idx)))

    rows_exp1 = []
    rows_exp2 = []

    for global_idx, (day_test, X_test, test_block, session_idx) in enumerate(global_blocks):
        fs = day_test["sampling_rate"]
        train_slice = global_blocks[:global_idx]

        base = {
            "file": day_test["file_name"],
            "day": day_test.get("day"),
            "subject": day_test.get("subject"),
            "task": task,
            "clf": clf_name,
            "filter": filter_mode,
            "window_sec": window_sec,
            "test_block": test_block,
            "global_block_idx": global_idx,
            "session_idx": session_idx,
            "n_train_blocks": len(train_slice),
        }

        test_windows, y_test, _, _ = collect_test_windows(
            X=X_test,
            trials=day_test["trials"],
            test_blocks=[test_block],
            task=task,
            fs=fs,
            window_sec=window_sec,
        )

        same_session_train = [
            (day_tr, X_tr, block_tr, s)
            for day_tr, X_tr, block_tr, s in train_slice
            if s == session_idx
        ]

        rows_exp1.append(
            _run_previous_blocks_variant(
                base=base,
                variant="exp1",
                train_items=same_session_train,
                test_windows=test_windows,
                y_test=y_test,
                task=task,
                clf_name=clf_name,
                window_sec=window_sec,
                feature_set=feature_set,
            )
        )

        rows_exp2.append(
            _run_previous_blocks_variant(
                base=base,
                variant="exp2",
                train_items=train_slice,
                test_windows=test_windows,
                y_test=y_test,
                task=task,
                clf_name=clf_name,
                window_sec=window_sec,
                feature_set=feature_set,
            )
        )

    return pd.DataFrame(rows_exp1), pd.DataFrame(rows_exp2)


def _run_previous_blocks_variant(
    base,
    variant,
    train_items,
    test_windows,
    y_test,
    task,
    clf_name,
    window_sec,
    feature_set,
):
    if len(y_test) == 0:
        return {
            **base,
            "variant": variant,
            "P_train": np.nan,
            "P_test": np.nan,
            "status": "no_test_data",
            "error": None,
        }

    if len(train_items) == 0:
        return {
            **base,
            "variant": variant,
            "P_train": np.nan,
            "P_test": 0.5,
            "status": "no_train_random",
            "error": None,
        }

    X_parts = []
    y_parts = []

    train_windows = []
    window_labels = []
    for day_tr, X_tr, block_tr, _ in train_items:
        windows, labels = collect_windows_from_blocks(
            X=X_tr, trials=day_tr["trials"], blocks=[block_tr], task=task,
            fs=day_tr["sampling_rate"], window_sec=window_sec,
        )
        train_windows.extend(windows)
        window_labels.append(labels)

        X_part, y_part = collect_train_timepoints(
            X=X_tr,
            trials=day_tr["trials"],
            train_blocks=[block_tr],
            task=task,
        )

        if len(y_part) > 0:
            X_parts.append(X_part)
            y_parts.append(y_part)

    if len(X_parts) == 0:
        return {
            **base,
            "variant": variant,
            "P_train": np.nan,
            "P_test": 0.5,
            "status": "no_train_random",
            "error": None,
        }

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    try:
        P_train, P_test = fit_and_predict(
            clf_name=clf_name,
            X_train=X_train,
            y_train=y_train,
            test_windows=test_windows,
            y_test=y_test,
            train_windows=train_windows,
            y_train_windows=np.concatenate(window_labels),
            feature_set=feature_set,
        )

        return {
            **base,
            "variant": variant,
            "P_train": P_train,
            "P_test": P_test,
            "status": "ok",
            "error": None,
        }

    except Exception as e:
        return {
            **base,
            "variant": variant,
            "P_train": np.nan,
            "P_test": np.nan,
            "status": "error",
            "error": repr(e),
        }


def precompute_filters(days, filters):
    return {
        filter_mode: [
            (
                day,
                preprocess_nirs(
                    X=day["X"],
                    fs=day["sampling_rate"],
                    mode=filter_mode,
                ),
            )
            for day in days
        ]
        for filter_mode in filters
    }


# ============================================================
# 3. Channel averaging / collapsed features experiment
# ============================================================

def collect_train_collapsed(X, trials, train_blocks, task, fs, group_size=1, agg="mean", window_sec=1.0):
    window_samples = max(1, int(round(window_sec * fs)))
    train_blocks = set(np.asarray(train_blocks, dtype=int).tolist())

    X_parts = []
    y_parts = []

    for _, row in trials.iterrows():
        if int(row["block_id"]) not in train_blocks:
            continue

        label = int(row["label"])
        _, y_tmp = transform_labels(np.array([label]), task)
        if len(y_tmp) == 0:
            continue

        task_label = int(y_tmp[0])
        pos = int(row["start_idx"])
        end = int(row["end_idx"])

        while pos + window_samples <= end:
            X_parts.append(
                window_to_grouped_feature(
                    window=X[pos:pos + window_samples],
                    group_size=group_size,
                    agg=agg,
                )
            )
            y_parts.append(task_label)
            pos += window_samples

    if not X_parts:
        return (
            np.empty((0, n_features_after_grouping(33, group_size))),
            np.empty((0,), dtype=int),
        )

    return np.vstack(X_parts), np.asarray(y_parts, dtype=int)


def collect_test_collapsed(X, trials, test_blocks, task, fs, group_size=1, agg="mean", window_sec=1.0):
    X_test, y_test = collect_train_collapsed(
        X=X,
        trials=trials,
        train_blocks=test_blocks,
        task=task,
        fs=fs,
        group_size=group_size,
        agg=agg,
        window_sec=window_sec,
    )
    return X_test, y_test, np.empty((len(y_test),), dtype=int)


def run_lobo_collapsed(day, task, filter_mode, group_size=1, agg="mean", model_name="LDA", window_sec=1.0):
    fs = day["sampling_rate"]
    trials = day["trials"]
    X_filtered = preprocess_nirs(X=day["X"], fs=fs, mode=filter_mode)
    unique_blocks = np.sort(trials["block_id"].unique())

    rows = []

    for train_blocks in combinations(unique_blocks, 2):
        train_blocks = np.asarray(train_blocks, dtype=int)
        test_blocks = np.setdiff1d(unique_blocks, train_blocks)

        X_train, y_train = collect_train_collapsed(
            X=X_filtered,
            trials=trials,
            train_blocks=train_blocks,
            task=task,
            fs=fs,
            group_size=group_size,
            agg=agg,
            window_sec=window_sec,
        )

        X_test, y_test, _ = collect_test_collapsed(
            X=X_filtered,
            trials=trials,
            test_blocks=test_blocks,
            task=task,
            fs=fs,
            group_size=group_size,
            agg=agg,
            window_sec=window_sec,
        )

        base = {
            "file": day["file_name"],
            "day": day.get("day"),
            "subject": day.get("subject"),
            "task": task,
            "filter": filter_mode,
            "group_size": group_size,
            "agg": agg,
            "model": model_name,
            "window_sec": window_sec,
            "n_features": X_train.shape[1] if len(X_train) > 0 else None,
            "train_blocks": tuple(train_blocks.tolist()),
            "test_blocks": tuple(test_blocks.tolist()),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

        if len(y_train) == 0 or len(y_test) == 0:
            rows.append({**base, "score": np.nan, "status": "skipped", "error": None})
            continue

        try:
            clf = GaussianWindowDiscriminant(mode=model_name)
            clf.fit(X_train, y_train)
            y_pred = np.asarray([clf.predict_window(x.reshape(1, -1)) for x in X_test])
            P, _, _ = dissertation_score(y_test, y_pred)
            rows.append({**base, "score": P, "status": "ok", "error": None})
        except Exception as e:
            rows.append({**base, "score": np.nan, "status": "failed", "error": repr(e)})

    return pd.DataFrame(rows)


def run_experiment_grid(
    *, preprocessed, task_filters, classifiers, feature_set,
    window_sec, seed, n_permutations, description, days, channels_1based=None,
    components=False, include_channels=False, include_window=False,
):
    """Применить существующий run_one_permutation к сетке условий, не меняя его логику."""

    if include_channels and channels_1based is None:
        raise ValueError("channels_1based is required when include_channels=True")
    variants = preprocessed if components else {None: preprocessed}
    # Используем одну последовательность перестановок для всех условий.
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(days)) for _ in range(n_permutations)]
    total = len(variants) * len(classifiers) * sum(map(len, task_filters.values())) * n_permutations
    exp1, exp2 = [], []
    with tqdm(total=total, desc=description) as pbar:
        for component, prepared in variants.items():
            for task, filters in task_filters.items():
                for clf in classifiers:
                    for filter_mode in filters:
                        for perm_idx, perm_order in enumerate(permutations):
                            first, second = run_one_permutation(
                                perm_order=perm_order,
                                task=task,
                                clf_name=clf,
                                filter_mode=filter_mode,
                                preprocessed=prepared,
                                window_sec=window_sec,
                                feature_set=feature_set,
                            )
                            for frame in (first, second):
                                frame["perm"] = perm_idx
                                if include_window:
                                    frame["window_sec"] = window_sec
                                if component is not None:
                                    frame["hemo_component"] = component
                                    if include_channels:
                                        frame["channels_1based"] = str(channels_1based)
                            exp1.append(first)
                            exp2.append(second)
                            pbar.update(1)
    return pd.concat(exp1, ignore_index=True), pd.concat(exp2, ignore_index=True)
