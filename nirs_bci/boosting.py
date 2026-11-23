from itertools import combinations
import numpy as np
import pandas as pd

from .evaluation import dissertation_score_fixed_labels, get_task_classes
from .features import (
    windows_to_features, make_feature_names, catboost_channel_importance,
    feature_correlations_with_label, channel_correlations,
)
from .models import make_catboost_model
from .preprocessing import preprocess_nirs
from .segmentation import collect_windows_from_blocks


def run_catboost_importance_one_fold(day, train_blocks, test_blocks, task, filter_mode, features, window_sec=1.0, random_seed=42):
    fs = day["sampling_rate"]
    trials = day["trials"]
    X_filtered = preprocess_nirs(day["X"], fs, filter_mode)
    n_channels = X_filtered.shape[1] // 2
    train_windows, y_train = collect_windows_from_blocks(X_filtered, trials, train_blocks, task, fs, window_sec)
    test_windows, y_test = collect_windows_from_blocks(X_filtered, trials, test_blocks, task, fs, window_sec)
    base = {
        "file": day["file_name"], "day": day.get("day"), "subject": day.get("subject"),
        "task": task, "filter": filter_mode, "features": "_".join(features), "window_sec": window_sec,
        "train_blocks": tuple(np.asarray(train_blocks, dtype=int).tolist()),
        "test_blocks": tuple(np.asarray(test_blocks, dtype=int).tolist()),
        "n_train_windows": len(y_train), "n_test_windows": len(y_test),
    }
    if len(y_train) == 0 or len(y_test) == 0:
        return pd.DataFrame(), {**base, "P_test": np.nan, "status": "skipped_empty_data", "error": None}

    X_train_feat = windows_to_features(train_windows, features)
    X_test_feat = windows_to_features(test_windows, features)
    feature_names = make_feature_names(n_channels, features)
    model = make_catboost_model(random_seed=random_seed, conservative=True)
    try:
        model.fit(X_train_feat, y_train)
        y_pred = model.predict(X_test_feat)
        P_test, _, _ = dissertation_score_fixed_labels(y_test, y_pred, labels=get_task_classes(task))
        channel_imp, _ = catboost_channel_importance(model, feature_names, n_channels)
        corr_features = feature_correlations_with_label(X_train_feat, y_train, feature_names)
        corr_channels = channel_correlations(corr_features, n_channels)
        out = channel_imp.merge(corr_channels, on="channel", how="left")
        for key, value in base.items():
            out[key] = [value] * len(out)
        out["P_test"] = P_test
        out["status"] = "ok"
        out["error"] = None
        return out, {**base, "P_test": P_test, "status": "ok", "error": None}
    except Exception as e:
        return pd.DataFrame(), {**base, "P_test": np.nan, "status": "failed", "error": repr(e)}


def run_catboost_importance_all_folds(days, task, filter_mode, features, window_sec=1.0, random_seed=42):
    importance_tables, fold_rows = [], []
    jobs = []
    for day_idx, day in enumerate(days):
        unique_blocks = np.sort(day["trials"]["block_id"].unique())
        if len(unique_blocks) != 3:
            raise ValueError(f"{day['file_name']}: expected 3 blocks, got {len(unique_blocks)}")
        for train_blocks in combinations(unique_blocks, 2):
            train_blocks = np.asarray(train_blocks, dtype=int)
            test_blocks = np.setdiff1d(unique_blocks, train_blocks)
            jobs.append((day_idx, day, train_blocks, test_blocks))

    for day_idx, day, train_blocks, test_blocks in jobs:
        imp_df, fold_row = run_catboost_importance_one_fold(day, train_blocks, test_blocks, task, filter_mode, features, window_sec, random_seed)
        fold_row["day_idx"] = day_idx
        fold_rows.append(fold_row)
        if not imp_df.empty:
            imp_df["day_idx"] = day_idx
            importance_tables.append(imp_df)

    importance_all = pd.concat(importance_tables, ignore_index=True) if importance_tables else pd.DataFrame()
    return importance_all, pd.DataFrame(fold_rows)


def aggregate_catboost_channel_importance(importance_all):
    ok = importance_all[importance_all["status"] == "ok"].copy()
    return (
        ok.groupby("channel")
        .agg(
            total_importance_median=("total_importance", "median"),
            total_importance_q25=("total_importance", lambda x: np.percentile(x, 25)),
            total_importance_q75=("total_importance", lambda x: np.percentile(x, 75)),
            HbO_importance_median=("HbO_importance", "median"),
            HbR_importance_median=("HbR_importance", "median"),
            total_abs_corr_median=("total_abs_corr_mean", "median"),
            total_abs_corr_q25=("total_abs_corr_mean", lambda x: np.percentile(x, 25)),
            total_abs_corr_q75=("total_abs_corr_mean", lambda x: np.percentile(x, 75)),
            HbO_abs_corr_median=("HbO_abs_corr_mean", "median"),
            HbR_abs_corr_median=("HbR_abs_corr_mean", "median"),
            n_folds=("total_importance", "size"),
            mean_P_test=("P_test", "mean"),
        )
        .reset_index()
        .sort_values("total_importance_median", ascending=False)
    )
