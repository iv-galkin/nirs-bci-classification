import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_task_classes(task: str):
    if task in {"left_vs_right", "rest_vs_active"}:
        return np.array([0, 1], dtype=int)
    raise ValueError(f"Unknown task: {task}")


def dissertation_score(y_true, y_pred):
    """P = mean of diagonal of Q; rows predicted, columns true."""
    labels = np.unique(y_true)
    return dissertation_score_fixed_labels(y_true, y_pred, labels=labels)


def dissertation_score_fixed_labels(y_true, y_pred, labels):
    labels = np.asarray(labels, dtype=int)
    cm_sklearn = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm_sklearn.T
    col_sums = cm.sum(axis=0, keepdims=True)
    Q = np.divide(cm, np.maximum(col_sums, 1), out=np.zeros_like(cm, dtype=float), where=col_sums != 0)
    P = np.mean(np.diag(Q))
    return P, cm, Q


def aggregate_lobo_by_day(results: pd.DataFrame):
    results_ok = results[results["status"] == "ok"].copy()
    return (
        results_ok
        .groupby(["file", "day", "subject", "task", "model", "filter"], dropna=False)
        .agg(P=("score", "mean"), n_folds=("score", "size"))
        .reset_index()
    )


def summarize_median_iqr(aggregated: pd.DataFrame, filters, filter_labels=None, digits: int = 3):
    if filter_labels is None:
        filter_labels = {f: f for f in filters}
    rows = []
    for filter_mode in filters:
        values = aggregated.loc[aggregated["filter"] == filter_mode, "P"].dropna().astype(float).values
        if len(values) == 0:
            rows.append({
                "filter": filter_mode,
                "filter_label": filter_labels.get(filter_mode, filter_mode),
                "n": 0,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "P_median_iqr": "nan [nan; nan]",
            })
            continue
        median = np.median(values)
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        rows.append({
            "filter": filter_mode,
            "filter_label": filter_labels.get(filter_mode, filter_mode),
            "n": len(values),
            "median": median,
            "q25": q25,
            "q75": q75,
            "P_median_iqr": f"{median:.{digits}f} [{q25:.{digits}f}; {q75:.{digits}f}]",
        })
    return pd.DataFrame(rows)


def summarize_reduction(results: pd.DataFrame):
    ok = results[results["status"] == "ok"].copy()
    by_day = (
        ok.groupby(["day", "group_size", "agg", "n_features"], dropna=False)["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "P"})
    )
    summary = (
        by_day.groupby(["group_size", "agg", "n_features"], dropna=False)["P"]
        .agg(
            n="count",
            median="median",
            q25=lambda x: np.percentile(x, 25),
            q75=lambda x: np.percentile(x, 75),
        )
        .reset_index()
    )
    summary["P_str"] = summary.apply(lambda r: f"{r['median']:.3f} [{r['q25']:.3f}; {r['q75']:.3f}]", axis=1)
    return summary.sort_values("median", ascending=False)


def summarize_experiment_grid(results, group_cols, sort_cols):
    """Единая агрегация median / 25%-75% / train-test gap для всех вариантов."""
    valid = results.loc[results["status"].isin(["ok", "no_train_random"])].copy()
    valid["gap_train_minus_test"] = valid["P_train"] - valid["P_test"]
    percentile = lambda q: (lambda s: np.percentile(s.dropna(), q) if s.dropna().size else np.nan)
    summary = valid.groupby(group_cols, dropna=False).agg(
        P_train_median=("P_train", "median"),
        P_test_median=("P_test", "median"),
        P_test_q25=("P_test", percentile(25)),
        P_test_q75=("P_test", percentile(75)),
        gap_median=("gap_train_minus_test", "median"),
        gap_q25=("gap_train_minus_test", percentile(25)),
        gap_q75=("gap_train_minus_test", percentile(75)),
        n=("P_test", "size"),
    ).reset_index()
    for prefix, label in (("P_test", "P_test"), ("gap", "gap")):
        summary[f"{label}, median [25%; 75%]"] = summary.apply(
            lambda row: (
                f"{row[f'{prefix}_median']:.3f} "
                f"[{row[f'{prefix}_q25']:.3f}; {row[f'{prefix}_q75']:.3f}]"
                if pd.notna(row[f"{prefix}_median"]) else "nan [nan; nan]"
            ), axis=1,
        )
    return summary.sort_values(sort_cols).reset_index(drop=True)
