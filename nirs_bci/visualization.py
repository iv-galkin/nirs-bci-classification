import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from .preprocessing import preprocess_nirs, split_hbo_hbr


def plot_filter_boxplot(
    aggregated,
    filters,
    filter_labels=None,
    title="Classification",
    chance_level=0.5,
    figsize=(7, 5),
):
    if filter_labels is None:
        filter_labels = {f: f for f in filters}

    data = [
        aggregated.loc[aggregated["filter"] == f, "P"].dropna().astype(float).values
        for f in filters
    ]
    positions = np.arange(1, len(filters) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, positions=positions, widths=0.55, showmeans=False, patch_artist=False)

    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), pos) + jitter, values, s=25, alpha=0.8)

    ax.axhline(chance_level, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("P")
    ax.set_xticks(positions)
    ax.set_xticklabels([filter_labels.get(f, f) for f in filters], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_lda_qda_filter_boxplot(
    agg_lda,
    agg_qda,
    filters,
    filter_labels,
    title,
    chance_level=0.5,
    figsize=(11, 5),
):
    """One boxplot comparing LDA/QDA across filters."""
    lda = agg_lda.copy()
    lda["model"] = "LDA"

    qda = agg_qda.copy()
    qda["model"] = "QDA"

    aggregated = pd.concat([lda, qda], ignore_index=True)

    data = []
    labels = []

    for filter_mode in filters:
        for model in ["LDA", "QDA"]:
            values = (
                aggregated.loc[
                    (aggregated["filter"] == filter_mode) & (aggregated["model"] == model),
                    "P",
                ]
                .dropna()
                .astype(float)
                .values
            )
            data.append(values)
            labels.append(f"{model}\n{filter_labels.get(filter_mode, filter_mode)}")

    positions = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, positions=positions, widths=0.55, showmeans=False, patch_artist=False)

    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), pos) + jitter, values, s=25, alpha=0.8)

    ax.axhline(chance_level, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("P")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for k in range(1, len(filters)):
        ax.axvline(2 * k + 0.5, linestyle=":", linewidth=1, alpha=0.4)

    fig.tight_layout()
    return fig, ax


def plot_channel_selection_boxplot(
    aggregated,
    selection_methods,
    method_labels=None,
    title="Channel selection",
    chance_level=0.5,
    figsize=(9, 5),
):
    if method_labels is None:
        method_labels = {m: m for m in selection_methods}

    data = [
        aggregated.loc[aggregated["selection_method"] == method, "P"].dropna().astype(float).values
        for method in selection_methods
    ]
    positions = np.arange(1, len(selection_methods) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, positions=positions, widths=0.55, showmeans=False, patch_artist=False)

    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), pos) + jitter, values, s=25, alpha=0.8)

    ax.axhline(chance_level, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("P")
    ax.set_xticks(positions)
    ax.set_xticklabels([method_labels.get(m, m) for m in selection_methods], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_psd_hbo_hbr(day, filters, filter_labels=None, max_freq=0.1, nperseg=None):
    fs = day["sampling_rate"]
    X_raw = day["X"]

    if filter_labels is None:
        filter_labels = {f: f for f in filters}

    if nperseg is None:
        nperseg = min(4096, X_raw.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for mode in filters:
        X = preprocess_nirs(X_raw, fs, mode)
        X_hbo, X_hbr = split_hbo_hbr(X)

        for ax, X_part, component_name in [(axes[0], X_hbo, "HbO"), (axes[1], X_hbr, "HbR")]:
            freqs, psds = welch(
                X_part,
                fs=fs,
                axis=0,
                nperseg=min(nperseg, X_part.shape[0]),
            )
            median_psd = np.nanmedian(psds, axis=1)
            mask = freqs <= max_freq
            ax.semilogy(freqs[mask], median_psd[mask], linewidth=2, label=filter_labels.get(mode, mode))
            ax.axvline(0.002, linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(0.018, linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(component_name)
            ax.set_xlabel("Frequency, Hz")
            ax.grid(alpha=0.3)

    axes[0].set_ylabel("PSD")
    axes[1].legend()
    fig.suptitle(f"Low-frequency PSD: {day.get('file_name', '')}", fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_dissertation_style_with_summary(
    results_exp1,
    results_exp2,
    task,
    clf_name,
    filter_mode,
    window_sec=None,
    figsize=(10, 4),
):
    def get_curves(results, variant):
        ok = results[
            (results["task"] == task)
            & (results["clf"] == clf_name)
            & (results["filter"] == filter_mode)
            & (results["variant"] == variant)
            & (results["status"].isin(["ok", "no_train_random"]))
        ].copy()

        if len(ok) == 0:
            return pd.DataFrame(columns=["global_block_idx", "P_test", "P_train"])

        return (
            ok.groupby("global_block_idx", dropna=False)
            .agg(P_test=("P_test", "median"), P_train=("P_train", "median"))
            .reset_index()
            .sort_values("global_block_idx")
        )

    def get_summary(results, variant):
        ok = results[
            (results["task"] == task)
            & (results["clf"] == clf_name)
            & (results["filter"] == filter_mode)
            & (results["variant"] == variant)
            & (results["status"].isin(["ok", "no_train_random"]))
        ].copy()

        if len(ok) == 0:
            return {"P_train_median": np.nan, "P_test_median": np.nan, "gap_median": np.nan, "n": 0}

        return {
            "P_train_median": ok["P_train"].median(),
            "P_test_median": ok["P_test"].median(),
            "gap_median": (ok["P_train"] - ok["P_test"]).median(),
            "n": len(ok),
        }

    curve1 = get_curves(results_exp1, "exp1")
    curve2 = get_curves(results_exp2, "exp2")
    s1 = get_summary(results_exp1, "exp1")
    s2 = get_summary(results_exp2, "exp2")

    fig, ax = plt.subplots(figsize=figsize)

    if len(curve1) > 0:
        ax.plot(curve1["global_block_idx"], curve1["P_test"], marker="o", markersize=4, label="exp1 P_test")
        ax.plot(curve1["global_block_idx"], curve1["P_train"], marker="o", markersize=4, linestyle="--", alpha=0.7, label="exp1 P_train")

    if len(curve2) > 0:
        ax.plot(curve2["global_block_idx"], curve2["P_test"], marker="o", markersize=4, label="exp2 P_test")
        ax.plot(curve2["global_block_idx"], curve2["P_train"], marker="o", markersize=4, linestyle="--", alpha=0.7, label="exp2 P_train")

    ax.axhline(0.5, linestyle="--", linewidth=0.8, label="chance")

    title = f"{task} | {clf_name} | {filter_mode}"
    if window_sec is not None:
        title += f" | window={window_sec}s"

    ax.set_title(title)
    ax.set_xlabel("Block number")
    ax.set_ylabel("P")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    text = (
        f"exp1: P_train={s1['P_train_median']:.3f}, P_test={s1['P_test_median']:.3f}, "
        f"gap={s1['gap_median']:.3f}; n={s1['n']}\n"
        f"exp2: P_train={s2['P_train_median']:.3f}, P_test={s2['P_test_median']:.3f}, "
        f"gap={s2['gap_median']:.3f}; n={s2['n']}"
    )
    ax.text(0.01, 0.02, text, transform=ax.transAxes, fontsize=9, va="bottom", bbox=dict(boxstyle="round", alpha=0.15))

    fig.tight_layout()
    return fig, ax


def plot_ml_models_boxplot(
    results_all,
    task,
    filter_mode,
    classifiers,
    chance_level=0.5,
    figsize=(11, 5),
):
    """Boxplot comparing P_test for multiple classifiers in exp1/exp2."""
    df = results_all[
        (results_all["task"] == task)
        & (results_all["filter"] == filter_mode)
        & (results_all["clf"].isin(classifiers))
        & (results_all["status"].isin(["ok", "no_train_random"]))
    ].copy()

    data = []
    labels = []

    for variant in ["exp1", "exp2"]:
        for clf in classifiers:
            values = (
                df.loc[(df["variant"] == variant) & (df["clf"] == clf), "P_test"]
                .dropna()
                .astype(float)
                .values
            )
            data.append(values)
            labels.append(f"{variant}\n{clf}")

    positions = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, positions=positions, widths=0.55, showmeans=False, patch_artist=False)

    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), pos) + jitter, values, s=25, alpha=0.8)

    ax.axhline(chance_level, linestyle="--", linewidth=1)
    ax.set_title(f"{task}, filter={filter_mode}: model comparison")
    ax.set_ylabel("P_test")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_diss_component_boxplot(results_all, task, clf_name, variant="exp2", window_sec=5):
    df = results_all[
        (results_all["task"] == task)
        & (results_all["clf"] == clf_name)
        & (results_all["variant"] == variant)
        & (results_all["status"].isin(["ok", "no_train_random"]))
    ].copy()

    components = [
        "HbO_diss_channels",
        "HbR_diss_channels",
        "HbO_plus_HbR_diss_channels",
        "HbO_minus_HbR_diss_channels",
    ]

    data = [
        df.loc[df["hemo_component"] == c, "P_test"].dropna().astype(float).values
        for c in components
    ]

    positions = np.arange(1, len(components) + 1)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showmeans=False,
        patch_artist=False,
    )

    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue

        jitter = (
            np.linspace(-0.08, 0.08, len(values))
            if len(values) > 1
            else np.array([0.0])
        )

        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=22,
            alpha=0.75,
        )

    ax.axhline(0.5, linestyle="--", linewidth=1, label="chance")

    ax.set_xticks(positions)
    ax.set_xticklabels(components, rotation=25, ha="right")
    ax.set_ylabel("P_test")
    ax.set_title(f"{task} | {clf_name} | {variant} | diss channels | SMA | window={window_sec:g}s")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_lda_catboost_component_boxplot(results_all_lda, results_all_cb, task, hemo_component, variant="exp2"):
    df = pd.concat(
        [
            results_all_lda,
            results_all_cb,
        ],
        ignore_index=True,
    )

    df = df[
        (df["task"] == task)
        & (df["hemo_component"] == hemo_component)
        & (df["variant"] == variant)
        & (df["status"].isin(["ok", "no_train_random"]))
    ].copy()

    classifiers = [
        "LDA",
        "CatBoost_mean",
        "CatBoost_all",
    ]

    data = [
        df.loc[df["clf"] == clf, "P_test"].dropna().astype(float).values
        for clf in classifiers
    ]

    positions = np.arange(1, len(classifiers) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showmeans=False,
        patch_artist=False,
    )

    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue

        jitter = (
            np.linspace(-0.08, 0.08, len(values))
            if len(values) > 1
            else np.array([0.0])
        )

        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=22,
            alpha=0.75,
        )

    ax.axhline(0.5, linestyle="--", linewidth=1, label="chance")

    ax.set_xticks(positions)
    ax.set_xticklabels(classifiers, rotation=20, ha="right")
    ax.set_ylabel("P_test")
    ax.set_title(f"{task} | {hemo_component} | {variant} | SMA | window=1s")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_quality_by_block(
    results_all,
    task,
    hemo_component,
    variant="exp2",
    metric="P_test",
    classifiers=None,
    title=None,
    chance_level=0.5,
    figsize=(10, 5),
):
    df = results_all[
        (results_all["task"] == task)
        & (results_all["hemo_component"] == hemo_component)
        & (results_all["variant"] == variant)
        & (results_all["status"].isin(["ok", "no_train_random"]))
    ].copy()

    if classifiers is None:
        classifiers = [
            "LDA",
            "CatBoost_mean",
            "CatBoostConservative_mean",
            "CatBoostTiny_mean",
        ]

    df = df[df["clf"].isin(classifiers)].copy()

    # агрегируем по номеру блока и модели
    curve = (
        df
        .groupby(["clf", "global_block_idx"], dropna=False)
        .agg(
            median_metric=(metric, "median"),
            q25_metric=(metric, lambda x: np.percentile(x.dropna(), 25) if x.dropna().size > 0 else np.nan),
            q75_metric=(metric, lambda x: np.percentile(x.dropna(), 75) if x.dropna().size > 0 else np.nan),
            n=(metric, "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    for clf_name in classifiers:
        cur = curve[curve["clf"] == clf_name].sort_values("global_block_idx")

        if len(cur) == 0:
            continue

        x = cur["global_block_idx"].values
        y = cur["median_metric"].values
        y25 = cur["q25_metric"].values
        y75 = cur["q75_metric"].values

        ax.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=clf_name,
        )

        ax.fill_between(
            x,
            y25,
            y75,
            alpha=0.15,
        )

    ax.axhline(
        chance_level,
        linestyle="--",
        linewidth=1,
        label="chance",
    )

    ax.set_xlabel("Global block index")
    ax.set_ylabel(metric)

    if title is None:
        title = f"{task} | {hemo_component} | {variant} | {metric}"

    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax, curve


def plot_author_boxplot_like_matlab(
    df, subject_col="Subject", recall_col="RecallValue",
    chance_level=1/3, jitter_amount=0.15, figsize=(14, 6), title=None,
):
    """Python-аналог авторской MATLAB-функции PlotBoxplots."""
    plot_df = df[[subject_col, recall_col]].dropna().copy()
    plot_df[subject_col] = plot_df[subject_col].astype(int)
    subjects = sorted(plot_df[subject_col].unique())
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=figsize)
    for subject in subjects:
        values = plot_df.loc[plot_df[subject_col] == subject, recall_col].astype(float).to_numpy()
        jitter = rng.uniform(-jitter_amount, jitter_amount, size=len(values))
        ax.scatter(np.full(len(values), subject) + jitter, values, s=30, alpha=0.6, zorder=3)
    boxplot_data = [
        plot_df.loc[plot_df[subject_col] == subject, recall_col].astype(float).to_numpy()
        for subject in subjects
    ]
    ax.boxplot(
        boxplot_data, positions=subjects, widths=0.55, patch_artist=False,
        showfliers=True, medianprops={"linewidth": 2}, zorder=2,
    )
    ax.plot([0, len(subjects) + 1], [chance_level, chance_level], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Subject id")
    ax.set_ylabel("Recall")
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(subjects)
    ax.set_xlim(0, len(subjects) + 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax
