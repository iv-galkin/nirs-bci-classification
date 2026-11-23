import numpy as np
import pandas as pd


def window_to_stat_features(window, features):
    """Convert one window (samples x features) to feature vector."""
    window = np.asarray(window, dtype=float)
    parts = []

    if "mean" in features:
        parts.append(window.mean(axis=0))
    if "std" in features:
        parts.append(window.std(axis=0))
    if "slope" in features:
        t = np.arange(window.shape[0], dtype=float)
        t = t - t.mean()
        denom = np.sum(t**2)
        if denom == 0:
            slope = np.zeros(window.shape[1])
        else:
            slope = (t[:, None] * (window - window.mean(axis=0))).sum(axis=0) / denom
        parts.append(slope)
    if "amplitude" in features:
        parts.append(window.max(axis=0) - window.min(axis=0))

    if not parts:
        raise ValueError("features must contain at least one statistic")
    return np.concatenate(parts)


def windows_to_features(windows, features):
    if len(windows) == 0:
        return np.empty((0, 0))
    return np.vstack([window_to_stat_features(w, features) for w in windows])


def make_feature_names(n_channels: int, features):
    """Feature names for X order [HbO all channels | HbR all channels]."""
    base_names = [f"HbO_ch{ch}" for ch in range(n_channels)] + [f"HbR_ch{ch}" for ch in range(n_channels)]
    return [f"{feat}_{base}" for feat in features for base in base_names]


def catboost_channel_importance(model, feature_names, n_channels: int):
    importances = model.get_feature_importance(type="PredictionValuesChange")
    if len(importances) != len(feature_names):
        raise ValueError(f"importance length {len(importances)} != feature_names length {len(feature_names)}")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    rows = []
    for ch in range(n_channels):
        hbo_mask = df["feature"].str.contains(fr"HbO_ch{ch}$", regex=True)
        hbr_mask = df["feature"].str.contains(fr"HbR_ch{ch}$", regex=True)
        hbo_imp = df.loc[hbo_mask, "importance"].sum()
        hbr_imp = df.loc[hbr_mask, "importance"].sum()
        rows.append({
            "channel": ch,
            "HbO_importance": hbo_imp,
            "HbR_importance": hbr_imp,
            "total_importance": hbo_imp + hbr_imp,
        })
    return pd.DataFrame(rows).sort_values("total_importance", ascending=False), df.sort_values("importance", ascending=False)


def feature_correlations_with_label(X_feat, y, feature_names):
    X_feat = np.asarray(X_feat, dtype=float)
    y = np.asarray(y, dtype=float)
    rows = []
    for j, name in enumerate(feature_names):
        x = X_feat[:, j]
        if np.std(x) == 0 or np.std(y) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(x, y)[0, 1]
        rows.append({"feature": name, "corr": corr, "abs_corr": abs(corr) if not np.isnan(corr) else np.nan})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def channel_correlations(corr_df: pd.DataFrame, n_channels: int):
    rows = []
    for ch in range(n_channels):
        hbo_mask = corr_df["feature"].str.contains(fr"HbO_ch{ch}$", regex=True)
        hbr_mask = corr_df["feature"].str.contains(fr"HbR_ch{ch}$", regex=True)
        hbo_corr = corr_df.loc[hbo_mask, "abs_corr"].mean()
        hbr_corr = corr_df.loc[hbr_mask, "abs_corr"].mean()
        rows.append({
            "channel": ch,
            "HbO_abs_corr_mean": hbo_corr,
            "HbR_abs_corr_mean": hbr_corr,
            "total_abs_corr_mean": np.nanmean([hbo_corr, hbr_corr]),
        })
    return pd.DataFrame(rows).sort_values("total_abs_corr_mean", ascending=False).reset_index(drop=True)


def collapse_window(window, agg: str = "mean"):
    if agg == "mean":
        return np.asarray(window).mean(axis=0)
    if agg == "median":
        return np.median(window, axis=0)
    raise ValueError(f"Unknown aggregation: {agg}")


def group_channels(x, group_size: int, agg: str = "mean"):
    """Group HbO and HbR channels separately in feature vector [HbO | HbR]."""
    x = np.asarray(x, dtype=float)
    n_ch = len(x) // 2
    x_hbo = x[:n_ch]
    x_hbr = x[n_ch:]

    def pool(arr):
        parts = []
        for i in range(0, len(arr), group_size):
            chunk = arr[i:i + group_size]
            parts.append(chunk.mean() if agg == "mean" else np.median(chunk))
        return np.array(parts)

    return np.concatenate([pool(x_hbo), pool(x_hbr)])


def n_features_after_grouping(n_ch: int = 33, group_size: int = 1):
    import math
    return math.ceil(n_ch / group_size) * 2


def window_to_grouped_feature(window, group_size: int = 1, agg: str = "mean"):
    x = collapse_window(window, agg="mean")
    if group_size > 1:
        x = group_channels(x, group_size, agg=agg)
    return x


def select_hemo_component_with_channels(X, selected_channels, component="hbo"):
    """Выбрать HbO, HbR или их сумму/разность для заданных каналов (0-based)."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] % 2:
        raise ValueError("Expected X = [HbO, HbR], i.e. even number of features")
    n_channels = X.shape[1] // 2
    selected_channels = np.asarray(selected_channels, dtype=int)
    if selected_channels.min() < 0 or selected_channels.max() >= n_channels:
        raise ValueError(f"selected_channels out of range: {selected_channels.tolist()}")

    hbo = X[:, :n_channels][:, selected_channels]
    hbr = X[:, n_channels:][:, selected_channels]
    if component == "hbo":
        return hbo
    if component == "hbr":
        return hbr
    if component == "hbo_plus_hbr":
        return hbo + hbr
    if component == "hbo_minus_hbr":
        return hbo - hbr
    raise ValueError(f"Unknown component: {component}")


def make_diss_channel_component_preprocessed(preprocessed, selected_channels, component):
    return {
        filter_mode: [
            (day, select_hemo_component_with_channels(X, selected_channels, component))
            for day, X in day_x_list
        ]
        for filter_mode, day_x_list in preprocessed.items()
    }
