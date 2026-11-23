"""Utilities for the NITRC stroke-patient NIRS dataset.

This module implements an online-like reconstruction of the patient-series
baseline described in the Scientific Data paper and clarified by M. R. Isaev:

* labels: 1 = rest, 2 = left imagery, 3 = right imagery, 0 = not classified;
* two-stage LDA: rest vs active, then left vs right;
* features: filtered HbO + HbR for all channels;
* epoch decision: product of per-sample class likelihoods, implemented by the
  GaussianWindowDiscriminant score over the whole window;
* train history: previous blocks of current session and/or previous sessions;
* evaluation: all blocks except the first block of a patient.

Important: exact online filtering and exact custom shrinkage from the original
system are not fully available, so this module is an offline/online-like
reconstruction rather than a byte-identical reproduction of the MATLAB system.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import cheby1, sosfilt, sosfiltfilt

from .evaluation import dissertation_score_fixed_labels
from .models import GaussianWindowDiscriminant

TrainingMode = Literal[
    "previous_blocks_and_sessions",
    "current_session_only",
    "previous_sessions_only",
]
FilterMode = Literal["causal", "zero_phase"]
EpochLabelRule = Literal["strict", "center", "majority"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_nitrc_concentration_name(path: str | Path) -> tuple[int, int, int]:
    """Parse Subject1_day1_session1_concentrations.mat."""
    path = Path(path)
    pattern = r"Subject(\d+)_day(\d+)_session(\d+)_concentrations\.mat"
    match = re.match(pattern, path.name)
    if match is None:
        raise ValueError(f"Cannot parse NITRC concentration filename: {path.name}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def build_nitrc_file_table(data_dir: str | Path) -> pd.DataFrame:
    """Return sorted table of all *_concentrations.mat files."""
    data_dir = Path(data_dir)
    rows = []
    for path in sorted(data_dir.glob("*_concentrations.mat")):
        subject, day, session = parse_nitrc_concentration_name(path)
        rows.append(
            {
                "path": path,
                "file": path.name,
                "subject": subject,
                "day": day,
                "session": session,
            }
        )
    return pd.DataFrame(rows).sort_values(["subject", "day", "session"]).reset_index(drop=True)


def load_nitrc_concentration_session(path: str | Path) -> dict:
    """Load one NITRC MATLAB_Concentrations file.

    Returned X has feature order [HbO_1..HbO_C, HbR_1..HbR_C].
    """
    path = Path(path)
    data = loadmat(path, squeeze_me=True, struct_as_record=False)

    hbo = np.asarray(data["HbO"], dtype=float)
    hbr = np.asarray(data["HbR"], dtype=float)
    labels = np.asarray(data["Tasks_labels"]).astype(int).ravel()
    time = np.asarray(data["Time"], dtype=float).ravel()
    fs = float(np.asarray(data["Frequency"]).ravel()[0])
    tasks = np.asarray(data.get("Tasks", []))

    if hbo.shape != hbr.shape:
        raise ValueError(f"HbO and HbR shapes differ: {hbo.shape} vs {hbr.shape}")
    if hbo.shape[0] != labels.shape[0] or hbo.shape[0] != time.shape[0]:
        raise ValueError("Time dimension mismatch between HbO/HbR, labels, and time")

    subject, day, session = parse_nitrc_concentration_name(path)
    X = np.concatenate([hbo, hbr], axis=1)

    return {
        "file": str(path),
        "file_name": path.name,
        "subject": subject,
        "day": day,
        "session": session,
        "HbO": hbo,
        "HbR": hbr,
        "X": X,
        "labels": labels,
        "time": time,
        "sampling_rate": fs,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Article filters
# ---------------------------------------------------------------------------

def _stage1_sos(fs: float):
    """Rest vs active: 2nd-order Chebyshev-I band-pass 0.022--0.039 Hz."""
    return cheby1(N=2, rp=1, Wn=[0.022, 0.039], btype="bandpass", fs=fs, output="sos")


def _stage2_sos(fs: float):
    """Left vs right: 1st-order Chebyshev-I high-pass 0.005 Hz."""
    return cheby1(N=1, rp=1, Wn=0.005, btype="highpass", fs=fs, output="sos")


def apply_article_filters(hbo, hbr, fs: float, mode: FilterMode = "causal") -> tuple[np.ndarray, np.ndarray]:
    """Return X_stage1 and X_stage2, each time x (2*n_channels).

    mode="causal" is closer to online filtering; mode="zero_phase" is useful as
    an offline diagnostic but is not the article's online baseline.
    """
    hbo = np.asarray(hbo, dtype=float)
    hbr = np.asarray(hbr, dtype=float)

    sos1 = _stage1_sos(fs)
    sos2 = _stage2_sos(fs)

    filt = sosfilt if mode == "causal" else sosfiltfilt
    hbo1 = filt(sos1, hbo, axis=0)
    hbr1 = filt(sos1, hbr, axis=0)
    hbo2 = filt(sos2, hbo, axis=0)
    hbr2 = filt(sos2, hbr, axis=0)

    return np.concatenate([hbo1, hbr1], axis=1), np.concatenate([hbo2, hbr2], axis=1)


# ---------------------------------------------------------------------------
# Blocks and epochs
# ---------------------------------------------------------------------------

def assign_nitrc_blocks(labels) -> np.ndarray:
    """Assign block id to non-zero samples.

    In the NITRC protocol, one block contains 4 active imagery trials:
    2 left and 2 right in random order. Rest segments preceding active imagery
    are assigned to the same current block. Label 0 receives block id -1.
    """
    labels = np.asarray(labels).astype(int).ravel()
    block_ids = np.full(labels.shape, -1, dtype=int)

    if len(labels) == 0:
        return block_ids

    starts = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            starts.append(i)
    starts = np.asarray(starts, dtype=int)
    ends = np.r_[starts[1:], len(labels)]

    current_block = 0
    active_in_block = 0

    for start, end in zip(starts, ends):
        label = int(labels[start])
        if label == 0:
            continue

        block_ids[start:end] = current_block
        if label in {2, 3}:
            active_in_block += 1
            if active_in_block == 4:
                current_block += 1
                active_in_block = 0

    return block_ids


def make_nitrc_epochs(
    labels,
    block_ids,
    fs: float,
    *,
    epoch_sec: float = 1.0,
    step_sec: float = 0.25,
    label_rule: EpochLabelRule = "strict",
) -> pd.DataFrame:
    """Build classification epochs by samples.

    Default corresponds to the paper: 1 s epochs with 250 ms shift. For fs=15.625
    this gives 16 samples and 4-sample step after rounding.

    label_rule:
        strict   - all samples in the epoch must have the same non-zero label
                   and same block id;
        center   - label/block are taken from the central sample; zero center is
                   skipped;
        majority - majority non-zero label is used if all non-zero samples are
                   from one block.
    """
    labels = np.asarray(labels).astype(int).ravel()
    block_ids = np.asarray(block_ids).astype(int).ravel()
    window = max(1, int(round(epoch_sec * fs)))
    step = max(1, int(round(step_sec * fs)))

    rows = []
    for start in range(0, len(labels) - window + 1, step):
        end = start + window
        lab_win = labels[start:end]
        blk_win = block_ids[start:end]

        if label_rule == "strict":
            unique_labels = np.unique(lab_win)
            unique_blocks = np.unique(blk_win)
            if len(unique_labels) != 1 or unique_labels[0] == 0:
                continue
            if len(unique_blocks) != 1 or unique_blocks[0] < 0:
                continue
            label = int(unique_labels[0])
            block = int(unique_blocks[0])

        elif label_rule == "center":
            center = start + window // 2
            label = int(labels[center])
            block = int(block_ids[center])
            if label == 0 or block < 0:
                continue

        elif label_rule == "majority":
            nonzero = lab_win != 0
            if not np.any(nonzero):
                continue
            blocks = np.unique(blk_win[nonzero])
            if len(blocks) != 1 or blocks[0] < 0:
                continue
            vals, counts = np.unique(lab_win[nonzero], return_counts=True)
            label = int(vals[np.argmax(counts)])
            block = int(blocks[0])

        else:
            raise ValueError(f"Unknown label_rule: {label_rule}")

        rows.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "label": label,
                "block": block,
                "window_samples": int(window),
                "step_samples": int(step),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Two-stage classifier
# ---------------------------------------------------------------------------

def _fit_two_stage_lda(
    X1_train,
    X2_train,
    y_train,
    covariance_reg: float = 0.0,
    covariance_shrinkage: str = "none",
    shrinkage_alpha: float = 0.0,
):
    y_train = np.asarray(y_train).astype(int).ravel()

    mask_nonzero = y_train != 0
    X1 = X1_train[mask_nonzero]
    y_raw = y_train[mask_nonzero]

    # Stage 1: rest vs active
    # 0 = rest, 1 = active
    y_stage1 = (y_raw != 1).astype(int)

    if len(np.unique(y_stage1)) < 2:
        raise ValueError("Stage 1 needs both rest and active samples")

    clf1 = GaussianWindowDiscriminant(
        mode="LDA",
        covariance_reg=covariance_reg,
        shrinkage_method=covariance_shrinkage,
        shrinkage_alpha=shrinkage_alpha,
    )
    clf1.fit(X1, y_stage1)

    # Stage 2: left vs right
    active_mask = np.isin(y_train, [2, 3])
    y_stage2 = y_train[active_mask]

    if len(np.unique(y_stage2)) < 2:
        raise ValueError("Stage 2 needs both left and right samples")

    clf2 = GaussianWindowDiscriminant(
        mode="LDA",
        covariance_reg=covariance_reg,
        shrinkage_method=covariance_shrinkage,
        shrinkage_alpha=shrinkage_alpha,
    )
    clf2.fit(X2_train[active_mask], y_stage2)

    return clf1, clf2


def _predict_two_stage_epoch(clf1, clf2, X1_epoch, X2_epoch) -> int:
    pred1 = int(clf1.predict_window(X1_epoch))
    if pred1 == 0:
        return 1
    return int(clf2.predict_window(X2_epoch))


def prepare_nitrc_session(
    path: str | Path,
    *,
    filter_mode: FilterMode = "causal",
    epoch_sec: float = 1.0,
    step_sec: float = 0.25,
    label_rule: EpochLabelRule = "strict",
) -> dict:
    sess = load_nitrc_concentration_session(path)
    X1, X2 = apply_article_filters(sess["HbO"], sess["HbR"], sess["sampling_rate"], mode=filter_mode)
    block_ids = assign_nitrc_blocks(sess["labels"])
    epochs = make_nitrc_epochs(
        sess["labels"],
        block_ids,
        sess["sampling_rate"],
        epoch_sec=epoch_sec,
        step_sec=step_sec,
        label_rule=label_rule,
    )
    sess.update({"X_stage1": X1, "X_stage2": X2, "block_ids": block_ids, "epochs": epochs})
    return sess


def evaluate_nitrc_subject_online_like(
    subject_files: pd.DataFrame,
    *,
    training_mode: TrainingMode = "previous_blocks_and_sessions",
    filter_mode: FilterMode = "causal",
    epoch_sec: float = 1.0,
    step_sec: float = 0.25,
    label_rule: EpochLabelRule = "strict",
    covariance_reg: float = 0.0,
    covariance_shrinkage: str = "none",
    shrinkage_alpha: float = 0.0,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one subject in an online-like chronological protocol.

    training_mode:
        previous_blocks_and_sessions - paper baseline-like rule;
        current_session_only         - ablation without previous sessions;
        previous_sessions_only       - cross-session transfer ablation.

    covariance_reg:
        Ridge-style diagonal loading.

    covariance_shrinkage:
        "none", "diagonal", or "spherical".

    shrinkage_alpha:
        Shrinkage intensity in [0, 1].
    """
    subject_files = subject_files.sort_values(["day", "session"]).reset_index(drop=True)

    past_X1: list[np.ndarray] = []
    past_X2: list[np.ndarray] = []
    past_y: list[np.ndarray] = []

    epoch_rows = []
    block_rows = []
    global_block_idx = 0

    for _, row in subject_files.iterrows():
        sess = prepare_nitrc_session(
            row["path"],
            filter_mode=filter_mode,
            epoch_sec=epoch_sec,
            step_sec=step_sec,
            label_rule=label_rule,
        )

        X1 = sess["X_stage1"]
        X2 = sess["X_stage2"]
        y = sess["labels"]
        block_ids = sess["block_ids"]
        epochs = sess["epochs"]

        unique_blocks = sorted(epochs["block"].unique()) if len(epochs) else []

        for block in unique_blocks:
            test_epochs = epochs[epochs["block"] == block]

            # Isaev clarification: all blocks except the first block of a patient
            # were classified. This also avoids zero-history training.
            if global_block_idx == 0:
                status = "skipped_first_patient_block"
                y_true_block = []
                y_pred_block = []

            else:
                X1_parts = []
                X2_parts = []
                y_parts = []

                if (
                    training_mode in {"previous_blocks_and_sessions", "previous_sessions_only"}
                    and past_y
                ):
                    X1_parts.append(np.vstack(past_X1))
                    X2_parts.append(np.vstack(past_X2))
                    y_parts.append(np.concatenate(past_y))

                if training_mode in {"previous_blocks_and_sessions", "current_session_only"}:
                    current_prev = (block_ids >= 0) & (block_ids < block) & (y != 0)

                    if np.any(current_prev):
                        X1_parts.append(X1[current_prev])
                        X2_parts.append(X2[current_prev])
                        y_parts.append(y[current_prev])

                if not y_parts:
                    status = "skipped_no_train"
                    y_true_block = []
                    y_pred_block = []

                else:
                    X1_train = np.vstack(X1_parts)
                    X2_train = np.vstack(X2_parts)
                    y_train = np.concatenate(y_parts)

                    y_true_block = []
                    y_pred_block = []

                    try:
                        clf1, clf2 = _fit_two_stage_lda(
                            X1_train=X1_train,
                            X2_train=X2_train,
                            y_train=y_train,
                            covariance_reg=covariance_reg,
                            covariance_shrinkage=covariance_shrinkage,
                            shrinkage_alpha=shrinkage_alpha,
                        )

                        for _, ep in test_epochs.iterrows():
                            start = int(ep["start_idx"])
                            end = int(ep["end_idx"])
                            y_true = int(ep["label"])

                            y_pred = _predict_two_stage_epoch(
                                clf1,
                                clf2,
                                X1[start:end],
                                X2[start:end],
                            )

                            y_true_block.append(y_true)
                            y_pred_block.append(y_pred)

                            epoch_rows.append(
                                {
                                    "subject": sess["subject"],
                                    "day": sess["day"],
                                    "session": sess["session"],
                                    "file": sess["file_name"],
                                    "block": int(block),
                                    "global_block_idx": int(global_block_idx),
                                    "training_mode": training_mode,
                                    "filter_mode": filter_mode,
                                    "label_rule": label_rule,
                                    "covariance_reg": covariance_reg,
                                    "covariance_shrinkage": covariance_shrinkage,
                                    "shrinkage_alpha": shrinkage_alpha,
                                    "start_idx": start,
                                    "end_idx": end,
                                    "y_true": y_true,
                                    "y_pred": y_pred,
                                }
                            )

                        status = "ok"

                    except Exception as exc:
                        status = "error"
                        if verbose:
                            warnings.warn(
                                f"{sess['file_name']} block={block}: {exc!r}"
                            )

            if y_true_block:
                P, cm, Q = dissertation_score_fixed_labels(
                    y_true_block,
                    y_pred_block,
                    labels=[1, 2, 3],
                )
            else:
                P, cm, Q = np.nan, None, None

            block_rows.append(
                {
                    "subject": sess["subject"],
                    "day": sess["day"],
                    "session": sess["session"],
                    "file": sess["file_name"],
                    "block": int(block),
                    "global_block_idx": int(global_block_idx),
                    "training_mode": training_mode,
                    "filter_mode": filter_mode,
                    "label_rule": label_rule,
                    "covariance_reg": covariance_reg,
                    "covariance_shrinkage": covariance_shrinkage,
                    "shrinkage_alpha": shrinkage_alpha,
                    "n_epochs": len(y_true_block),
                    "recall_macro": P,
                    "confusion_matrix": cm.tolist() if cm is not None else None,
                    "Q": Q.tolist() if Q is not None else None,
                    "status": status,
                }
            )

            global_block_idx += 1

        # After each session, it becomes available for future sessions.
        nonzero = y != 0
        past_X1.append(X1[nonzero])
        past_X2.append(X2[nonzero])
        past_y.append(y[nonzero])

    return pd.DataFrame(epoch_rows), pd.DataFrame(block_rows)


def evaluate_nitrc_all_subjects(
    file_table: pd.DataFrame,
    *,
    training_modes: list[TrainingMode] | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run evaluate_nitrc_subject_online_like for every subject."""
    if training_modes is None:
        training_modes = ["previous_blocks_and_sessions"]

    all_epochs = []
    all_blocks = []
    for mode in training_modes:
        for subject in sorted(file_table["subject"].unique()):
            subject_files = file_table[file_table["subject"] == subject].copy()
            ep, bl = evaluate_nitrc_subject_online_like(
                subject_files,
                training_mode=mode,
                **kwargs,
            )
            if len(ep):
                all_epochs.append(ep)
            all_blocks.append(bl)

    epochs = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()
    blocks = pd.concat(all_blocks, ignore_index=True) if all_blocks else pd.DataFrame()
    return epochs, blocks


def summarize_online_like_blocks(block_results: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR of block-level recalls by training mode and preprocessing settings."""
    ok = block_results[
        block_results["status"] == "ok"
    ].dropna(subset=["recall_macro"]).copy()

    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "training_mode",
        "filter_mode",
        "label_rule",
        "covariance_reg",
    ]

    if "covariance_shrinkage" in ok.columns:
        group_cols.append("covariance_shrinkage")

    if "shrinkage_alpha" in ok.columns:
        group_cols.append("shrinkage_alpha")

    rows = []

    for key, g in ok.groupby(group_cols, dropna=False):
        values = g["recall_macro"].astype(float).values

        rows.append(
            dict(
                zip(group_cols, key),
                n_blocks=len(values),
                median=float(np.median(values)),
                q25=float(np.percentile(values, 25)),
                q75=float(np.percentile(values, 75)),
            )
        )

    out = pd.DataFrame(rows)

    out["recall_str"] = out.apply(
        lambda r: f"{r['median']:.4f} [{r['q25']:.4f}; {r['q75']:.4f}]",
        axis=1,
    )

    return out
