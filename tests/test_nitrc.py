import numpy as np
import pandas as pd
import pytest

from nirs_bci import nitrc


def test_blocks_after_four_active_runs():
    labels = [0, 1, 2, 2, 1, 3, 1, 3, 1, 2, 0, 1, 3]
    assert nitrc.assign_nitrc_blocks(labels).tolist() == [
        -1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        1,
        1,
    ]


@pytest.mark.parametrize(
    "rule,expected",
    [
        ("strict", [(3, 3, 1)]),
        ("center", [(0, 2, 0), (1, 2, 0), (2, 3, 1), (3, 3, 1)]),
        ("majority", [(0, 2, 0), (3, 3, 1)]),
    ],
)
def test_epoch_stride_boundaries_and_labels(rule, expected):
    epochs = nitrc.make_nitrc_epochs(
        [0, 2, 2, 3, 3, 3], [-1, 0, 0, 1, 1, 1], fs=3, step_sec=1 / 3, label_rule=rule
    )
    assert (
        list(epochs[["start_idx", "label", "block"]].itertuples(index=False, name=None))
        == expected
    )
    assert (epochs.end_idx - epochs.start_idx == 3).all()


def test_two_stage_signal_and_labels():
    y = np.repeat([0, 1, 2, 3], 3)
    noise = np.tile([-1, 0, 1], 4)
    x1 = (np.repeat([999, -10, 10, 10], 3) + noise)[:, None]
    x2 = (np.repeat([999, 999, -5, 5], 3) + noise)[:, None]
    models = nitrc._fit_two_stage_lda(x1, x2, y)
    for a, b, label in [(-10, 5, 1), (10, -5, 2), (10, 5, 3)]:
        assert nitrc._predict_two_stage_epoch(*models, [[a]], [[b]]) == label


@pytest.mark.parametrize(
    "mode,histories,blocks",
    [
        ("previous_blocks_and_sessions", [[0], [0, 1], [0, 1, 10]], [1, 2, 3]),
        ("current_session_only", [[0], [10]], [1, 3]),
        ("previous_sessions_only", [[0, 1], [0, 1]], [2, 3]),
    ],
)
def test_chronological_training(monkeypatch, mode, histories, blocks):
    def session(day, **kwargs):
        x = np.array([0, 1])[:, None] + 10 * (day - 1)
        epochs = pd.DataFrame(
            dict(block=[0, 1], label=[2, 3], start_idx=[0, 1], end_idx=[1, 2])
        )
        return dict(
            X_stage1=x,
            X_stage2=x + 100,
            labels=np.array([2, 3]),
            epochs=epochs,
            block_ids=np.array([0, 1]),
            subject=1,
            day=day,
            session=1,
            file_name=str(day),
        )

    observed = []

    def fit(X1_train, X2_train, y_train, **kwargs):
        observed.append(X1_train[:, 0].tolist())
        np.testing.assert_array_equal(X2_train, X1_train + 100)
        np.testing.assert_array_equal(y_train, 2 + X1_train[:, 0] % 2)
        return None, None

    monkeypatch.setattr(nitrc, "prepare_nitrc_session", session)
    monkeypatch.setattr(nitrc, "_fit_two_stage_lda", fit)
    monkeypatch.setattr(
        nitrc, "_predict_two_stage_epoch", lambda a, b, x, z: 2 + x[0, 0] % 2
    )
    files = pd.DataFrame(dict(path=[2, 1], day=[2, 1], session=[1, 1]))
    ep, bl = nitrc.evaluate_nitrc_subject_online_like(files, training_mode=mode)
    assert observed == histories
    assert ep.global_block_idx.tolist() == blocks
    assert bl.status.iloc[0] == "skipped_first_patient_block"
    assert bl.loc[bl.status == "ok", "global_block_idx"].tolist() == blocks
    np.testing.assert_allclose(bl.loc[bl.status == "ok", "recall_macro"], 1 / 3)


def test_summary_counts_only_scored_blocks():
    bl = pd.DataFrame(
        dict(
            status=["ok", "error", "ok"],
            recall_macro=[0.25, 1, 0.75],
            training_mode="m",
            filter_mode="causal",
            label_rule="strict",
            covariance_reg=0,
        )
    )
    row = nitrc.summarize_online_like_blocks(bl).iloc[0]
    assert (row.n_blocks, row.q25, row["median"], row.q75) == (2, 0.375, 0.5, 0.625)


@pytest.mark.parametrize("mode", ["causal", "zero_phase"])
def test_article_filter_bands(mode):
    fs = 1
    t = np.arange(4000)
    tones = np.sin(2 * np.pi * t[:, None] * np.array([0.001, 0.03, 0.2]))
    band, high = nitrc.apply_article_filters(tones, -tones, fs, mode)
    for signal in (band, high):
        assert signal.shape == (4000, 6) and np.isfinite(signal).all()
        np.testing.assert_allclose(signal[:, :3], -signal[:, 3:])
    band_gain, high_gain = [
        np.sqrt(2 * np.mean(x[1000:3000, :3] ** 2, axis=0)) for x in (band, high)
    ]
    assert band_gain[0] < 0.05 and band_gain[1] > 0.75 and band_gain[2] < 0.05
    assert high_gain[0] < 0.4 and min(high_gain[1:]) > 0.9


def test_causal_article_filter_prefix_independent_of_future():
    before = np.zeros((300, 2))
    after = before.copy()
    after[100:] = 5
    a = nitrc.apply_article_filters(before, before, 1, "causal")
    b = nitrc.apply_article_filters(after, after, 1, "causal")
    for unchanged, perturbed in zip(a, b):
        np.testing.assert_array_equal(unchanged[:100], perturbed[:100])
        assert np.any(perturbed[100:] != 0)
