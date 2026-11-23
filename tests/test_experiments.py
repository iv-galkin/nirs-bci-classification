from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from nirs_bci import evaluation, experiments


def test_lobo_observations_are_disjoint(monkeypatch):
    trials = pd.DataFrame(
        dict(
            block_id=[0, 0, 1, 1, 2, 2],
            label=[2, 3] * 3,
            trial_id=range(6),
            start_idx=range(0, 12, 2),
            end_idx=range(2, 14, 2),
        )
    )
    day = dict(
        file_name="day", sampling_rate=2, trials=trials, X=np.arange(12)[:, None]
    )
    model = Mock()
    model.predict_windows.return_value = [0, 1]
    monkeypatch.setattr(experiments, "GaussianWindowDiscriminant", lambda **kw: model)
    rows = experiments.run_lobo_one_day(day, "left_vs_right", "LDA", "raw")
    assert rows.score.tolist() == [1, 1, 1]
    for i, start in enumerate([8, 4, 0]):
        train, labels = model.fit.call_args_list[i].args
        (windows,) = model.predict_windows.call_args_list[i].args
        held_out = list(range(start, start + 4))
        np.testing.assert_array_equal(np.ravel(windows), held_out)
        np.testing.assert_array_equal(
            train[:, 0], [j for j in range(12) if j not in held_out]
        )
        np.testing.assert_array_equal(labels, [0, 0, 1, 1] * 2)


@pytest.mark.parametrize("name", ["LDA", "SVM_linear_mean", "CatBoost_mean"])
def test_classifier_training_unit(monkeypatch, name):
    model = Mock()
    model.predict_windows.side_effect = [[1, 0], [0, 1]]
    model.predict.side_effect = [[0, 1], [1, 0]]
    monkeypatch.setattr(experiments, "make_classifier", lambda _: model)
    raw = np.array([0, 2, 99, 6, 10, 77])[:, None]
    train = [raw[:2], raw[3:5]]
    test = [np.array([[12], [16]]), np.array([[20], [24]])]
    scores = experiments.fit_and_predict(
        name, raw, [0, 0, 0, 1, 1, 1], test, [0, 1], train, [0, 1], {name: ["mean"]}
    )
    assert scores == (1, 0)
    x, labels = model.fit.call_args.args
    if name == "LDA":
        np.testing.assert_array_equal(x, raw)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(
            model.predict_windows.call_args_list[0].args[0], test
        )
        np.testing.assert_array_equal(
            model.predict_windows.call_args_list[1].args[0], train
        )
    else:
        np.testing.assert_array_equal(x, [[1], [8]])
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_array_equal(model.predict.call_args.args[0], [[14], [22]])


def test_no_history_and_summary_weight():
    result = experiments._run_previous_blocks_variant(
        {}, "exp1", [], [np.ones((2, 1))], [0], "left_vs_right", "LDA", 1, None
    )
    assert result["status"] == "no_train_random" and np.isnan(result["P_train"])
    assert result["P_test"] == 0.5
    rows = pd.DataFrame(
        [
            result,
            dict(variant="exp1", status="ok", P_train=1, P_test=0.75),
            dict(variant="exp1", status="error", P_train=0, P_test=0),
        ]
    )
    row = evaluation.summarize_experiment_grid(rows, ["variant"], ["variant"]).iloc[0]
    assert (row.n, row.P_train_median, row.P_test_median, row.gap_median) == (
        2,
        1,
        0.625,
        0.25,
    )
    assert (row.P_test_q25, row.P_test_q75) == (0.5625, 0.6875)


@pytest.mark.parametrize("variant", ["exp1", "exp2"])
@pytest.mark.parametrize("history", [[], [object()]])
def test_no_test_windows_have_no_test_score(variant, history):
    row = experiments._run_previous_blocks_variant(
        {}, variant, history, [], [], "left_vs_right", "LDA", 1, None
    )
    assert row["status"] == "no_test_data"
    assert np.isnan(row["P_train"]) and np.isnan(row["P_test"])


def test_previous_blocks_scores_and_aggregation():
    # Block 0 is rest-only. Blocks 1 and 2 each contain both imagery classes.
    trials = pd.DataFrame(dict(
        block_id=[0, 1, 1, 2, 2], trial_id=range(5), label=[1, 2, 3, 2, 3],
        start_idx=[0, 2, 4, 6, 8], end_idx=[2, 4, 6, 8, 10],
    ))
    day = dict(file_name="day", sampling_rate=2, trials=trials)
    x = np.array([5, 5, 0, 1, 9, 10, 0, 1, 9, 10])[:, None]
    frames = experiments.run_one_permutation(
        [0], "left_vs_right", "LDA", "raw", {"raw": [(day, x)]}
    )
    for variant, rows in zip(("exp1", "exp2"), frames):
        assert rows.variant.tolist() == [variant] * 3
        assert rows.status.tolist() == ["no_test_data", "no_train_random", "ok"]
        np.testing.assert_allclose(rows.P_train, [np.nan, np.nan, 1])
        np.testing.assert_allclose(rows.P_test, [np.nan, .5, 1])
        summary = evaluation.summarize_experiment_grid(rows, ["variant"], ["variant"]).iloc[0]
        assert summary.n == 2
        assert (summary.P_test_q25, summary.P_test_median, summary.P_test_q75) == (.625, .75, .875)
        assert summary.P_train_median == 1 and summary.gap_median == 0


@pytest.mark.xfail(
    strict=True,
    reason="exp1 reports global rather than same-session training block count",
)
def test_exp1_first_block_always_reports_zero_train_blocks():
    trials = pd.DataFrame(
        dict(block_id=[0], label=[2], trial_id=[0], start_idx=[0], end_idx=[2])
    )
    day = dict(file_name="day", sampling_rate=1, trials=trials)
    exp1, _ = experiments.run_one_permutation(
        [0, 1], "left_vs_right", "LDA", "raw", {"raw": [(day, np.ones((2, 1)))] * 2}
    )
    assert exp1.n_train_blocks.tolist() == [0, 0]
