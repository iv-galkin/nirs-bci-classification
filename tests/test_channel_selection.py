import numpy as np
import pandas as pd
import pytest

from nirs_bci import channel_selection as cs


def protocol_trials():
    return pd.DataFrame(
        dict(
            block_id=np.repeat([0, 1, 2], 8),
            trial_id=range(24),
            label=[1, 2, 1, 3, 1, 2, 1, 3] * 3,
            start_idx=range(24),
            end_idx=range(1, 25),
        )
    )


def test_nested_folds_partition_only_outer_training_trials():
    trials = protocol_trials()
    allowed = set(range(8)) | set(range(16, 24))
    folds = cs.make_half_block_validation_splits(trials, [0, 2])
    validation_sets = set()
    for fold in folds:
        train, val = (
            set(fold["inner_train_trial_ids"]),
            set(fold["inner_val_trial_ids"]),
        )
        assert train.isdisjoint(val) and train | val == allowed
        subset = trials.iloc[sorted(val)]
        assert subset.block_id.unique().tolist() == [fold["val_from_block"]]
        assert sorted(subset.label) == [1, 1, 2, 3]
        validation_sets.add(tuple(sorted(val)))
    assert len(validation_sets) == len(folds) == 48


@pytest.mark.parametrize(
    "method,expected",
    [
        ("all_channels", (0, 1, 2, 3)),
        ("best_single", (0,)),
        ("best_pair", (1, 2)),
        ("greedy_from_single", (0, 3)),
        ("greedy_from_pair", (1, 2, 3)),
    ],
)
def test_selection_strategy_and_outer_fold(monkeypatch, method, expected):
    # Single and pair winners lie on different greedy paths; adding more can hurt.
    scores = {(0,): 0.5, (0, 3): 0.7, (1, 2): 0.8, (1, 2, 3): 0.9}
    seen = []

    def score(x, trials, blocks, task, model, channels, fs, window):
        seen.append(tuple(blocks))
        return scores.get(tuple(sorted(channels)), 0.1)

    monkeypatch.setattr(cs, "score_channel_set_inner_cv", score)
    day = dict(
        file_name="day",
        sampling_rate=1,
        trials=protocol_trials(),
        X=np.random.default_rng(42).normal(size=(24, 8)),
    )
    rows = cs.run_lobo_one_day_with_channel_selection(
        day, "rest_vs_active", "LDA", "raw", method
    )
    assert rows.status.tolist() == ["ok"] * 3
    assert rows.selected_channels.tolist() == [expected] * 3
    assert rows.test_blocks.tolist() == [(2,), (1,), (0,)]
    if method == "all_channels":
        assert not seen
    else:
        from itertools import groupby

        assert [key for key, _ in groupby(seen)] == [(0, 1), (0, 2), (1, 2)]


def test_inner_fit_and_validation_observations(monkeypatch):
    trials = protocol_trials()
    allowed = set(range(8)) | set(range(16, 24))
    seen = []

    class Probe:
        def __init__(self, mode):
            pass

        def fit(self, x, y):
            self.train = set(x[:, 0].astype(int))
            np.testing.assert_array_equal(
                y, (trials.label.to_numpy()[x[:, 0].astype(int)] != 1)
            )

        def predict_windows(self, windows):
            ids = np.asarray(windows)[:, 0, 0].astype(int)
            seen.append((self.train, set(ids)))
            return (trials.label.to_numpy()[ids] != 1).astype(int)

    monkeypatch.setattr(cs, "GaussianWindowDiscriminant", Probe)
    x = np.column_stack([np.arange(24), 100 + np.arange(24)])
    score = cs.score_channel_set_inner_cv(
        x, trials, [0, 2], "rest_vs_active", "LDA", [0], 1
    )
    assert score == 1 and len(seen) == 48
    for train, val in seen:
        assert len(train) == 12 and len(val) == 4
        assert train.isdisjoint(val) and train | val == allowed
