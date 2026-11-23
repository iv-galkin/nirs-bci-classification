import numpy as np
import pandas as pd
import pytest

from nirs_bci import segmentation as seg


def test_segment_endpoints_and_trial_block_transition():
    states = [0, 0, 1, 1, 2, 1, 3, 1, 2, 1, 3, 0, 1]
    segments = seg.extract_segments(states, fs=2)
    assert segments.start_idx.tolist() == [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert segments.end_idx.tolist() == [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert segments.duration_sec.tolist() == [1, 1] + [0.5] * 9
    trials = seg.extract_trials_from_segments(segments)
    assert trials.trial_id.tolist() == list(range(9))
    assert trials.block_id.tolist() == [0] * 8 + [1]
    assert trials.label.tolist() == [1, 2, 1, 3, 1, 2, 1, 3, 1]


@pytest.mark.parametrize(
    "task,mask,labels",
    [
        ("left_vs_right", [False, False, True, True], [0, 1]),
        ("rest_vs_active", [False, True, True, True], [0, 1, 1]),
    ],
)
def test_task_labels(task, mask, labels):
    selected, actual = seg.transform_labels([0, 1, 2, 3], task)
    np.testing.assert_array_equal(selected, mask)
    np.testing.assert_array_equal(actual, labels)


def test_trial_windows_discard_each_tail():
    trials = pd.DataFrame(
        dict(
            start_idx=[0, 5, 8],
            end_idx=[5, 8, 10],
            label=[2, 3, 2],
            trial_id=[8, 9, 10],
            block_id=[1, 1, 2],
        )
    )
    windows, labels, ids, blocks = seg.collect_test_windows(
        np.arange(10)[:, None], trials, [1], "left_vs_right", fs=4, window_sec=0.5
    )
    np.testing.assert_array_equal(
        np.asarray(windows)[:, :, 0], [[0, 1], [2, 3], [5, 6]]
    )
    assert (labels.tolist(), ids.tolist(), blocks.tolist()) == (
        [0, 0, 1],
        [8, 8, 9],
        [1, 1, 1],
    )
