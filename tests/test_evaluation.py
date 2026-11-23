import numpy as np
import pandas as pd
import pytest

from nirs_bci import evaluation as ev


def test_confusion_orientation_and_macro_recall():
    p, cm, q = ev.dissertation_score([0, 0, 0, 1, 1], [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(cm, [[1, 0], [2, 2]])
    np.testing.assert_allclose(q, [[1 / 3, 0], [2 / 3, 1]])
    assert p == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "truth,pred,cm",
    [
        ([0, 0, 1], [0, 0, 0], [[2, 1], [0, 0]]),
        ([1, 1], [1, 1], [[0, 0], [0, 2]]),
    ],
)
def test_fixed_classes(truth, pred, cm):
    p, counts, _ = ev.dissertation_score_fixed_labels(truth, pred, [0, 1])
    np.testing.assert_array_equal(counts, cm)
    assert p == 0.5


@pytest.mark.xfail(
    strict=True,
    reason="Predictions outside observed truth labels disappear from the denominator",
)
def test_unseen_prediction_is_still_an_error():
    p, cm, _ = ev.dissertation_score([0, 0], [0, 1])
    assert cm.sum() == 2
    assert p <= 0.5


def test_days_have_equal_weight_after_fold_average():
    folds = pd.DataFrame(
        dict(
            file=["A", "A", "A", "B"],
            day=[1, 1, 1, 2],
            subject="s",
            task="t",
            model="LDA",
            filter="raw",
            score=[0, 0.5, 1, 1],
            status=["ok", "ok", "failed", "ok"],
        )
    )
    days = ev.aggregate_lobo_by_day(folds)
    assert days.n_folds.tolist() == [2, 1]
    assert days.P.tolist() == [0.25, 1]
    summary = ev.summarize_median_iqr(days, ["raw"]).iloc[0]
    assert (summary["n"], summary.q25, summary["median"], summary.q75) == (
        2,
        0.4375,
        0.625,
        0.8125,
    )
