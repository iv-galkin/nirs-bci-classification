import numpy as np
import pytest

from nirs_bci.models import GaussianWindowDiscriminant


@pytest.mark.parametrize("mode,variances", [("LDA", (2.5, 2.5)), ("QDA", (1, 4))])
def test_window_discriminants(mode, variances):
    # Class means 0 and 6, unbiased sample variances 1 and 4.
    model = GaussianWindowDiscriminant(mode).fit(
        np.array([-1, 0, 1, 4, 6, 8])[:, None], [2, 2, 2, 7, 7, 7]
    )
    scores = model.window_scores([[1], [0]])
    a, b = variances
    assert scores == pytest.approx(
        {2: 2 * np.log(a) + 1 / a, 7: 2 * np.log(b) + 61 / b}
    )
    np.testing.assert_array_equal(model.predict_windows([[[0]], [[6]]]), [2, 7])


@pytest.mark.parametrize(
    "method,alpha,ridge,expected",
    [
        ("none", 0, 0.4, [[2, 2], [2, 5]]),
        ("diagonal", 0.5, 0, [[1, 1], [1, 4]]),
        ("spherical", 1, 0, [[2.5, 0], [0, 2.5]]),
    ],
)
def test_regularized_covariance(method, alpha, ridge, expected):
    x = np.array([[-1, -2], [0, 0], [1, 2], [4, 3], [5, 5], [6, 7]])
    model = GaussianWindowDiscriminant("LDA", ridge, method, alpha).fit(
        x, [1, 1, 1, 2, 2, 2]
    )
    for cls in (1, 2):
        np.testing.assert_allclose(model.covariances_[cls], expected)


@pytest.mark.xfail(
    strict=True,
    reason="A one-column window silently broadcasts to the fitted feature count",
)
def test_predict_rejects_wrong_number_of_channels():
    model = GaussianWindowDiscriminant(covariance_reg=1).fit(
        [[1, 2], [2, 4], [4, 2], [3, 6]], [0, 0, 1, 1]
    )
    with pytest.raises(ValueError):
        model.predict_window([[1], [2]])
