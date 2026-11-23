import numpy as np
import pytest

from nirs_bci import features as f
from nirs_bci.channel_selection import channel_feature_indices


@pytest.mark.parametrize(
    "component,expected",
    [
        ("hbo", [3, 1]),
        ("hbr", [30, 10]),
        ("hbo_plus_hbr", [33, 11]),
        ("hbo_minus_hbr", [-27, -9]),
    ],
)
def test_component_and_channel_order(component, expected):
    x = [[1, 2, 3, 10, 20, 30]]
    np.testing.assert_array_equal(
        f.select_hemo_component_with_channels(x, [2, 0], component), [expected]
    )
    np.testing.assert_array_equal(channel_feature_indices([2, 0], 3), [2, 5, 0, 3])


def test_window_statistics():
    result = f.windows_to_features(
        [[[2, 10], [4, 6]]], ["mean", "std", "slope", "amplitude"]
    )
    np.testing.assert_allclose(result, [[3, 8, 1, 2, 2, -4, 2, 4]])


@pytest.mark.xfail(
    strict=True,
    reason="Feature names follow caller order, feature values use canonical order",
)
def test_feature_names_describe_their_values():
    stats = ["std", "mean"]
    values = f.window_to_stat_features([[2, 10], [4, 6]], stats)
    assert dict(zip(f.make_feature_names(1, stats), values)) == {
        "std_HbO_ch0": 1,
        "std_HbR_ch0": 2,
        "mean_HbO_ch0": 3,
        "mean_HbR_ch0": 8,
    }
