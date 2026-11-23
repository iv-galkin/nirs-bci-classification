import h5py
import numpy as np
import pytest

from nirs_bci.data import load_all_days
from nirs_bci.spectral import compute_band_power


def test_synthetic_matlab_layout_and_numeric_day_order(tmp_path):
    for day in [10, 2]:
        with h5py.File(tmp_path / f"Day_{day}.mat", "w") as f:
            f["day"] = day
            f["HbO"] = [[1, 2, 3], [4, 5, 6]]
            f["Hb"] = [[10, 20, 30], [40, 50, 60]]
            f["States"] = [[0, 2, 3]]
            f["sampling_rate"] = 2
    days = load_all_days(tmp_path)
    assert [d["day"] for d in days] == [2, 10]
    np.testing.assert_array_equal(
        days[0]["X"], [[1, 4, 10, 40], [2, 5, 20, 50], [3, 6, 30, 60]]
    )
    assert days[0]["trials"].start_idx.tolist() == [1, 2]


def test_triangle_spectrum_band_integral():
    # Area of a triangle with base 0.4 Hz and peak density 5.
    f = np.array([0, 0.2, 0.4, 0.6, 0.8])
    p = np.array([100, 0, 5, 0, 100])
    assert compute_band_power(f, p, 0.2, 0.6) == pytest.approx(1)
    assert np.isnan(compute_band_power(f, p, 0.39, 0.41))
