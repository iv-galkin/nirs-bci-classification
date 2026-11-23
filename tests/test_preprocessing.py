import numpy as np
import pytest

from nirs_bci.preprocessing import preprocess_nirs, sma_detrend_explicit_available_only


@pytest.mark.parametrize(
    "mode,slow_range",
    [
        ("raw", (0.999, 1.001)),
        ("sma", (0.20, 0.30)),
        ("highpass_0002", (0.6, 0.9)),
        ("highpass_0018", (0.005, 0.05)),
    ],
)
def test_filter_frequency_response(mode, slow_range):
    # Separate tones in separate channels; discard edges and measure whole cycles.
    t = np.arange(8000)
    x = np.sin(2 * np.pi * t[:, None] * np.array([0.002, 0.1]))
    y = preprocess_nirs(x, 1, mode)
    assert y.shape == x.shape and np.isfinite(y).all()
    gains = np.sqrt(2 * np.mean(y[2000:6000] ** 2, axis=0))
    assert slow_range[0] < gains[0] < slow_range[1]
    assert 0.97 < gains[1] < 1.03


def test_sma_edges_use_available_samples():
    y = sma_detrend_explicit_available_only([[0], [3], [9]], 3)
    np.testing.assert_allclose(y[:, 0], [-1.5, -1, 3])


@pytest.mark.parametrize("mode", ["sma", "highpass_0002", "highpass_0018"])
def test_offline_filters_detect_future_signal_in_prefix(mode):
    # Diagnostic: filtering the full session before splitting is noncausal.
    x = np.zeros((600, 1))
    x[300:] = 1
    assert abs(preprocess_nirs(x, 1, mode)[299, 0]) > 0.1
