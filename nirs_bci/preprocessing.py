import numpy as np
from scipy.signal import cheby1, sosfiltfilt


def sma_detrend_explicit_available_only(X, window: int = 201):
    """Subtract simple moving average without artificial edge padding."""
    X = np.asarray(X, dtype=float)

    if window % 2 == 0:
        raise ValueError("SMA window must be odd")

    half = window // 2
    n_time = X.shape[0]
    trend = np.empty_like(X, dtype=float)

    for t in range(n_time):
        left = max(0, t - half)
        right = min(n_time, t + half + 1)
        trend[t] = X[left:right].mean(axis=0)

    return X - trend


def highpass_cheby1_filter(X, fs: float, cutoff: float, order: int = 1, rp: float = 1):
    """Offline zero-phase Chebyshev I high-pass filter."""
    X = np.asarray(X, dtype=float)
    nyq = fs / 2

    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive: {cutoff}")
    if cutoff >= nyq:
        raise ValueError(f"cutoff={cutoff} must be smaller than Nyquist={nyq}")

    sos = cheby1(
        N=order,
        rp=rp,
        Wn=cutoff / nyq,
        btype="highpass",
        output="sos",
    )
    return sosfiltfilt(sos, X, axis=0)


def preprocess_nirs(X, fs: float, mode: str):
    """Apply one fixed preprocessing mode."""
    if mode == "raw":
        return np.asarray(X, dtype=float).copy()

    if mode == "sma":
        return sma_detrend_explicit_available_only(X, window=201)

    if mode == "highpass_0002":
        return highpass_cheby1_filter(X, fs, cutoff=0.002, order=1, rp=1)

    if mode == "highpass_0018":
        return highpass_cheby1_filter(X, fs, cutoff=0.018, order=1, rp=1)

    raise ValueError(f"Unknown preprocessing mode: {mode}")


def split_hbo_hbr(X):
    """
    Split feature matrix with order:
        [HbO_0, ..., HbO_C-1, HbR_0, ..., HbR_C-1]
    """
    X = np.asarray(X, dtype=float)

    if X.shape[1] % 2 != 0:
        raise ValueError("Expected even number of features")

    n_channels = X.shape[1] // 2
    return X[:, :n_channels], X[:, n_channels:]


# Backward-compatible alias. Prefer importing make_fnirs_bands from spectral.py.
def make_fnirs_bands(fs: float):
    from .spectral import make_fnirs_bands as _make_fnirs_bands

    return _make_fnirs_bands(fs)
