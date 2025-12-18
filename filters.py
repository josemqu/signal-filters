import numpy as np
from scipy.signal import butter, filtfilt


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    window = int(window)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def low_pass_butterworth(
    x: np.ndarray,
    fs_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    nyq = 0.5 * fs_hz
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError("cutoff_hz must be between 0 and Nyquist")

    b, a = butter(int(order), cutoff_hz / nyq, btype="low", analog=False)
    return filtfilt(b, a, x)


def kalman_1d(
    z: np.ndarray,
    q: float,
    r: float,
    x0: float | None = None,
    p0: float = 1.0,
) -> np.ndarray:
    """Simple 1D Kalman filter (random walk model).

    State model:
      x_k = x_{k-1} + w,  w ~ N(0, q)
      z_k = x_k + v,      v ~ N(0, r)

    Parameters q and r are variances.
    """

    if q <= 0 or r <= 0:
        raise ValueError("q and r must be > 0")

    z = np.asarray(z, dtype=float)
    n = z.size
    if n == 0:
        return z.copy()

    x_hat = np.empty(n, dtype=float)

    x = float(z[0] if x0 is None else x0)
    p = float(p0)

    for k in range(n):
        # Predict
        p = p + q

        # Update
        k_gain = p / (p + r)
        x = x + k_gain * (z[k] - x)
        p = (1.0 - k_gain) * p

        x_hat[k] = x

    return x_hat
