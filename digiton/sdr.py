import numpy as np
from scipy import signal


def iq_downconvert(real_signal: np.ndarray, fc_hz: float, fs_hz: float, lpf_hz: float = 500.0) -> np.ndarray:
    """
    Downconvert real passband to complex baseband at fc using complex LO and low-pass filter.
    Returns complex64 array I+jQ.
    """
    t = np.arange(real_signal.size) / fs_hz
    lo = np.exp(-1j * 2 * np.pi * fc_hz * t)
    mixed = real_signal.astype(np.float32) * lo.astype(np.complex64)
    sos = signal.butter(4, lpf_hz, 'low', fs=fs_hz, output='sos')
    iq = signal.sosfilt(sos, mixed)
    return iq.astype(np.complex64)


def detect_spin(iq_chunk: np.ndarray, fs_hz: float):
    """
    Determine rotation direction from I/Q chunk via amplitude-weighted instantaneous frequency.
    Returns: ('right'|'left'|'ambiguous', avg_freq_hz)
    """
    if iq_chunk.size < 4:
        return None, 0.0
    amp = np.abs(iq_chunk)
    if np.max(amp) < 1e-3:
        return None, 0.0
    phase = np.unwrap(np.angle(iq_chunk))
    inst_freq = np.diff(phase) / (2 * np.pi * (1.0 / fs_hz))
    weights = amp[1:] ** 2
    if np.sum(weights) <= 0:
        return None, 0.0
    avg_freq = float(np.average(inst_freq, weights=weights))
    if avg_freq > 50:
        return 'right', avg_freq
    if avg_freq < -50:
        return 'left', avg_freq
    return 'ambiguous', avg_freq


def coherent_integrate(trains: np.ndarray, repeats: int) -> np.ndarray:
    """
    Coherently sum repeated pulses. If `trains` is 1-D, it is assumed to be a
    concatenation of repeats. If 2-D, shape should be (repeats, samples_per_pulse).
    """
    arr = np.asarray(trains)
    if arr.ndim == 1:
        n = arr.size // repeats
        arr = arr[: n * repeats].reshape(repeats, n)
    return np.sum(arr, axis=0)
