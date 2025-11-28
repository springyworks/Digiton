import numpy as np


def gaussian_envelope(t: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-t**2 / (2.0 * sigma * sigma))


def morlet_pulse(fc_hz: float, spin_offset_hz: float, sigma: float, fs: float, spin: str = 'right', trunc_sigmas: float = 3.0) -> np.ndarray:
    """
    Generate a real-valued Gaussian-enveloped sinusoid (Morlet-like) at fc±offset.
    spin: 'right' → +offset, 'left' → -offset
    Duration: ±trunc_sigmas * sigma (default 3σ → ~99.7% energy)
    """
    offset = +abs(spin_offset_hz) if spin == 'right' else -abs(spin_offset_hz)
    half = trunc_sigmas * sigma
    n = int(np.ceil((2 * half) * fs)) + 1
    t = np.linspace(-half, half, n)
    env = gaussian_envelope(t, sigma)
    return env * np.cos(2 * np.pi * (fc_hz + offset) * t)
