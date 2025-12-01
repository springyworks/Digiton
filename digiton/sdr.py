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
    Determine rotation direction from I/Q chunk via Lag-1 Autocorrelation (Pulse-Pair).
    This is significantly more robust to noise than instantaneous frequency unwrapping.
    Returns: ('right'|'left'|'ambiguous', avg_freq_hz)
    """
    if iq_chunk.size < 4:
        return None, 0.0
    
    # Check for silence/noise floor
    # Use a relative threshold or absolute? 
    # The caller usually passes a chunk that is already gated, but safety is good.
    if np.max(np.abs(iq_chunk)) < 1e-4:
        return None, 0.0

    # Lag-1 Autocorrelation (Pulse-Pair Algorithm)
    # R(1) = sum(x[n] * conj(x[n-1]))
    # This calculates the sum of phase vectors between adjacent samples,
    # naturally weighting by signal power and cancelling out zero-mean noise.
    x = iq_chunk
    r1 = np.sum(x[1:] * np.conj(x[:-1]))
    
    if np.abs(r1) < 1e-12:
        return None, 0.0
        
    # Calculate average phase change per sample
    avg_phase_diff = np.angle(r1)
    
    # Convert to Frequency
    avg_freq = avg_phase_diff * fs_hz / (2 * np.pi)
    
    if avg_freq > 50:
        return 'right', avg_freq
    if avg_freq < -50:
        return 'left', avg_freq
    return 'ambiguous', avg_freq


def detect_spin_robust(iq_chunks: list[np.ndarray], fs_hz: float):
    """
    Accumulate Lag-1 Autocorrelation across multiple chunks to improve SNR for frequency detection.
    This allows detecting spin even when individual pings are buried in noise (-30dB),
    provided the sequence detection (matched filter) found them.
    """
    total_r1 = 0j
    total_samples = 0
    
    for chunk in iq_chunks:
        if chunk.size < 4: continue
        
        # We skip the amplitude check here because we expect low SNR.
        # The gating is done by the caller (detect_presence).
        
        # R(1) for this chunk
        r1 = np.sum(chunk[1:] * np.conj(chunk[:-1]))
        total_r1 += r1
        total_samples += chunk.size
        
    if total_samples == 0 or np.abs(total_r1) < 1e-12:
        return None, 0.0
        
    avg_phase_diff = np.angle(total_r1)
    avg_freq = avg_phase_diff * fs_hz / (2 * np.pi)
    
    if avg_freq > 50: return 'right', avg_freq
    if avg_freq < -50: return 'left', avg_freq
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
