import numpy as np

def gaussian_pulse(t, t0, sigma, f0):
    """
    Generate a Gabor atom (Gaussian envelope * Complex Exponential).
    E(t) = (1/(sigma * sqrt(2pi))) * exp(-(t-t0)^2 / (2*sigma^2))
    x(t) = E(t) * exp(j * 2pi * f0 * t)
    """
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    # Normalize energy to 1
    envelope /= np.sqrt(np.sum(envelope**2))
    carrier = np.exp(1j * 2 * np.pi * f0 * t)
    return envelope * carrier

def measure_spread(x, fs):
    """
    Measure time and frequency spread of a signal x.
    """
    N = len(x)
    t = np.arange(N) / fs
    
    # Time spread
    prob_t = np.abs(x)**2
    prob_t /= np.sum(prob_t)
    t_mean = np.sum(t * prob_t)
    sigma_t = np.sqrt(np.sum((t - t_mean)**2 * prob_t))
    
    # Frequency spread
    # Use FFT
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Shift to center
    X_shifted = np.fft.fftshift(X)
    freqs_shifted = np.fft.fftshift(freqs)
    
    # Power spectrum
    prob_f = np.abs(X_shifted)**2
    prob_f /= np.sum(prob_f)
    
    # Calculate mean frequency (should be f0)
    f_mean = np.sum(freqs_shifted * prob_f)
    
    # Calculate spread
    sigma_f = np.sqrt(np.sum((freqs_shifted - f_mean)**2 * prob_f))
    
    return sigma_t, sigma_f
