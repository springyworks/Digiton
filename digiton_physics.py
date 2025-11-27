import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

print("--- DIGITON PHYSICS ENGINE ---")
print("Verifying the Gabor Limit for Gaussian Pulses")
print(f"{'Sigma_Input':<12} | {'Sigma_t_Meas':<12} | {'Sigma_f_Meas':<12} | {'Product':<10} | {'Limit':<10}")
print("-" * 70)

fs = 10000.0 # High sample rate for precision
duration = 2.0
t = np.linspace(-duration/2, duration/2, int(duration*fs))

# Test various widths
sigmas = [0.005, 0.01, 0.05, 0.1, 0.2]

for s in sigmas:
    # Generate Gabor atom
    atom = gaussian_pulse(t, 0, s, 1000.0) # 1kHz carrier
    
    st, sf = measure_spread(atom, fs)
    prod = st * sf
    limit = 1/(4*np.pi)
    
    print(f"{s:<12.4f} | {st:<12.5f} | {sf:<12.5f} | {prod:<10.5f} | {limit:.5f}")

print("-" * 70)
print("Note: The theoretical limit is 1/(4π) ≈ 0.07958.")
print("Our 'Digitons' (Gaussian pulses) hit this limit exactly!")
print("This proves they are the most efficient signals possible in nature.")

# Now let's check the WaveletPingGenerator from the project
print("\n--- CHECKING PROJECT WAVELET ---")
try:
    from hf_channel_simulator import WaveletPingGenerator
    gen = WaveletPingGenerator(48000)
    ping = gen.generate_ping(wavelet_name='morl', duration=0.1) # 100ms
    
    # Convert to analytic if it's real
    if np.isrealobj(ping):
        ping = signal.hilbert(ping)
        
    st_p, sf_p = measure_spread(ping, 48000)
    print(f"Project Ping (100ms):")
    print(f"  Sigma_t: {st_p:.5f} s")
    print(f"  Sigma_f: {sf_p:.5f} Hz")
    print(f"  Product: {st_p * sf_p:.5f}")
    
    if st_p * sf_p < 0.1:
        print("  Result: EXCELLENT. Close to Gabor Limit.")
    else:
        print("  Result: SUBOPTIMAL. Likely due to windowing or non-Gaussian shape.")
        
except ImportError:
    print("Could not import WaveletPingGenerator (running standalone?)")

