import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_spin_morlet(t, f0, sigma, spin='right'):
    """
    Generate a Morlet Wavelet with specific 'Spin' (Frequency Sign).
    
    Right Spin (Standard): e^(+j * 2pi * f0 * t)
    Left Spin (Conjugate): e^(-j * 2pi * f0 * t)
    """
    envelope = np.exp(-t**2 / (2 * sigma**2))
    
    if spin == 'right':
        # Positive Frequency -> Counter-Clockwise / Right Hand
        carrier = np.exp(1j * 2 * np.pi * f0 * t)
    else:
        # Negative Frequency -> Clockwise / Left Hand
        carrier = np.exp(-1j * 2 * np.pi * f0 * t)
        
    return envelope * carrier

def run_spin_investigation():
    print("--- DIGITON SPIN INVESTIGATION ---")
    print("Testing 'Right Turn' (+Freq) vs 'Left Turn' (-Freq)")
    
    fs = 48000
    duration = 0.01  # 10ms zoom
    t = np.linspace(-duration/2, duration/2, int(duration*fs))
    
    f0 = 1000
    sigma = 0.002
    
    # Generate Spins
    sig_right = generate_spin_morlet(t, f0, sigma, 'right')
    sig_left = generate_spin_morlet(t, f0, sigma, 'left')
    
    # Get Real Parts (What we send over Mono Audio)
    real_right = np.real(sig_right)
    real_left = np.real(sig_left)
    
    # Check difference
    diff = np.max(np.abs(real_right - real_left))
    
    print(f"\nDifference between Real(Right) and Real(Left): {diff:.10f}")
    if diff < 1e-9:
        print("CRITICAL PHYSICS RESULT: The Real parts are IDENTICAL.")
        print("In Mono Audio (SSB), 'Left Turn' and 'Right Turn' look exactly the same.")
        print("We cannot distinguish them unless we use Stereo I/Q (SDR).")
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 3D Right Spin
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(t*1000, np.real(sig_right), np.imag(sig_right), 'g', linewidth=1.5)
    ax1.set_title("1. Right Turn (+Freq)\nThe Standard Digiton")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Real")
    ax1.set_zlabel("Imag")
    
    # 2. 3D Left Spin
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(t*1000, np.real(sig_left), np.imag(sig_left), 'r', linewidth=1.5)
    ax2.set_title("2. Left Turn (-Freq)\nThe Conjugate Digiton")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Real")
    ax2.set_zlabel("Imag")
    
    # 3. The "Shadows" (Real Part)
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(t*1000, real_right, 'g', linewidth=3, alpha=0.5, label='Real(Right)')
    ax3.plot(t*1000, real_left, 'r--', linewidth=1.5, label='Real(Left)')
    ax3.set_title("3. The 'Shadows' (Mono Audio Transmission)\nThey overlap perfectly! Physics prevents distinguishing them in Mono.")
    ax3.set_xlabel("Time (ms)")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('data/14_digiton_spin_test.png')
    print("Visualization saved to 'data/14_digiton_spin_test.png'")

if __name__ == "__main__":
    run_spin_investigation()
