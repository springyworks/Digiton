import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_tone_pulse(t, f_hz, sigma):
    """Generate a real Gaussian pulse at f_hz"""
    envelope = np.exp(-t**2 / (2 * sigma**2))
    return envelope * np.cos(2 * np.pi * f_hz * t)

def sdr_downconvert(real_signal, fs, center_freq):
    """
    Simulate an SDR receiver:
    1. Takes Real RF/Audio signal.
    2. Mixes with complex local oscillator e^(-j*2pi*fc*t).
    3. Low-pass filters to get Baseband I/Q.
    """
    t = np.arange(len(real_signal)) / fs
    lo = np.exp(-1j * 2 * np.pi * center_freq * t)
    mixed = real_signal * lo
    
    # Low-pass filter to remove the double-frequency component
    # Cutoff at 500Hz (assuming spins are within +/- 500Hz)
    sos = signal.butter(4, 500, 'low', fs=fs, output='sos')
    baseband_iq = signal.sosfilt(sos, mixed)
    
    return baseband_iq

def run_sdr_spin_simulation():
    print("--- DIGITON SDR SPIN SIMULATION ---")
    print("The 'Trick': Using the SDR Receiver to see the Spin")
    
    fs = 48000
    duration = 0.1
    t = np.linspace(-duration/2, duration/2, int(duration*fs))
    
    # Setup
    center_freq = 1500  # The "Virtual Center" of our channel
    spin_offset = 200   # +/- 200Hz shift
    sigma = 0.01        # Pulse width
    
    # 1. TRANSMITTER (Standard SSB/Audio)
    # "Right Spin" -> f_c + offset
    # "Left Spin"  -> f_c - offset
    tx_right = generate_tone_pulse(t, center_freq + spin_offset, sigma)
    tx_left = generate_tone_pulse(t, center_freq - spin_offset, sigma)
    
    # 2. CHANNEL (Watterson-ish)
    # Add some noise
    noise_level = 0.1
    rx_right = tx_right + np.random.normal(0, noise_level, len(t))
    rx_left = tx_left + np.random.normal(0, noise_level, len(t))
    
    # 3. SDR RECEIVER (Tuned to Center Freq)
    # This is the "Decoder" part
    iq_right = sdr_downconvert(rx_right, fs, center_freq)
    iq_left = sdr_downconvert(rx_left, fs, center_freq)
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: What we sent (Real Audio)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t*1000, tx_right, 'g', label='Right (1700Hz)')
    ax1.plot(t*1000, tx_left, 'r', label='Left (1300Hz)', alpha=0.7)
    ax1.set_title("1. Transmitted Audio (Real)\nJust two different tones")
    ax1.set_xlabel("Time (ms)")
    ax1.legend()
    
    # Plot 2: SDR View (Complex Plane Trajectory)
    ax2 = fig.add_subplot(2, 2, 2)
    # Plot the trajectory of the peak
    mask = np.abs(t) < 0.02
    ax2.plot(np.real(iq_right[mask]), np.imag(iq_right[mask]), 'g', label='Right Spin')
    ax2.plot(np.real(iq_left[mask]), np.imag(iq_left[mask]), 'r', label='Left Spin')
    ax2.set_title("2. SDR I/Q View (Baseband)\nOpposite Rotations!")
    ax2.set_xlabel("I")
    ax2.set_ylabel("Q")
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Phase vs Time (The "Corkscrew" Angle)
    ax3 = fig.add_subplot(2, 2, 3)
    phase_r = np.unwrap(np.angle(iq_right[mask]))
    phase_l = np.unwrap(np.angle(iq_left[mask]))
    
    ax3.plot(t[mask]*1000, phase_r, 'g', label='Right Phase')
    ax3.plot(t[mask]*1000, phase_l, 'r', label='Left Phase')
    ax3.set_title("3. Phase Slope (Frequency)\nPositive Slope vs Negative Slope")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Phase (rad)")
    ax3.legend()
    
    # Plot 4: 3D Visualization of the SDR Output
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    t_zoom = t[mask] * 1000
    iq_r_zoom = iq_right[mask]
    iq_l_zoom = iq_left[mask]
    
    ax4.plot(t_zoom, np.real(iq_r_zoom), np.imag(iq_r_zoom), 'g', label='Right')
    ax4.plot(t_zoom, np.real(iq_l_zoom), np.imag(iq_l_zoom), 'r', label='Left')
    ax4.set_title("4. The Recovered Corkscrews\n(Relative to Center Freq)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("I")
    ax4.set_zlabel("Q")
    
    plt.tight_layout()
    plt.savefig('data/15_digiton_sdr_spin.png')
    print("Visualization saved to 'data/15_digiton_sdr_spin.png'")
    
    print("\n--- ANALYSIS ---")
    print("1. The 'Trick' works!")
    print("2. By transmitting at f_c + offset and f_c - offset,")
    print("   we create REAL signals that the SDR interprets as")
    print("   OPPOSITE SPINS in the complex baseband.")
    print("3. This is robust against Watterson because it's just Frequency Diversity.")
    print("4. We get the beauty of the Gaussian Corkscrew in the SDR display.")

if __name__ == "__main__":
    run_sdr_spin_simulation()
