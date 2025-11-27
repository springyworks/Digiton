import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import os
from scipy.io import wavfile

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_complex_morlet(t, f0, sigma):
    """
    Generate a complex Morlet wavelet (Gabor atom).
    s(t) = exp(-t^2 / (2*sigma^2)) * exp(j * 2*pi * f0 * t)
    """
    envelope = np.exp(-t**2 / (2 * sigma**2))
    carrier = np.exp(1j * 2 * np.pi * f0 * t)
    return envelope * carrier

def simulate_ssb_chain(complex_signal):
    """
    Simulate what happens over SSB:
    1. We can only send REAL audio to the radio (Microphone/Line In).
    2. Radio shifts it up (USB).
    3. Receiver shifts it down.
    4. Receiver gets REAL audio.
    5. Receiver performs Hilbert Transform to recover Analytic Signal (I/Q).
    """
    # 1. Transmit Real Part (Audio)
    tx_audio = np.real(complex_signal)
    
    # 2. Channel (Ideal for now, just passing the audio)
    rx_audio = tx_audio
    
    # 3. Receiver Recovery (Hilbert Transform)
    # This restores the "Corkscrew" (Analytic Signal)
    rx_analytic = signal.hilbert(rx_audio)
    
    return tx_audio, rx_analytic

def run_corkscrew_investigation():
    print("--- DIGITON CORKSCREW INVESTIGATION ---")
    print("Visualizing the 'Spin' of the Gabor Atom")
    
    fs = 48000
    duration = 0.02  # 20ms zoom
    t = np.linspace(-duration/2, duration/2, int(duration*fs))
    
    f0 = 1000  # 1kHz tone
    sigma = 0.005 # Short pulse
    
    # Generate the "Perfect" Corkscrew
    corkscrew = generate_complex_morlet(t, f0, sigma)
    
    # Simulate SSB survival
    tx_real, rx_recovered = simulate_ssb_chain(corkscrew)
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: The 3D Corkscrew (Original)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(t*1000, np.real(corkscrew), np.imag(corkscrew), label='Original I/Q', linewidth=1.5)
    ax1.set_title("1. The 'Digiton' Corkscrew (Source)\nComplex Morlet Wavelet")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Real (I)")
    ax1.set_zlabel("Imag (Q)")
    ax1.legend()
    
    # Plot 2: What we actually send (Real Audio)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t*1000, tx_real, 'g', label='Transmitted Audio (Real)')
    ax2.plot(t*1000, np.abs(corkscrew), 'k--', alpha=0.3, label='Envelope')
    ax2.set_title("2. What goes over SSB (Real Audio)\nThe 'Shadow' of the Corkscrew")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Recovered 3D Corkscrew (Receiver)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    # Plot recovered
    ax3.plot(t*1000, np.real(rx_recovered), np.imag(rx_recovered), 'r', label='Recovered I/Q', alpha=0.8)
    ax3.set_title("3. Recovered via Hilbert Transform\nThe Corkscrew is Reborn!")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Real (I)")
    ax3.set_zlabel("Imag (Q)")
    ax3.legend()
    
    # Plot 4: Phase Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    phase_orig = np.angle(corkscrew)
    phase_rec = np.angle(rx_recovered)
    
    # Unwrap for cleaner plot
    phase_orig_u = np.unwrap(phase_orig)
    phase_rec_u = np.unwrap(phase_rec)
    
    ax4.plot(t*1000, phase_orig_u, 'b', label='Original Phase')
    ax4.plot(t*1000, phase_rec_u, 'r--', label='Recovered Phase')
    ax4.set_title("4. Phase Integrity\nDoes the 'Spin' survive?")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Phase (radians)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('data/12_digiton_corkscrew.png')
    print("Visualization saved to 'data/12_digiton_corkscrew.png'")
    
    # --- ANALYSIS ---
    print("\n--- ANALYSIS: DOES IT SURVIVE SSB? ---")
    print("1. SSB Radio transmits Real signals (Audio).")
    print("2. This flattens our 3D corkscrew into a 2D shadow (Projection).")
    print("3. HOWEVER, because the signal is narrowband (Bandlimited),")
    print("   the Hilbert Transform at the receiver can mathematically")
    print("   reconstruct the missing Imaginary part.")
    print("4. RESULT: The Corkscrew DOES survive, provided we do the math!")
    print("\n--- WHAT DOES THIS BRING US? ---")
    print("1. INSTANTANEOUS PHASE: We can track the exact rotation of the signal.")
    print("2. SUB-SAMPLE TIMING: The phase tells us exactly 'where' in the wave cycle we are.")
    print("   This allows Time-of-Arrival precision better than 1 sample!")
    print("3. COHERENT DEMODULATION: We can detect phase shifts (PSK) inside the pulse.")
    
    # Save Audio
    # Normalize
    tx_norm = tx_real / np.max(np.abs(tx_real)) * 0.9
    wavfile.write('data/12_digiton_corkscrew.wav', fs, (tx_norm * 32767).astype(np.int16))
    print("Audio saved to 'data/12_digiton_corkscrew.wav'")

if __name__ == "__main__":
    run_corkscrew_investigation()
