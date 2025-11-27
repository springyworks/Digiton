import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import os
from scipy.io import wavfile

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_chirplet(t, f_center, bandwidth, duration, direction='up'):
    """
    Generate a Gaussian Chirplet (Chirped Gabor Atom).
    
    Args:
        t: Time vector
        f_center: Center frequency (Hz)
        bandwidth: Frequency sweep range (Hz)
        duration: Pulse duration (sigma of Gaussian)
        direction: 'up' or 'down'
    """
    # Gaussian Envelope
    sigma = duration
    envelope = np.exp(-t**2 / (2 * sigma**2))
    
    # Chirp Rate (Hz/s)
    # We want to sweep 'bandwidth' Hz over roughly the pulse width (2*sigma)
    k = bandwidth / (2 * sigma)
    if direction == 'down':
        k = -k
        
    # Instantaneous Frequency: f(t) = f_center + k*t
    # Phase: phi(t) = 2*pi * (f_center*t + 0.5*k*t^2)
    phase = 2 * np.pi * (f_center * t + 0.5 * k * t**2)
    
    # Analytic Signal (Complex)
    chirplet = envelope * np.exp(1j * phase)
    
    return chirplet

def run_chirplet_investigation():
    print("--- DIGITON CHIRPLET INVESTIGATION ---")
    print("Testing 'Left Turn' (Down-Chirp) vs 'Right Turn' (Up-Chirp)")
    
    fs = 48000
    duration_view = 0.1
    t = np.linspace(-duration_view/2, duration_view/2, int(duration_view*fs))
    
    f_center = 1500
    bw = 500  # Sweep 500Hz
    pulse_width = 0.02 # 20ms
    
    # Generate Chirplets
    chirp_up = generate_chirplet(t, f_center, bw, pulse_width, 'up')
    chirp_down = generate_chirplet(t, f_center, bw, pulse_width, 'down')
    
    # --- ORTHOGONALITY CHECK ---
    # Cross-correlation at t=0
    corr_val = np.abs(np.vdot(chirp_up, chirp_down)) / np.sqrt(np.vdot(chirp_up, chirp_up) * np.vdot(chirp_down, chirp_down))
    
    print(f"\nOrthogonality Check:")
    print(f"Correlation between UP and DOWN: {corr_val:.4f}")
    if corr_val < 0.5:
        print("Result: DISTINCT. We can use both simultaneously!")
    else:
        print("Result: HIGH CORRELATION. Hard to distinguish.")

    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Spectrograms
    ax1 = fig.add_subplot(2, 2, 1)
    f, t_spec, Sxx = signal.spectrogram(np.real(chirp_up), fs, nperseg=256, noverlap=240)
    # Center time
    t_spec = t_spec - t_spec[-1]/2
    ax1.pcolormesh(t_spec*1000, f, Sxx, shading='gouraud', cmap='viridis')
    ax1.set_title("1. Right Turn (Up-Chirp)\nFrequency rises")
    ax1.set_ylim(1000, 2000)
    ax1.set_ylabel("Frequency (Hz)")
    
    ax2 = fig.add_subplot(2, 2, 2)
    f, t_spec, Sxx = signal.spectrogram(np.real(chirp_down), fs, nperseg=256, noverlap=240)
    t_spec = t_spec - t_spec[-1]/2
    ax2.pcolormesh(t_spec*1000, f, Sxx, shading='gouraud', cmap='magma')
    ax2.set_title("2. Left Turn (Down-Chirp)\nFrequency falls")
    ax2.set_ylim(1000, 2000)
    
    # 3. 3D Corkscrew Comparison
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    # Zoom in on center
    mask = np.abs(t) < 0.01
    t_zoom = t[mask]
    c_up_zoom = chirp_up[mask]
    
    ax3.plot(t_zoom*1000, np.real(c_up_zoom), np.imag(c_up_zoom), 'g', linewidth=1)
    ax3.set_title("3. The 'Right' Corkscrew (3D)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("I")
    ax3.set_zlabel("Q")
    
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    c_down_zoom = chirp_down[mask]
    ax4.plot(t_zoom*1000, np.real(c_down_zoom), np.imag(c_down_zoom), 'r', linewidth=1)
    ax4.set_title("4. The 'Left' Corkscrew (3D)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("I")
    ax4.set_zlabel("Q")
    
    plt.tight_layout()
    plt.savefig('data/13_digiton_chirplets.png')
    print("Visualization saved to 'data/13_digiton_chirplets.png'")
    
    # Save Audio
    # Combine Up and Down for the audio demo (separated by silence)
    silence = np.zeros(int(0.1 * fs))
    combined = np.concatenate([np.real(chirp_up), silence, np.real(chirp_down)])
    combined_norm = combined / np.max(np.abs(combined)) * 0.9
    wavfile.write('data/13_digiton_chirplets.wav', fs, (combined_norm * 32767).astype(np.int16))
    print("Audio saved to 'data/13_digiton_chirplets.wav'")

if __name__ == "__main__":
    run_chirplet_investigation()
