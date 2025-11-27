import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sys
import os

# Check if we are in a headless environment (for the agent)
HEADLESS = os.environ.get('HEADLESS', 'False').lower() == 'true'

def analyze_wav_3d(wav_path, center_freq=1500, output_image='data/3d_corkscrew.png'):
    print(f"--- 3D DIGITON ANALYZER ---")
    print(f"Loading: {wav_path}")
    
    # 1. Load WAV
    try:
        fs, data = wavfile.read(wav_path)
    except FileNotFoundError:
        print(f"Error: File {wav_path} not found.")
        return

    # Normalize
    if data.dtype == np.int16:
        data = data / 32768.0
    
    # If stereo, take one channel
    if len(data.shape) > 1:
        data = data[:, 0]
        
    duration = len(data) / fs
    print(f"Duration: {duration:.2f}s | Sample Rate: {fs}Hz")
    
    # 2. Downconvert to I/Q (The "Trick")
    print(f"Downconverting with Center Freq: {center_freq}Hz...")
    t = np.arange(len(data)) / fs
    lo = np.exp(-1j * 2 * np.pi * center_freq * t)
    mixed = data * lo
    
    # Low Pass Filter to remove 2*fc component
    # Cutoff at 500Hz (since spin is +/- 200Hz)
    sos = signal.butter(4, 500, 'low', fs=fs, output='sos')
    iq_signal = signal.sosfilt(sos, mixed)
    
    # 3. Downsample for Plotting
    # We want to see the spirals. Spin is ~200Hz.
    # We need ~20 points per rotation to look good -> 4000 Hz plotting rate.
    target_plot_fs = 4000
    decimate_factor = int(fs / target_plot_fs)
    if decimate_factor < 1: decimate_factor = 1
    
    iq_plot = iq_signal[::decimate_factor]
    t_plot = t[::decimate_factor]
    
    print(f"Plotting {len(t_plot)} points...")

    # 4. 3D Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract components
    I = np.real(iq_plot)
    Q = np.imag(iq_plot)
    
    # Color by Amplitude to hide noise
    amp = np.abs(iq_plot)
    # Mask noise for cleaner plot
    mask = amp > 0.05
    
    # Plot the "Corkscrews"
    # We plot segments where signal exists
    ax.scatter(t_plot[mask], I[mask], Q[mask], c=t_plot[mask], cmap='turbo', s=1, alpha=0.6)
    
    # Add "Shadows" for reference
    # Shadow on I/Q plane (at end of time)
    # ax.plot(np.full_like(I[mask], t_plot[-1]), I[mask], Q[mask], 'k.', alpha=0.1)
    
    # Labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("In-Phase (I)")
    ax.set_zlabel("Quadrature (Q)")
    ax.set_title(f"3D Digiton Corkscrews (I/Q vs Time)\nSource: {os.path.basename(wav_path)}")
    
    # Set view to see the spirals
    ax.view_init(elev=20, azim=-45)
    
    # Limits
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    print(f"Saving static view to {output_image}")
    plt.savefig(output_image, dpi=150)
    
    print("\nTo fly around interactively, run this script locally!")
    # plt.show() # Uncomment this line for local interactive mode

if __name__ == "__main__":
    target_file = 'data/02_digiton_chat_spin.wav'
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    analyze_wav_3d(target_file)
