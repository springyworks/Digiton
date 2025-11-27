import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def generate_g_ping(t, t_center, f_hz, width_s, amplitude=1.0):
    """
    Generates a Gaussian Ping (G-Ping).
    Mathematically identical to a Real Morlet Wavelet.
    """
    sigma = width_s / 6.0 
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * f_hz * t)
    return amplitude * envelope * carrier

def simulate_tf_grid():
    print("--- DIGITON TIME-FREQUENCY GRID DEMO ---")
    print("Protocol: TFMA (Time-Frequency Multiple Access)")
    print("Concept: Users pick a Time Slot AND a Frequency Lane.")
    print("-" * 60)

    fs = 48000
    cycle_duration = 3.2 
    num_slots = 8
    slot_duration = cycle_duration / num_slots # 0.4s
    
    t = np.linspace(0, cycle_duration, int(cycle_duration * fs))
    sig = np.zeros_like(t)
    
    # Define Frequency Lanes (Audio Tones)
    # In SSB bandwidth (300-2700Hz)
    lanes = {
        "LOW": 600,
        "MID": 1200,
        "HIGH": 1800,
        "ULTRA": 2400
    }
    
    # --- SCENARIO ---
    # 1. MASTER (Net Control)
    # Occupies Slot 0 on ALL Lanes (Broadband Beacon) or just Low?
    # Let's say Master beacons on LOW to save bandwidth, or sends a chord.
    # Let's send a CHORD (Low+Mid+High) to mark the start clearly.
    print("[Slot 0] Master sends SYNC CHORD (Low+Mid+High)")
    sig += generate_g_ping(t, 0.2, lanes["LOW"], 0.05, 0.5)
    sig += generate_g_ping(t, 0.2, lanes["MID"], 0.05, 0.5)
    sig += generate_g_ping(t, 0.2, lanes["HIGH"], 0.05, 0.5)
    
    # 2. ALICE (Slot 2, LOW Lane)
    print("[Slot 2] Alice responds on LOW Lane (600Hz)")
    sig += generate_g_ping(t, slot_duration*2 + 0.2, lanes["LOW"], 0.05)
    
    # 3. BOB (Slot 2, HIGH Lane) - SAME TIME, DIFFERENT FREQ!
    print("[Slot 2] Bob responds on HIGH Lane (1800Hz) - NO COLLISION!")
    sig += generate_g_ping(t, slot_duration*2 + 0.2, lanes["HIGH"], 0.05)
    
    # 4. CHARLIE (Slot 5, MID Lane)
    print("[Slot 5] Charlie responds on MID Lane (1200Hz)")
    sig += generate_g_ping(t, slot_duration*5 + 0.2, lanes["MID"], 0.05)
    
    # 5. DAVE (Slot 6, ULTRA Lane) - Fast Data
    print("[Slot 6] Dave sends Fast Data on ULTRA Lane (2400Hz)")
    # Dave sends a burst
    for i in range(8):
        sig += generate_g_ping(t, slot_duration*6 + 0.05 + i*0.02, lanes["ULTRA"], 0.01)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 10))
    
    # 1. Time Domain (Waveform)
    plt.subplot(2, 1, 1)
    plt.plot(t, sig, 'k', linewidth=0.5)
    plt.title("1. Time Domain: The 'Rhythm' (Slots)")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)
    
    # Draw Slots
    for i in range(num_slots + 1):
        plt.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.5)
        if i < num_slots:
            plt.text(i * slot_duration + 0.05, 1.2, f"SLOT {i}", color='green', fontsize=8)
            
    plt.text(slot_duration*2 + 0.05, 0.8, "Alice & Bob\n(Same Time)", color='red', fontsize=9)

    # 2. Spectrogram (The Grid)
    plt.subplot(2, 1, 2)
    f, t_spec, Sxx = signal.spectrogram(sig, fs, nperseg=1024, noverlap=900)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    plt.title("2. Time-Frequency Grid: The 'Score' (Slots + Lanes)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.ylim(0, 3000)
    
    # Annotate the Grid
    # Master
    plt.text(0.2, 600, "MASTER", color='white', fontweight='bold', ha='center')
    plt.text(0.2, 1200, "MASTER", color='white', fontweight='bold', ha='center')
    plt.text(0.2, 1800, "MASTER", color='white', fontweight='bold', ha='center')
    
    # Users
    plt.text(slot_duration*2 + 0.2, 600, "ALICE", color='white', fontweight='bold', ha='center')
    plt.text(slot_duration*2 + 0.2, 1800, "BOB", color='white', fontweight='bold', ha='center')
    plt.text(slot_duration*5 + 0.2, 1200, "CHARLIE", color='white', fontweight='bold', ha='center')
    plt.text(slot_duration*6 + 0.2, 2400, "DAVE", color='white', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig('data/05_digiton_tf_grid.png')
    print("Visualization saved to 'data/05_digiton_tf_grid.png'")

    # Save Audio
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig_norm = sig / max_val * 0.9
    else:
        sig_norm = sig
    
    sig_int16 = (sig_norm * 32767).astype(np.int16)
    wavfile.write('data/05_digiton_tf_grid.wav', fs, sig_int16)
    print("Audio saved to 'data/05_digiton_tf_grid.wav'")

if __name__ == "__main__":
    simulate_tf_grid()
