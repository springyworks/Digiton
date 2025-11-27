import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from hf_channel_simulator import HFChannelSimulator

def generate_g_ping(t, t_center, f_hz, width_s, amplitude=1.0):
    sigma = width_s / 6.0 
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * f_hz * t)
    return amplitude * envelope * carrier

def simulate_tf_deep_test():
    print("--- DIGITON TF-GRID DEEP TEST (-60dB) ---")
    print("Scenario: Recovering a Time-Frequency Grid from -60dB Noise")
    print("Method: Synchronous Epoch Folding (Cycle Stacking)")
    print("-" * 60)

    fs = 12000 # Lower sample rate to save memory/time, still covers HF
    cycle_duration = 0.5 # Short cycle for demo
    num_repeats = 256 # Needed for -60dB
    total_time = cycle_duration * num_repeats
    
    print(f"Cycle Duration: {cycle_duration}s")
    print(f"Repeats: {num_repeats}")
    print(f"Total Time: {total_time}s")
    
    # Generate ONE Clean Cycle
    t_cycle = np.linspace(0, cycle_duration, int(cycle_duration * fs))
    sig_cycle = np.zeros_like(t_cycle)
    
    # Lanes
    f_low = 600
    f_mid = 1200
    f_high = 1800
    
    # Slot 0 (0.0 - 0.1): Master Chord
    sig_cycle += generate_g_ping(t_cycle, 0.05, f_low, 0.03, 0.5)
    sig_cycle += generate_g_ping(t_cycle, 0.05, f_mid, 0.03, 0.5)
    sig_cycle += generate_g_ping(t_cycle, 0.05, f_high, 0.03, 0.5)
    
    # Slot 1 (0.15 - 0.25): Alice (Low) + Bob (High)
    sig_cycle += generate_g_ping(t_cycle, 0.2, f_low, 0.03, 0.8) # Alice
    sig_cycle += generate_g_ping(t_cycle, 0.2, f_high, 0.03, 0.8) # Bob
    
    # Slot 2 (0.3 - 0.4): Charlie (Mid)
    sig_cycle += generate_g_ping(t_cycle, 0.35, f_mid, 0.03, 0.8) # Charlie
    
    # Create the Full Train
    print("Generating Full Signal Train...")
    sig_full = np.tile(sig_cycle, num_repeats)
    
    # Apply HF Channel (Watterson Fading)
    print("Applying HF Channel (Watterson Fading)...")
    # We use the simulator but bypass its internal noise to add our own -60dB
    channel = HFChannelSimulator(fs, snr_db=100) # High SNR first
    # Add fading
    sig_faded = channel.simulate_hf_channel(sig_full, include_fading=True, freq_offset=False)
    
    # Add -60dB Noise
    # Signal Power of the cycle
    sig_power = np.mean(sig_faded**2)
    if sig_power == 0: sig_power = 1e-9
    
    target_snr = -60.0
    noise_power = sig_power / (10**(target_snr/10))
    noise_amp = np.sqrt(noise_power)
    
    print(f"Adding Noise (SNR {target_snr}dB)...")
    noise = np.random.normal(0, noise_amp, len(sig_faded))
    sig_noisy = sig_faded + noise
    
    # --- DEMODULATION (Epoch Folding) ---
    print("Demodulating (Stacking Cycles)...")
    
    # Reshape to (Num_Repeats, Samples_Per_Cycle)
    samples_per_cycle = len(sig_cycle)
    # Truncate to multiple of cycle
    limit = num_repeats * samples_per_cycle
    sig_stack = sig_noisy[:limit].reshape((num_repeats, samples_per_cycle))
    
    # Average (Coherent Integration)
    sig_recovered = np.mean(sig_stack, axis=0)
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 12))
    
    # 1. The Clean Cycle (Truth)
    plt.subplot(3, 1, 1)
    f, t_spec, Sxx = signal.spectrogram(sig_cycle, fs, nperseg=256, noverlap=200)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
    plt.title("1. The Transmitted Grid (Clean Cycle)")
    plt.ylabel("Freq (Hz)")
    plt.text(0.05, 1200, "MASTER", color='white', ha='center', fontweight='bold')
    plt.text(0.2, 600, "ALICE", color='white', ha='center', fontweight='bold')
    plt.text(0.2, 1800, "BOB", color='white', ha='center', fontweight='bold')
    plt.text(0.35, 1200, "CHARLIE", color='white', ha='center', fontweight='bold')
    
    # 2. The Received Signal (First Cycle)
    plt.subplot(3, 1, 2)
    # Show a snippet of the noisy signal
    plt.plot(np.arange(samples_per_cycle)/fs, sig_noisy[:samples_per_cycle], 'k', linewidth=0.5)
    plt.title(f"2. The Reality (-60dB Noise + Fading) - First Cycle")
    plt.ylim(-noise_amp*3, noise_amp*3)
    
    # 3. The Recovered Grid (After Stacking)
    plt.subplot(3, 1, 3)
    f_rec, t_rec, Sxx_rec = signal.spectrogram(sig_recovered, fs, nperseg=256, noverlap=200)
    plt.pcolormesh(t_rec, f_rec, 10 * np.log10(Sxx_rec + 1e-10), shading='gouraud', cmap='jet')
    plt.title(f"3. RECOVERED GRID (After {num_repeats}x Stacking)")
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig('data/08_digiton_tf_deep_test.png')
    print("Visualization saved to 'digiton_tf_deep_test.png'")
    
    # Save Audio
    # 1. Noisy (What you hear)
    wavfile.write('digiton_tf_deep_noisy.wav', fs, (sig_noisy[:fs*5]/np.max(np.abs(sig_noisy))*32767).astype(np.int16))
    
    # 2. Recovered (The Ghost)
    wavfile.write('digiton_tf_deep_recovered.wav', fs, (sig_recovered/np.max(np.abs(sig_recovered))*32767).astype(np.int16))
    print("Audio saved.")

if __name__ == "__main__":
    simulate_tf_deep_test()
