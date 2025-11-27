import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def generate_digiton(t, t_center, f_hz, width_s, amplitude=1.0):
    sigma = width_s / 6.0 
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * f_hz * t)
    return amplitude * envelope * carrier

def generate_deep_sequence(t, t_start, f_hz, count, interval):
    """
    Generates a sequence of pings (Coherent Integration Train).
    """
    sig = np.zeros_like(t)
    width = 0.05 # Standard width
    for i in range(count):
        t_c = t_start + i * interval
        sig += generate_digiton(t, t_c, f_hz, width)
    return sig

def simulate_deep_mode():
    print("--- DIGITON DEEP MODE SIMULATION (-50dB) ---")
    print("Scenario: Station ALICE is weak (-50dB). Negotiates 'Deep Mode' (Repetition).")
    print("-" * 60)

    # Parameters
    fs = 48000
    cycle_duration = 1.0 
    num_slots = 8
    slot_duration = cycle_duration / num_slots
    
    # We need enough time for a "Deep Train"
    # Let's say a Deep Train is 16 pings spaced by 0.1s
    # This takes 1.6s. It spans multiple cycles or a long slot?
    # For this demo, let's compress time or extend the cycle.
    # Let's extend the cycle to 4.0s to fit deep trains.
    cycle_duration = 4.0
    slot_duration = cycle_duration / num_slots # 0.5s per slot
    
    total_cycles = 2
    total_time = total_cycles * cycle_duration
    
    t = np.linspace(0, total_time, int(total_time * fs))
    clean_signal = np.zeros_like(t)
    
    freq_master = 800
    freq_alice = 1200 
    
    # --- CYCLE 1: Discovery (Deep Mode) ---
    print("[Cycle 1] Master CQ. Alice sends DEEP TRAIN (16 repeats) to be seen.")
    # Master CQ (Standard)
    clean_signal += generate_digiton(t, 0.5, freq_master, 0.05)
    
    # Alice Joins in Slot 2 with a DEEP TRAIN
    # 16 pings, spaced by 0.1s (10Hz rate)
    # Slot 2 starts at 1.0s.
    t_alice_start = 1.0 + 0.1 
    clean_signal += generate_deep_sequence(t, t_alice_start, freq_alice, count=16, interval=0.15)
    
    # --- CYCLE 2: Data (Deep Mode) ---
    print("[Cycle 2] Master ACKs. Alice sends DATA BIT '1' (Another Train).")
    # Master CQ
    clean_signal += generate_digiton(t, cycle_duration + 0.5, freq_master, 0.05)
    
    # Alice sends Data in Slot 2
    t_alice_data = cycle_duration + 1.0 + 0.1
    clean_signal += generate_deep_sequence(t, t_alice_data, freq_alice, count=16, interval=0.15)

    # --- NOISE GENERATION (-50dB) ---
    # Signal Power (approx)
    # A single sine wave amp 1 has power 0.5.
    # Duty cycle is low, but let's base SNR on peak power for simplicity of "Amplitude SNR" vs "Energy SNR".
    # -50dB means Noise Power is 10^5 times Signal Power.
    # Amplitude ratio: sqrt(10^5) = 316.
    # So Noise Amplitude ~ 300 * Signal Amplitude.
    
    print("Adding -50dB Noise...")
    noise_amp = 300.0 
    noise = np.random.normal(0, noise_amp, len(t))
    noisy_signal = clean_signal + noise
    
    # --- PROCESSING (Coherent Integration) ---
    print("Demodulating...")
    # 1. Matched Filter (Correlation with Single Ping)
    # Create template
    t_temp = np.linspace(-0.1, 0.1, int(0.2*fs))
    template = generate_digiton(t_temp, 0, freq_alice, 0.05)
    
    # Correlate
    # We use FFT convolution for speed
    corr = signal.fftconvolve(noisy_signal, template[::-1], mode='same')
    
    # 2. Coherent Integration (Stacking)
    # We know the repetition interval is 0.15s
    samples_per_interval = int(0.15 * fs)
    num_repeats = 16
    
    # Let's try to recover the "Deep Train" in Cycle 1
    # We fold the correlation output modulo the repetition interval
    # This is a simplified "Epoch Folding"
    
    # Focus on Cycle 1 Slot 2 region
    start_idx = int(1.0 * fs)
    end_idx = int(4.0 * fs)
    region = corr[start_idx:end_idx]
    
    # Fold
    folded = np.zeros(samples_per_interval)
    fold_count = len(region) // samples_per_interval
    
    for i in range(fold_count):
        chunk = region[i*samples_per_interval : (i+1)*samples_per_interval]
        if len(chunk) == len(folded):
            folded += chunk
            
    # Normalize
    folded /= fold_count

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 10))
    
    # 1. Clean Signal (What user hears)
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, clean_signal, 'g')
    ax1.set_title("1. Clean Signal (Audio Output) - 'Machine Gun' Repetition")
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. Noisy Signal (-50dB)
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(t[::10], noisy_signal[::10], 'k', linewidth=0.1) # Downsample for plot speed
    ax2.set_title("2. Noisy Signal (-50dB) - Signal is Invisible")
    ax2.set_ylim(-1000, 1000) # Noise is huge
    
    # 3. Raw Correlation (Single Ping Match)
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(t[::10], corr[::10], 'b', linewidth=0.5)
    ax3.set_title("3. Matched Filter Output (Single Ping) - Still Buried in Noise")
    ax3.set_ylim(-5000, 5000)
    
    # 4. Coherent Integration (The Magic)
    ax4 = plt.subplot(4, 1, 4)
    t_fold = np.linspace(0, 0.15, len(folded))
    ax4.plot(t_fold * 1000, folded, 'r', linewidth=2)
    ax4.set_title(f"4. Coherent Integration ({num_repeats}x Folded) - SIGNAL RECOVERED!")
    ax4.set_xlabel("Time (ms) within Repetition Interval")
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/02_digiton_deep_mode.png')
    print("Visualization saved to 'data/02_digiton_deep_mode.png'")

    # Save CLEAN Audio
    max_val = np.max(np.abs(clean_signal))
    if max_val > 0:
        signal_norm = clean_signal / max_val * 0.9
    else:
        signal_norm = clean_signal
    
    signal_int16 = (signal_norm * 32767).astype(np.int16)
    wavfile.write('data/02_digiton_deep_mode_clean.wav', fs, signal_int16)
    print("Audio saved to 'digiton_deep_mode_clean.wav'")

if __name__ == "__main__":
    simulate_deep_mode()
