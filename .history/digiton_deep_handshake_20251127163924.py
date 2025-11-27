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
    width = 0.05 
    for i in range(count):
        t_c = t_start + i * interval
        sig += generate_digiton(t, t_c, f_hz, width)
    return sig

def simulate_deep_handshake():
    print("--- DIGITON DEEP HANDSHAKE SIMULATION (-50dB) ---")
    print("Scenario: Cold Start. Alice requests entry. Master Handshakes. Message Exchange.")
    print("-" * 60)

    # Parameters
    fs = 48000
    cycle_duration = 4.0 # Long cycle for deep mode
    num_slots = 8
    slot_duration = cycle_duration / num_slots # 0.5s per slot
    
    total_cycles = 4
    total_time = total_cycles * cycle_duration
    
    t = np.linspace(0, total_time, int(total_time * fs))
    clean_signal = np.zeros_like(t)
    
    freq_master = 800
    freq_alice = 1200 
    
    # Train Parameters
    train_count = 12 # 12 pings
    train_interval = 0.15 # 150ms spacing
    
    # --- CYCLE 1: REQUEST ---
    print("[Cycle 1] Discovery: Alice sends Request (Deep Train) in Slot 2")
    # Master CQ
    clean_signal += generate_digiton(t, 0.5, freq_master, 0.05)
    
    # Alice Request (Slot 2 starts at 1.0s)
    t_req = 1.0 + 0.1
    clean_signal += generate_deep_sequence(t, t_req, freq_alice, train_count, train_interval)
    
    # --- CYCLE 2: HANDSHAKE / GRANT ---
    print("[Cycle 2] Handshake: Master sends ACK/GRANT (Deep Train) in Slot 2")
    t_c2 = cycle_duration
    # Master CQ
    clean_signal += generate_digiton(t, t_c2 + 0.5, freq_master, 0.05)
    
    # Master ACK in Slot 2 (Echoing the slot back to Alice)
    # Master uses his frequency (800Hz) to distinguish
    t_ack = t_c2 + 1.0 + 0.1
    clean_signal += generate_deep_sequence(t, t_ack, freq_master, train_count, train_interval)

    # --- CYCLE 3: MESSAGE (ALICE) ---
    print("[Cycle 3] Message: Alice sends 'HELLO' (Deep Train)")
    t_c3 = 2 * cycle_duration
    # Master CQ
    clean_signal += generate_digiton(t, t_c3 + 0.5, freq_master, 0.05)
    
    # Alice Message
    t_msg = t_c3 + 1.0 + 0.1
    clean_signal += generate_deep_sequence(t, t_msg, freq_alice, train_count, train_interval)

    # --- CYCLE 4: REPLY (MASTER) ---
    print("[Cycle 4] Reply: Master sends 'ROGER' (Deep Train)")
    t_c4 = 3 * cycle_duration
    # Master CQ
    clean_signal += generate_digiton(t, t_c4 + 0.5, freq_master, 0.05)
    
    # Master Reply
    t_reply = t_c4 + 1.0 + 0.1
    clean_signal += generate_deep_sequence(t, t_reply, freq_master, train_count, train_interval)

    # --- NOISE GENERATION (-50dB) ---
    print("Adding -50dB Noise for visualization...")
    noise_amp = 300.0 
    noise = np.random.normal(0, noise_amp, len(t))
    noisy_signal = clean_signal + noise

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 10))
    
    # 1. Timeline (Clean)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, clean_signal, 'k', linewidth=0.8)
    ax1.set_title("1. The Conversation (Clean Audio) - Deep Mode Handshake")
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlim(0, total_time)
    ax1.grid(True, alpha=0.3)
    
    # Annotations
    # Cycle 1
    ax1.text(0.2, 1.2, "CYCLE 1: REQUEST", fontweight='bold')
    ax1.text(t_req + 0.5, 0.8, "Alice: REQUEST", color='blue', ha='center')
    
    # Cycle 2
    ax1.text(cycle_duration + 0.2, 1.2, "CYCLE 2: GRANT", fontweight='bold')
    ax1.text(t_ack + 0.5, 0.8, "Master: GRANT (ACK)", color='red', ha='center')
    
    # Cycle 3
    ax1.text(2*cycle_duration + 0.2, 1.2, "CYCLE 3: MSG", fontweight='bold')
    ax1.text(t_msg + 0.5, 0.8, "Alice: DATA", color='blue', ha='center')
    
    # Cycle 4
    ax1.text(3*cycle_duration + 0.2, 1.2, "CYCLE 4: REPLY", fontweight='bold')
    ax1.text(t_reply + 0.5, 0.8, "Master: REPLY", color='red', ha='center')

    # Draw Slot Boundaries
    for i in range(int(total_cycles * num_slots)):
        ax1.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.2)

    # 2. The Reality (-50dB)
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t[::20], noisy_signal[::20], 'k', linewidth=0.1)
    ax2.set_title("2. The Reality (-50dB Noise Floor) - Total Silence")
    ax2.set_ylim(-1000, 1000)
    
    # 3. Spectrogram (Recovered View)
    # We simulate what the modem "sees" after processing (Integration)
    # Since we can't easily plot the integrated output for the whole timeline without running the full stack,
    # We will plot the Spectrogram of the CLEAN signal to show the structure clearly.
    ax3 = plt.subplot(3, 1, 3)
    f, t_spec, Sxx = signal.spectrogram(clean_signal, fs, nperseg=1024, noverlap=512)
    ax3.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax3.set_title("3. Spectrogram (Signal Structure)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(0, 2000)
    
    plt.tight_layout()
    plt.savefig('data/09_digiton_deep_handshake.png')
    print("Visualization saved to 'digiton_deep_handshake.png'")

    # Save CLEAN Audio
    max_val = np.max(np.abs(clean_signal))
    if max_val > 0:
        signal_norm = clean_signal / max_val * 0.9
    else:
        signal_norm = clean_signal
    
    signal_int16 = (signal_norm * 32767).astype(np.int16)
    wavfile.write('digiton_deep_handshake_clean.wav', fs, signal_int16)
    print("Audio saved to 'digiton_deep_handshake_clean.wav'")

if __name__ == "__main__":
    simulate_deep_handshake()
