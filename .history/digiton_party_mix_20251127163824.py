import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def generate_digiton(t, t_center, f_hz, width_s, amplitude=1.0):
    sigma = width_s / 6.0 
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * f_hz * t)
    return amplitude * envelope * carrier

def generate_burst(t, t_start, f_hz, data_bits, bit_duration, amplitude=1.0):
    sig = np.zeros_like(t)
    current_t = t_start
    for bit in data_bits:
        if bit == 1:
            sig += generate_digiton(t, current_t + bit_duration/2, f_hz, bit_duration*0.8, amplitude)
        current_t += bit_duration
    return sig

def text_to_bits(text):
    bits = []
    for char in text:
        val = ord(char)
        for i in range(7, -1, -1):
            bits.append((val >> i) & 1)
    return bits

def simulate_party_mix():
    print("--- DIGITON PARTY MIX SIMULATION ---")
    print("Scenario: Mixed Speed Party.")
    print("  1. ALICE (Far/Weak): -10dB SNR. Needs SLOW mode.")
    print("  2. BOB (Close/Strong): +20dB SNR. Uses FAST mode.")
    print("-" * 60)

    fs = 48000
    cycle_duration = 3.2 
    num_slots = 8
    slot_duration = cycle_duration / num_slots # 0.4s
    
    total_cycles = 3
    total_time = total_cycles * cycle_duration
    
    t = np.linspace(0, total_time, int(total_time * fs))
    
    # We create a "Clean Logic" signal (what the protocol intends)
    # and a "Real World" signal (with amplitude differences).
    # The user wants to HEAR the negotiation, so we'll normalize the clean signal for audio.
    sig_clean = np.zeros_like(t)
    
    # Frequencies
    f_master = 600  # Low pitch anchor
    f_alice = 1000  # Medium pitch
    f_bob = 1600    # High pitch
    
    # --- CYCLE 1: DISCOVERY & REQUESTS ---
    print("[Cycle 1] Discovery")
    # Master CQ (Slot 0)
    sig_clean += generate_digiton(t, 0.2, f_master, 0.05)
    
    # ALICE (Weak) - Slot 2
    # Sends a Standard Ping (Requesting entry).
    # In reality this is weak.
    t_alice_req = slot_duration * 2 + 0.2
    sig_clean += generate_digiton(t, t_alice_req, f_alice, 0.05, amplitude=0.5)
    
    # BOB (Strong) - Slot 5
    # Sends a Fast Burst Request (He knows he's strong).
    # "REQ" (1010)
    t_bob_req = slot_duration * 5 + 0.1
    sig_clean += generate_burst(t, t_bob_req, f_bob, [1,0,1,0], 0.01, amplitude=1.0)

    # --- CYCLE 2: NEGOTIATION (THE HANDSHAKE) ---
    print("[Cycle 2] Negotiation (Auto-Throttle)")
    t_c2 = cycle_duration
    # Master CQ
    sig_clean += generate_digiton(t, t_c2 + 0.2, f_master, 0.05)
    
    # MASTER -> ALICE (Slot 2)
    # Master detects Alice is weak. Sends "SLOW ACK" (Long, Low Tone).
    # A single long ping.
    t_ack_alice = t_c2 + slot_duration * 2 + 0.2
    sig_clean += generate_digiton(t, t_ack_alice, f_master, 0.15, amplitude=0.8) # 150ms width = SLOW
    
    # MASTER -> BOB (Slot 5)
    # Master detects Bob is strong. Sends "FAST ACK" (Fast Chirps).
    # "OK" (11)
    t_ack_bob = t_c2 + slot_duration * 5 + 0.1
    sig_clean += generate_burst(t, t_ack_bob, f_master, [1,1], 0.01, amplitude=0.8)

    # --- CYCLE 3: TRAFFIC (MIXED SPEEDS) ---
    print("[Cycle 3] Traffic")
    t_c3 = 2 * cycle_duration
    # Master CQ
    sig_clean += generate_digiton(t, t_c3 + 0.2, f_master, 0.05)
    
    # ALICE (Slot 2) - SLOW DATA
    # She sends 1 bit of data using a robust ping.
    t_alice_data = t_c3 + slot_duration * 2 + 0.2
    sig_clean += generate_digiton(t, t_alice_data, f_alice, 0.1, amplitude=0.5)
    
    # BOB (Slot 5) - FAST DATA
    # He sends "HI" (ASCII) in a burst.
    # 'H' (01001000) 'I' (01001001)
    t_bob_data = t_c3 + slot_duration * 5 + 0.05
    bits_hi = text_to_bits("HI")
    sig_clean += generate_burst(t, t_bob_data, f_bob, bits_hi, 0.005, amplitude=1.0) # 5ms bits

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 8))
    
    # Plot Timeline
    plt.subplot(2, 1, 1)
    plt.plot(t, sig_clean, 'k', linewidth=0.6)
    plt.title("Wavelet Party Protocol: Mixed Speed Negotiation (Auto-Throttle)")
    plt.ylabel("Amplitude")
    plt.xlim(0, total_time)
    plt.grid(True, alpha=0.3)
    
    # Annotations
    # Cycle 1
    plt.text(0.2, 1.1, "CYCLE 1: DISCOVERY", fontweight='bold')
    plt.text(t_alice_req, 0.6, "Alice (Weak)\nStd Ping", ha='center', color='blue', fontsize=8)
    plt.text(t_bob_req, 1.1, "Bob (Strong)\nFast REQ", ha='center', color='red', fontsize=8)
    
    # Cycle 2
    plt.text(t_c2 + 0.2, 1.1, "CYCLE 2: NEGOTIATION", fontweight='bold')
    plt.text(t_ack_alice, 0.9, "Master -> Alice\nSLOW ACK (Long Pulse)", ha='center', color='green', fontsize=8)
    plt.text(t_ack_bob, 0.9, "Master -> Bob\nFAST ACK (Burst)", ha='center', color='green', fontsize=8)
    
    # Cycle 3
    plt.text(t_c3 + 0.2, 1.1, "CYCLE 3: TRAFFIC", fontweight='bold')
    plt.text(t_alice_data, 0.6, "Alice\nSlow Data (1 bit)", ha='center', color='blue', fontsize=8)
    plt.text(t_bob_data, 1.1, "Bob\nFast Data 'HI' (16 bits)", ha='center', color='red', fontsize=8)

    # Draw Slots
    for i in range(int(total_cycles * num_slots)):
        plt.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.2)

    # Zoom on Cycle 3
    plt.subplot(2, 1, 2)
    # Zoom into Cycle 3 to see the speed difference
    zoom_start = t_c3
    zoom_end = t_c3 + cycle_duration
    idx_s = int(zoom_start * fs)
    idx_e = int(zoom_end * fs)
    
    plt.plot(t[idx_s:idx_e], sig_clean[idx_s:idx_e], 'b', linewidth=0.8)
    plt.title("Zoom: Cycle 3 Traffic (Contrast between Slow Alice and Fast Bob)")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/04_digiton_party_mix.png')
    print("Visualization saved to 'digiton_party_mix.png'")

    # Save Audio (Clean Logic)
    # Normalize
    max_val = np.max(np.abs(sig_clean))
    if max_val > 0:
        sig_norm = sig_clean / max_val * 0.9
    
    sig_int16 = (sig_norm * 32767).astype(np.int16)
    wavfile.write('data/04_digiton_party_mix.wav', fs, sig_int16)
    print("Audio saved to 'digiton_party_mix.wav'")

if __name__ == "__main__":
    simulate_party_mix()
