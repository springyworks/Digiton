import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def generate_digiton(t, t_center, f_hz, width_s, amplitude=1.0):
    sigma = width_s / 6.0 
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * f_hz * t)
    return amplitude * envelope * carrier

def generate_burst(t, t_start, f_hz, data_bits, bit_duration):
    sig = np.zeros_like(t)
    # Guard time
    current_t = t_start + bit_duration
    
    for bit in data_bits:
        if bit == 1:
            sig += generate_digiton(t, current_t, f_hz, bit_duration*0.8)
        current_t += bit_duration
    return sig

def text_to_bits(text):
    bits = []
    for char in text:
        # Get ASCII value
        val = ord(char)
        # Convert to 8 bits
        for i in range(7, -1, -1):
            bits.append((val >> i) & 1)
    return bits

def simulate_adaptive_speed():
    print("--- DIGITON ADAPTIVE SPEED TEST ---")
    print("Scenario: Close Neighbor (+10dB SNR).")
    print("Goal: Complete Handshake + Message in < 4 seconds (1 Cycle).")
    print("-" * 60)

    fs = 48000
    cycle_duration = 3.2 # Standard WPP Cycle
    num_slots = 8
    slot_duration = cycle_duration / num_slots # 0.4s per slot
    
    print(f"Cycle Duration: {cycle_duration}s")
    print(f"Slot Duration:  {slot_duration}s")
    
    t = np.linspace(0, cycle_duration, int(cycle_duration * fs))
    signal = np.zeros_like(t)
    
    freq_master = 800
    freq_alice = 1200
    
    # High Speed Parameters
    # With +10dB, we can use short pulses.
    # Let's use 5ms pulses (200 baud).
    # 8 bits = 40ms.
    # "HELLO" = 5 bytes = 40 bits = 200ms.
    # Fits easily in a 400ms slot!
    hs_bit_duration = 0.005 
    
    # --- SLOT 0: MASTER BEACON ---
    print("[0.0s] Slot 0: Master CQ (Standard Ping)")
    signal += generate_digiton(t, 0.2, freq_master, 0.05)
    
    # --- SLOT 2: ALICE REQUEST ---
    # Alice hears CQ, picks Slot 2.
    # She is close, so she sends a "Fast Request" burst.
    # Let's say REQ is byte 0xAA (10101010)
    print(f"[{slot_duration*2:.2f}s] Slot 2: Alice sends REQ (Fast Burst)")
    bits_req = [1,0,1,0,1,0,1,0]
    signal += generate_burst(t, slot_duration*2, freq_alice, bits_req, hs_bit_duration)
    
    # --- SLOT 3: MASTER GRANT ---
    # Master hears Alice in Slot 2.
    # Master replies immediately in Slot 3 (Fast Response).
    # GRANT = 0xCC
    print(f"[{slot_duration*3:.2f}s] Slot 3: Master sends GRANT (Fast Burst)")
    bits_grant = [1,1,0,0,1,1,0,0]
    signal += generate_burst(t, slot_duration*3, freq_master, bits_grant, hs_bit_duration)
    
    # --- SLOT 4: ALICE MESSAGE ---
    # Alice gets Grant. Sends "HELLO".
    print(f"[{slot_duration*4:.2f}s] Slot 4: Alice sends 'HELLO' (Fast Burst)")
    bits_msg = text_to_bits("HELLO")
    signal += generate_burst(t, slot_duration*4, freq_alice, bits_msg, hs_bit_duration)
    
    # --- SLOT 5: MASTER ACK ---
    # Master receives HELLO. Sends ACK.
    print(f"[{slot_duration*5:.2f}s] Slot 5: Master sends ACK")
    bits_ack = [1,1,1,1,0,0,0,0]
    signal += generate_burst(t, slot_duration*5, freq_master, bits_ack, hs_bit_duration)

    # --- NOISE (+10dB) ---
    # Signal amp is 1.0. Power 0.5.
    # +10dB SNR -> Noise Power = 0.05.
    # Noise Amp = sqrt(0.05) = 0.22.
    noise = np.random.normal(0, 0.22, len(t))
    noisy_signal = signal + noise

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 6))
    
    # Plot
    plt.plot(t, noisy_signal, 'k', linewidth=0.5, alpha=0.8)
    plt.title("Adaptive Speed: Close Neighbor (+10dB) - Full Conversation in 1 Cycle (3.2s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(-2, 2)
    plt.grid(True, alpha=0.3)
    
    # Draw Slots
    for i in range(num_slots + 1):
        plt.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.5)
        if i < num_slots:
            plt.text(i * slot_duration + 0.05, 1.8, f"SLOT {i}", color='green', fontsize=8)

    # Annotations
    plt.text(0.2, 1.2, "MASTER CQ", ha='center', color='blue', fontweight='bold')
    plt.text(slot_duration*2 + 0.2, 1.2, "ALICE: REQ", ha='center', color='red', fontweight='bold')
    plt.text(slot_duration*3 + 0.2, 1.2, "MASTER: GRANT", ha='center', color='blue', fontweight='bold')
    plt.text(slot_duration*4 + 0.2, 1.2, "ALICE: 'HELLO'", ha='center', color='red', fontweight='bold')
    plt.text(slot_duration*5 + 0.2, 1.2, "MASTER: ACK", ha='center', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig('data/03_digiton_adaptive_speed.png')
    print("Visualization saved to 'data/03_digiton_adaptive_speed.png'")

    # Save Audio
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 0:
        signal_norm = noisy_signal / max_val * 0.9
    else:
        signal_norm = noisy_signal
    
    signal_int16 = (signal_norm * 32767).astype(np.int16)
    wavfile.write('data/03_digiton_adaptive_speed.wav', fs, signal_int16)
    print("Audio saved to 'data/03_digiton_adaptive_speed.wav'")

if __name__ == "__main__":
    simulate_adaptive_speed()
