import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def generate_digiton(t, t_center, f_hz, width_s, amplitude=1.0):
    """
    Generates a 'Digiton' - A Gaussian-enveloped sinusoid.
    """
    sigma = width_s / 6.0 
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * f_hz * t)
    return amplitude * envelope * carrier

def generate_data_burst(t, t_start, f_hz, bits, bit_duration):
    """
    Generates a burst of high-speed digitons representing bits.
    """
    signal = np.zeros_like(t)
    width = bit_duration * 0.8 # Slight gap between pulses
    
    for i, bit in enumerate(bits):
        if bit == 1:
            t_center = t_start + (i * bit_duration) + (bit_duration/2)
            signal += generate_digiton(t, t_center, f_hz, width, amplitude=0.8)
            
    return signal

def simulate_speed_negotiation():
    print("--- DIGITON SPEED NEGOTIATION SIMULATION ---")
    print("Scenario: Station ALICE has strong SNR and negotiates High Speed.")
    print("-" * 60)

    # Parameters
    fs = 48000
    cycle_duration = 1.0 
    num_slots = 8
    slot_duration = cycle_duration / num_slots
    total_cycles = 4
    total_time = total_cycles * cycle_duration
    
    t = np.linspace(0, total_time, int(total_time * fs))
    signal = np.zeros_like(t)
    
    freq_master = 800
    freq_alice = 1200 # Alice is slightly higher pitch to distinguish
    
    std_width = 0.05 # 50ms Standard Ping
    hs_bit_duration = 0.015 # 15ms per bit (High Speed)
    
    # --- CYCLE 1: Discovery ---
    print("[Cycle 1] Discovery")
    # Master CQ (Slot 0)
    signal += generate_digiton(t, 0.0 + (slot_duration * 0.5), freq_master, std_width)
    
    # Alice Joins (Slot 2) - Standard Ping
    t_alice_1 = (slot_duration * 2) + (slot_duration * 0.5)
    signal += generate_digiton(t, t_alice_1, freq_alice, std_width)
    
    # --- CYCLE 2: Negotiation / Grant ---
    print("[Cycle 2] Negotiation (Master Grants High Speed)")
    t_c2 = cycle_duration
    # Master CQ (Slot 0)
    signal += generate_digiton(t, t_c2 + (slot_duration * 0.5), freq_master, std_width)
    
    # Master sends "Speed Up" command in Alice's Slot (Slot 2)
    # Represented by a "Double Chirp" or distinctive pattern from Master
    t_grant = t_c2 + (slot_duration * 2) + (slot_duration * 0.5)
    # Master sends two quick pings to signal "Go Fast"
    signal += generate_digiton(t, t_grant - 0.02, freq_master, 0.02) 
    signal += generate_digiton(t, t_grant + 0.02, freq_master, 0.02)
    
    # Alice listens in this cycle, confirms grant (no tx or simple ack)
    # Let's say she sends a quick "Roger" (short ping) at end of slot
    signal += generate_digiton(t, t_grant + 0.08, freq_alice, 0.02, amplitude=0.5)

    # --- CYCLE 3: High Speed Data Burst ---
    print("[Cycle 3] High Speed Data (Alice sends 8 bits)")
    t_c3 = 2 * cycle_duration
    # Master CQ
    signal += generate_digiton(t, t_c3 + (slot_duration * 0.5), freq_master, std_width)
    
    # Alice sends Data Burst in Slot 2
    # Bits: 1 0 1 1 0 1 1 1
    bits = [1, 0, 1, 1, 0, 1, 1, 1]
    t_burst_start = t_c3 + (slot_duration * 2) + 0.01 # Small guard time
    signal += generate_data_burst(t, t_burst_start, freq_alice, bits, hs_bit_duration)

    # --- CYCLE 4: Sustained High Speed ---
    print("[Cycle 4] High Speed Data (Alice sends 8 bits)")
    t_c4 = 3 * cycle_duration
    # Master CQ
    signal += generate_digiton(t, t_c4 + (slot_duration * 0.5), freq_master, std_width)
    
    # Alice sends another burst
    # Bits: 1 1 0 0 1 1 0 0
    bits2 = [1, 1, 0, 0, 1, 1, 0, 0]
    t_burst_start_2 = t_c4 + (slot_duration * 2) + 0.01
    signal += generate_data_burst(t, t_burst_start_2, freq_alice, bits2, hs_bit_duration)

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 8))
    
    # Top Plot: Full Timeline
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, signal, 'k', linewidth=0.6)
    ax1.set_title("Wavelet Party Protocol: Speed Negotiation & High Speed Burst")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, total_time)
    ax1.grid(True, alpha=0.3)
    
    # Annotations
    # Cycle 1
    ax1.text(0.1, 1.1, "Cycle 1: Discovery", fontsize=10, fontweight='bold')
    ax1.text(t_alice_1, 0.8, "Alice (Std)", ha='center', color='blue')
    
    # Cycle 2
    ax1.text(cycle_duration + 0.1, 1.1, "Cycle 2: Negotiation", fontsize=10, fontweight='bold')
    ax1.text(t_grant, 0.8, "Master: SPEED UP", ha='center', color='red')
    
    # Cycle 3
    ax1.text(2*cycle_duration + 0.1, 1.1, "Cycle 3: High Speed Burst", fontsize=10, fontweight='bold')
    ax1.text(t_burst_start + 0.06, 0.8, "Alice: 10110111", ha='center', color='blue')
    
    # Draw Slot Boundaries
    for i in range(int(total_cycles * num_slots)):
        ax1.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.2)

    # Bottom Plot: Zoom in on Cycle 3 (The Burst)
    ax2 = plt.subplot(2, 1, 2)
    # Zoom window: Cycle 3 start to Cycle 3 end
    zoom_start = 2 * cycle_duration
    zoom_end = 2 * cycle_duration + (4 * slot_duration) # Show first few slots
    
    # Get indices for zoom
    idx_start = int(zoom_start * fs)
    idx_end = int(zoom_end * fs)
    
    ax2.plot(t[idx_start:idx_end], signal[idx_start:idx_end], 'b', linewidth=1.0)
    ax2.set_title("Zoom: High Speed Data Burst (Slot 2)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    
    # Annotate bits
    burst_time_center = t_burst_start + (len(bits) * hs_bit_duration) / 2
    ax2.text(burst_time_center, 0.9, "High Speed Data (8 bits / 120ms)", ha='center', fontweight='bold')
    
    # Draw bit boundaries
    for i in range(len(bits)):
        bit_t = t_burst_start + i*hs_bit_duration
        ax2.axvline(bit_t, color='r', linestyle='--', alpha=0.2)
        ax2.text(bit_t + hs_bit_duration/2, 0.5, str(bits[i]), ha='center', color='red')

    plt.tight_layout()
    plt.savefig('data/07_digiton_speed_negotiation.png')
    print("Visualization saved to 'data/07_digiton_speed_negotiation.png'")

    # Save Audio
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal_norm = signal / max_val * 0.9
    else:
        signal_norm = signal
    
    signal_int16 = (signal_norm * 32767).astype(np.int16)
    wavfile.write('data/07_digiton_speed_negotiation.wav', fs, signal_int16)
    print("Audio saved to 'data/07_digiton_speed_negotiation.wav'")

if __name__ == "__main__":
    simulate_speed_negotiation()
