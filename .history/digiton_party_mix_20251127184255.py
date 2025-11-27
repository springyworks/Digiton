import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

SAMPLE_RATE = 8000
CENTER_FREQ = 1500
SPIN_OFFSET = 200

class SpinDigitonModem:
    def __init__(self, fs=SAMPLE_RATE):
        self.fs = fs
        
    def generate_pulse(self, spin='right', sigma=0.01, amplitude=1.0):
        t = np.linspace(-4*sigma, 4*sigma, int(8*sigma*self.fs))
        envelope = np.exp(-t**2 / (2 * sigma**2))
        freq = CENTER_FREQ + SPIN_OFFSET if spin == 'right' else CENTER_FREQ - SPIN_OFFSET
        return amplitude * envelope * np.cos(2 * np.pi * freq * t)

def generate_burst(t, t_start, modem, data_bits, bit_duration, amplitude=1.0):
    sig = np.zeros_like(t)
    current_t = t_start
    sigma = bit_duration / 5.0
    
    for bit in data_bits:
        spin = 'right' if bit == 1 else 'left'
        pulse = modem.generate_pulse(spin, sigma=sigma, amplitude=amplitude)
        
        start_idx = int((current_t + bit_duration/2) * SAMPLE_RATE)
        # Center the pulse in the bit slot
        start_idx -= len(pulse) // 2
        
        if start_idx >= 0 and start_idx + len(pulse) < len(sig):
            sig[start_idx:start_idx+len(pulse)] += pulse
            
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
    print("--- SPIN DIGITON PARTY MIX SIMULATION ---")
    print("Scenario: Mixed Speed Party with Spin Digiton Modem.")
    print("  1. ALICE (Far/Weak): -10dB SNR. Needs SLOW mode (Standard Pulses).")
    print("  2. BOB (Close/Strong): +20dB SNR. Uses FAST mode (Spin Bursts).")
    print("-" * 60)

    modem = SpinDigitonModem(SAMPLE_RATE)

    cycle_duration = 3.2 
    num_slots = 8
    slot_duration = cycle_duration / num_slots # 0.4s
    
    total_cycles = 3
    total_time = total_cycles * cycle_duration
    
    t = np.linspace(0, total_time, int(total_time * SAMPLE_RATE))
    sig_clean = np.zeros_like(t)
    
    # --- CYCLE 1: MASTER BEACON ---
    print("[0.0s] Cycle 1: Master CQ (Right Spin)")
    pulse_cq = modem.generate_pulse('right', sigma=0.02)
    idx_cq = int(0.2 * SAMPLE_RATE)
    sig_clean[idx_cq:idx_cq+len(pulse_cq)] += pulse_cq
    
    # --- ALICE (SLOW) in Slot 2 ---
    # Alice is weak (0.3 amplitude). She sends a standard pulse (Right Spin) to join.
    print(f"[{slot_duration*2:.2f}s] Slot 2: Alice (Weak) Joins (Standard Pulse)")
    pulse_alice = modem.generate_pulse('left', sigma=0.02, amplitude=0.3)
    idx_alice = int((slot_duration*2 + 0.2) * SAMPLE_RATE)
    sig_clean[idx_alice:idx_alice+len(pulse_alice)] += pulse_alice
    
    # --- BOB (FAST) in Slot 5 ---
    # Bob is strong (1.0). He sends a Fast Burst to join.
    print(f"[{slot_duration*5:.2f}s] Slot 5: Bob (Strong) Joins (Fast Burst)")
    bits_bob_join = [1,0,1,0,1,0,1,0] # 0xAA
    sig_clean += generate_burst(t, slot_duration*5, modem, bits_bob_join, 0.005, amplitude=1.0)
    
    # --- CYCLE 2: MASTER GRANTS ---
    print(f"[{cycle_duration:.2f}s] Cycle 2: Master CQ")
    idx_cq2 = int((cycle_duration + 0.2) * SAMPLE_RATE)
    sig_clean[idx_cq2:idx_cq2+len(pulse_cq)] += pulse_cq
    
    # Master Grants Alice (Slow) in Slot 2
    # Master Grants Bob (Fast) in Slot 5
    
    # --- ALICE SENDS DATA (SLOW) ---
    # Alice sends 1 bit per slot? Or just standard pulses.
    # Let's say she sends "1" (Right Spin) in Slot 2.
    print(f"[{cycle_duration + slot_duration*2:.2f}s] Slot 2: Alice sends Data '1' (Slow)")
    pulse_alice_data = modem.generate_pulse('right', sigma=0.02, amplitude=0.3)
    idx_alice_d = int((cycle_duration + slot_duration*2 + 0.2) * SAMPLE_RATE)
    sig_clean[idx_alice_d:idx_alice_d+len(pulse_alice_data)] += pulse_alice_data
    
    # --- BOB SENDS DATA (FAST) ---
    # Bob sends "HI" in Slot 5.
    print(f"[{cycle_duration + slot_duration*5:.2f}s] Slot 5: Bob sends 'HI' (Fast)")
    bits_hi = text_to_bits("HI")
    sig_clean += generate_burst(t, cycle_duration + slot_duration*5, modem, bits_hi, 0.005, amplitude=1.0)

    # --- NOISE ---
    # We add noise relative to the weak signal.
    # Alice is 0.3. -10dB means Noise Power is 10x Signal Power.
    # Signal Power ~ 0.3^2 = 0.09.
    # Noise Power = 0.9.
    # Noise Amp = sqrt(0.9) = 0.95.
    # This will bury Alice but Bob (1.0) will be visible (SNR ~ 0dB).
    # Let's be kinder for the demo so we can see Alice.
    # Let's say Noise is 0.1 (Alice SNR ~ 10dB, Bob SNR ~ 20dB).
    noise = np.random.normal(0, 0.1, len(t))
    sig_noisy = sig_clean + noise

    # --- SAVE WAV ---
    wav_data = sig_noisy / np.max(np.abs(sig_noisy))
    wavfile.write('data/digiton_party_mix_spin.wav', SAMPLE_RATE, (wav_data * 32767).astype(np.int16))
    print("Saved audio to data/digiton_party_mix_spin.wav")

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, sig_noisy, 'k', linewidth=0.5, alpha=0.7)
    plt.title("Party Mix: Weak/Slow Alice vs Strong/Fast Bob")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Draw Slot Boundaries
    for i in range(int(total_cycles * num_slots)):
        plt.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.3)

    plt.subplot(2, 1, 2)
    f, t_spec, Sxx = signal.spectrogram(sig_noisy, SAMPLE_RATE, nperseg=256, noverlap=128)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(1000, 2000)
    plt.title("Spectrogram (Notice Alice's Faint Pulses vs Bob's Bright Bursts)")
    plt.colorbar(label='dB')
    
    plt.tight_layout()
    plt.savefig('data/digiton_party_mix_spin.png')
    print("Saved plot to data/digiton_party_mix_spin.png")

if __name__ == "__main__":
    simulate_party_mix()
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
    print("Visualization saved to 'data/04_digiton_party_mix.png'")

    # Save Audio (Clean Logic)
    # Normalize
    max_val = np.max(np.abs(sig_clean))
    if max_val > 0:
        sig_norm = sig_clean / max_val * 0.9
    
    sig_int16 = (sig_norm * 32767).astype(np.int16)
    wavfile.write('data/04_digiton_party_mix.wav', fs, sig_int16)
    print("Audio saved to 'data/04_digiton_party_mix.wav'")

if __name__ == "__main__":
    simulate_party_mix()
