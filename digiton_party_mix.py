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
    wavfile.write('data/04_digiton_party_mix_spin.wav', SAMPLE_RATE, (wav_data * 32767).astype(np.int16))
    print("Saved audio to data/04_digiton_party_mix_spin.wav")

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1.plot(t, sig_noisy, 'k', linewidth=0.5, alpha=0.7)
    ax1.set_title("Party Mix: Weak/Slow Alice vs Strong/Fast Bob")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, total_time)
    ax1.grid(True, alpha=0.3)
    
    # Draw Slot Boundaries
    for i in range(int(total_cycles * num_slots)):
        ax1.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.3)

    f, t_spec, Sxx = signal.spectrogram(sig_noisy, SAMPLE_RATE, nperseg=256, noverlap=128)
    pcm = ax2.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(1000, 2000)
    ax2.set_title("Spectrogram (Notice Alice's Faint Pulses vs Bob's Bright Bursts)")
    plt.colorbar(pcm, ax=ax2, label='dB')
    ax2.set_title("Spectrogram (Notice Alice's Faint Pulses vs Bob's Bright Bursts)")
    plt.colorbar(pcm, ax=ax2, label='dB')
    
    plt.tight_layout()
    plt.savefig('data/04_digiton_party_mix_spin.png')
    print("Saved plot to data/04_digiton_party_mix_spin.png")

if __name__ == "__main__":
    simulate_party_mix()


