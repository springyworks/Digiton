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
        
    def generate_pulse(self, spin='right', sigma=0.01):
        """
        Generate a single Gaussian pulse with Frequency Offset (SDR Spin).
        Right Spin (+200Hz) -> 1700Hz
        Left Spin (-200Hz) -> 1300Hz
        """
        t = np.linspace(-4*sigma, 4*sigma, int(8*sigma*self.fs))
        envelope = np.exp(-t**2 / (2 * sigma**2))
        freq = CENTER_FREQ + SPIN_OFFSET if spin == 'right' else CENTER_FREQ - SPIN_OFFSET
        return envelope * np.cos(2 * np.pi * freq * t)

def generate_burst(t, t_start, modem, data_bits, bit_duration):
    sig = np.zeros_like(t)
    current_t = t_start + bit_duration
    
    # Sigma for bit duration (make it fit)
    # If bit_duration is 5ms, sigma should be ~1ms
    sigma = bit_duration / 5.0
    
    for bit in data_bits:
        # Spin Encoding: 1=Right, 0=Left
        spin = 'right' if bit == 1 else 'left'
        pulse = modem.generate_pulse(spin, sigma=sigma)
        
        start_idx = int(current_t * SAMPLE_RATE)
        end_idx = start_idx + len(pulse)
        
        if end_idx < len(sig):
            sig[start_idx:end_idx] += pulse
            
        current_t += bit_duration
    return sig

def text_to_bits(text):
    bits = []
    for char in text:
        val = ord(char)
        for i in range(7, -1, -1):
            bits.append((val >> i) & 1)
    return bits

def simulate_adaptive_speed():
    print("--- SPIN DIGITON ADAPTIVE SPEED TEST ---")
    print("Scenario: Close Neighbor (+10dB SNR).")
    print("Goal: Complete Handshake + Message in < 4 seconds (1 Cycle).")
    print("Modulation: Spin Digiton (1=Right, 0=Left)")
    print("-" * 60)

    modem = SpinDigitonModem(SAMPLE_RATE)

    cycle_duration = 3.2 # Standard WPP Cycle
    num_slots = 8
    slot_duration = cycle_duration / num_slots # 0.4s per slot
    
    t = np.linspace(0, cycle_duration, int(cycle_duration * SAMPLE_RATE))
    signal_arr = np.zeros_like(t)
    
    # High Speed Parameters
    # 5ms pulses (200 baud)
    hs_bit_duration = 0.005 
    
    # --- SLOT 0: MASTER BEACON ---
    print("[0.0s] Slot 0: Master CQ (Standard Ping - Right Spin)")
    # Standard Ping is longer (50ms)
    pulse_cq = modem.generate_pulse('right', sigma=0.01)
    idx_cq = int(0.2 * SAMPLE_RATE)
    signal_arr[idx_cq:idx_cq+len(pulse_cq)] += pulse_cq
    
    # --- SLOT 2: ALICE REQUEST ---
    print(f"[{slot_duration*2:.2f}s] Slot 2: Alice sends REQ (Fast Burst)")
    bits_req = [1,0,1,0,1,0,1,0]
    signal_arr += generate_burst(t, slot_duration*2, modem, bits_req, hs_bit_duration)
    
    # --- SLOT 3: MASTER GRANT ---
    print(f"[{slot_duration*3:.2f}s] Slot 3: Master sends GRANT (Fast Burst)")
    bits_grant = [1,1,0,0,1,1,0,0]
    signal_arr += generate_burst(t, slot_duration*3, modem, bits_grant, hs_bit_duration)
    
    # --- SLOT 4: ALICE MESSAGE ---
    print(f"[{slot_duration*4:.2f}s] Slot 4: Alice sends 'HELLO' (Fast Burst)")
    bits_msg = text_to_bits("HELLO")
    signal_arr += generate_burst(t, slot_duration*4, modem, bits_msg, hs_bit_duration)
    
    # --- SLOT 5: MASTER ACK ---
    print(f"[{slot_duration*5:.2f}s] Slot 5: Master sends ACK")
    bits_ack = [1,1,1,1,0,0,0,0]
    signal_arr += generate_burst(t, slot_duration*5, modem, bits_ack, hs_bit_duration)

    # --- NOISE (+10dB) ---
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = signal_arr + noise

    # --- SAVE WAV ---
    wav_data = noisy_signal / np.max(np.abs(noisy_signal))
    wavfile.write('data/03_digiton_adaptive_spin.wav', SAMPLE_RATE, (wav_data * 32767).astype(np.int16))
    print("Saved audio to data/03_digiton_adaptive_spin.wav")

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, noisy_signal, 'k', linewidth=0.5, alpha=0.7)
    plt.title("Adaptive Speed Spin Digiton (Time Domain)")
    plt.ylabel("Amplitude")
    plt.xlim(0, cycle_duration)
    plt.grid(True, alpha=0.3)
    
    # Draw Slot Boundaries
    for i in range(num_slots):
        plt.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.3)

    plt.subplot(2, 1, 2)
    f, t_spec, Sxx = signal.spectrogram(noisy_signal, SAMPLE_RATE, nperseg=128, noverlap=64)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time (s)')
    plt.xlim(0, cycle_duration)
    plt.ylim(1000, 2000)
    plt.title("Spectrogram (Notice High Speed Data Bursts)")
    plt.colorbar(label='dB')
    
    plt.tight_layout()
    plt.savefig('data/digiton_adaptive_spin.png')
    print("Saved plot to data/digiton_adaptive_spin.png")
    
    print("Saved plot to data/03_digiton_adaptive_spin.png")

if __name__ == "__main__":
    simulate_adaptive_speed()
