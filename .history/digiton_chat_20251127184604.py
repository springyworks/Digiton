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
        # 6 sigma duration for 99.7% energy
        t = np.linspace(-4*sigma, 4*sigma, int(8*sigma*self.fs))
        envelope = np.exp(-t**2 / (2 * sigma**2))
        
        freq = CENTER_FREQ + SPIN_OFFSET if spin == 'right' else CENTER_FREQ - SPIN_OFFSET
        
        return envelope * np.cos(2 * np.pi * freq * t)

def simulate_chat_sequence():
    print("--- SPIN DIGITON CHAT SIMULATION ---")
    print("Protocol: Wavelet Party Protocol (WPP)")
    print("Signal:   Spin Digiton (Gaussian with Freq Offset)")
    print("-" * 60)

    modem = SpinDigitonModem(SAMPLE_RATE)

    # Parameters
    cycle_duration = 1.0 # Speed up for demo
    num_slots = 8
    slot_duration = cycle_duration / num_slots
    total_cycles = 4
    total_time = total_cycles * cycle_duration
    
    t = np.linspace(0, total_time, int(total_time * SAMPLE_RATE))
    signal_arr = np.zeros_like(t)
    
    digiton_width = 0.02 # 20ms pulses (sharper for spin)
    
    def add_pulse(time_s, spin, amp=1.0):
        pulse = modem.generate_pulse(spin, sigma=digiton_width)
        start_idx = int(time_s * SAMPLE_RATE)
        end_idx = start_idx + len(pulse)
        if end_idx < len(signal_arr):
            signal_arr[start_idx:end_idx] += pulse * amp

    # --- CYCLE 1: Master Calls CQ ---
    print("[0.0s] Cycle 1: MASTER sends CQ (Right Spin)")
    add_pulse(0.0 + (slot_duration * 0.5), 'right')
    
    # --- CYCLE 2: Alice Joins ---
    print(f"[{cycle_duration:.1f}s] Cycle 2: MASTER sends CQ")
    add_pulse(cycle_duration + (slot_duration * 0.5), 'right')
    
    print(f"[{cycle_duration + 2*slot_duration:.1f}s] Cycle 2: ALICE responds (Left Spin)")
    add_pulse(cycle_duration + (2 * slot_duration) + (slot_duration * 0.5), 'left', 0.8)
    
    # --- CYCLE 3: Bob Joins, Alice Continues ---
    print(f"[{2*cycle_duration:.1f}s] Cycle 3: MASTER sends CQ")
    add_pulse(2*cycle_duration + (slot_duration * 0.5), 'right')
    
    print(f"[{2*cycle_duration + 2*slot_duration:.1f}s] Cycle 3: ALICE maintains")
    add_pulse(2*cycle_duration + (2 * slot_duration) + (slot_duration * 0.5), 'left', 0.8)
    
    print(f"[{2*cycle_duration + 5*slot_duration:.1f}s] Cycle 3: BOB joins (Right Spin)")
    add_pulse(2*cycle_duration + (5 * slot_duration) + (slot_duration * 0.5), 'right', 0.6)
    
    # --- CYCLE 4: Full Party (Data Exchange) ---
    print(f"[{3*cycle_duration:.1f}s] Cycle 4: MASTER sends CQ")
    add_pulse(3*cycle_duration + (slot_duration * 0.5), 'right')
    
    # Alice sends "Data" (Spin Sequence)
    print(f"[{3*cycle_duration + 2*slot_duration:.1f}s] Cycle 4: ALICE sends DATA (Right-Left)")
    t_alice_4a = 3*cycle_duration + (2 * slot_duration) + (slot_duration * 0.3)
    t_alice_4b = 3*cycle_duration + (2 * slot_duration) + (slot_duration * 0.7)
    add_pulse(t_alice_4a, 'right', 0.8)
    add_pulse(t_alice_4b, 'left', 0.8)
    
    # Bob maintains
    add_pulse(3*cycle_duration + (5 * slot_duration) + (slot_duration * 0.5), 'right', 0.6)

    # --- SAVE WAV ---
    wav_data = signal_arr / np.max(np.abs(signal_arr))
    wavfile.write('data/02_digiton_chat_spin.wav', SAMPLE_RATE, (wav_data * 32767).astype(np.int16))
    print("Saved audio to data/02_digiton_chat_spin.wav")

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, signal_arr, 'k', linewidth=0.5)
    plt.title("Spin Digiton Chat Sequence (Time Domain)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Draw Slot Boundaries
    for i in range(int(total_cycles * num_slots)):
        plt.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.3)

    plt.subplot(2, 1, 2)
    f, t_spec, Sxx = signal.spectrogram(signal_arr, SAMPLE_RATE, nperseg=256, noverlap=128)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(1000, 2000)
    plt.title("Spectrogram (Showing Spin/Frequency Shifts)")
    plt.colorbar(label='dB')
    
    plt.tight_layout()
    plt.savefig('data/digiton_chat_spin.png')
    print("Saved plot to data/digiton_chat_spin.png")

if __name__ == "__main__":
    simulate_chat_sequence()
