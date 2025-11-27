import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

SAMPLE_RATE = 8000
CENTER_FREQ = 1500
SPIN_OFFSET = 200 # +/- 200Hz shift

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
        
        # Smooth phase transition? For single pulse, phase 0 is fine.
        return envelope * np.cos(2 * np.pi * freq * t)

    def sdr_downconvert(self, real_signal):
        """Downconvert to Baseband I/Q at CENTER_FREQ"""
        t = np.arange(len(real_signal)) / self.fs
        lo = np.exp(-1j * 2 * np.pi * CENTER_FREQ * t)
        mixed = real_signal * lo
        # Low pass filter to remove double frequency component
        sos = signal.butter(4, 500, 'low', fs=self.fs, output='sos')
        return signal.sosfilt(sos, mixed)
        
    def detect_spin(self, iq_chunk):
        """
        Analyze I/Q chunk to determine spin direction.
        Returns: 'right', 'left', or None
        """
        # Amplitude threshold
        if np.max(np.abs(iq_chunk)) < 0.01:
            return None, 0
            
        # Weighted Average Frequency
        amp = np.abs(iq_chunk)
        phase = np.unwrap(np.angle(iq_chunk))
        inst_freq = np.diff(phase) / (2 * np.pi * (1/self.fs))
        
        # Weight frequency by amplitude squared (energy)
        # We need to align amp with inst_freq (which is 1 sample shorter)
        weights = amp[1:] ** 2
        if np.sum(weights) == 0:
            return None, 0
            
        avg_freq = np.average(inst_freq, weights=weights)
        
        if avg_freq > 50: # Threshold for +200Hz
            return 'right', avg_freq
        elif avg_freq < -50: # Threshold for -200Hz
            return 'left', avg_freq
        else:
            return 'ambiguous', avg_freq

def simulate_spin_chat():
    print("--- SPIN DIGITON MODEM SIMULATION ---")
    print(f"Center Freq: {CENTER_FREQ}Hz | Spin Offset: +/-{SPIN_OFFSET}Hz")
    
    modem = SpinDigitonModem(SAMPLE_RATE)
    
    # Timeline
    duration = 4.0
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE))
    tx_signal = np.zeros_like(t)
    
    # Helper to place pulses
    def add_pulse(time_s, spin, amp=1.0):
        pulse = modem.generate_pulse(spin, sigma=0.02) # 20ms width
        start_idx = int(time_s * SAMPLE_RATE)
        end_idx = start_idx + len(pulse)
        if end_idx < len(tx_signal):
            tx_signal[start_idx:end_idx] += pulse * amp
        return start_idx, end_idx

    events = []

    # --- CHAT SEQUENCE ---
    
    # 1. Master sends CQ (Right Spin)
    print("[0.5s] MASTER: CQ (Right Spin)")
    s, e = add_pulse(0.5, 'right')
    events.append((s, e, 'MASTER (CQ)'))
    
    # 2. Alice Responds (Left Spin)
    print("[1.2s] ALICE: ACK (Left Spin)")
    s, e = add_pulse(1.2, 'left', 0.8)
    events.append((s, e, 'ALICE (ACK)'))
    
    # 3. Bob Joins (Right Spin)
    print("[1.8s] BOB: JOIN (Right Spin)")
    s, e = add_pulse(1.8, 'right', 0.6)
    events.append((s, e, 'BOB (JOIN)'))
    
    # 4. Alice sends Data "101" (Right, Left, Right)
    print("[2.5s] ALICE: DATA '1' (Right)")
    s, e = add_pulse(2.5, 'right', 0.8)
    events.append((s, e, 'ALICE (1)'))
    
    print("[2.7s] ALICE: DATA '0' (Left)")
    s, e = add_pulse(2.7, 'left', 0.8)
    events.append((s, e, 'ALICE (0)'))
    
    print("[2.9s] ALICE: DATA '1' (Right)")
    s, e = add_pulse(2.9, 'right', 0.8)
    events.append((s, e, 'ALICE (1)'))

    # --- NOISE & DECODING ---
    print("\nAdding Noise and Decoding...")
    noise_level = 0.05
    rx_signal = tx_signal + np.random.normal(0, noise_level, len(tx_signal))
    
    # Downconvert entire signal for visualization/decoding
    rx_iq = modem.sdr_downconvert(rx_signal)
    
    # Decode events
    print("\n--- DECODING RESULTS ---")
    for start, end, label in events:
        # Look at the chunk where we know the pulse is (perfect sync for demo)
        # In real life we'd use an energy detector trigger
        chunk = rx_iq[start:end]
        spin, freq = modem.detect_spin(chunk)
        print(f"Event: {label:<15} | Freq: {freq:>6.1f}Hz | Detected: {spin.upper() if spin else 'NONE'}")

    # --- SAVE WAV ---
    # Normalize for WAV
    wav_data = rx_signal / np.max(np.abs(rx_signal))
    wavfile.write('data/spin_digiton_chat.wav', SAMPLE_RATE, (wav_data * 32767).astype(np.int16))
    print("\nSaved audio to data/spin_digiton_chat.wav")

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 8))
    
    # 1. Time Domain
    plt.subplot(3, 1, 1)
    plt.plot(t, rx_signal, 'k', alpha=0.7, linewidth=0.5)
    plt.title("Spin Digiton Chat (Time Domain)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # 2. Spectrogram
    plt.subplot(3, 1, 2)
    f, t_spec, Sxx = signal.spectrogram(rx_signal, SAMPLE_RATE, nperseg=256, noverlap=128)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(1000, 2000) # Zoom in on 1500Hz +/- 500
    plt.title("Spectrogram (Notice Frequency Shifts)")
    plt.colorbar(label='dB')
    
    # 3. Instantaneous Frequency (Demodulated)
    plt.subplot(3, 1, 3)
    # Calculate freq over time for the whole signal
    # Need to smooth it or it's too noisy
    inst_phase = np.unwrap(np.angle(rx_iq))
    inst_freq = np.diff(inst_phase) / (2 * np.pi * (1/SAMPLE_RATE))
    # Filter freq for plot
    b, a = signal.butter(2, 0.05)
    freq_smooth = signal.filtfilt(b, a, inst_freq)
    
    t_freq = t[1:]
    plt.plot(t_freq, freq_smooth, 'b', linewidth=1)
    plt.axhline(SPIN_OFFSET, color='g', linestyle='--', label='Right Spin (+200)')
    plt.axhline(-SPIN_OFFSET, color='r', linestyle='--', label='Left Spin (-200)')
    plt.axhline(0, color='k', alpha=0.3)
    plt.ylim(-400, 400)
    plt.ylabel("Freq Offset (Hz)")
    plt.xlabel("Time (s)")
    plt.legend(loc='upper right')
    plt.title("Demodulated Frequency (SDR View)")
    
    plt.tight_layout()
    plt.savefig('data/spin_digiton_chat.png')
    print("Saved plot to data/spin_digiton_chat.png")

if __name__ == "__main__":
    simulate_spin_chat()
