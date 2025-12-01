import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import sys

# Add project root to path to allow importing digiton package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from digiton.core.modem import SpinDigitonModem, SAMPLE_RATE, CENTER_FREQ, SPIN_OFFSET

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

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
    
    # 1. Initiator sends WWBEAT (Right Spin)
    print("[0.5s] INITIATOR: WWBEAT (Right Spin)")
    s, e = add_pulse(0.5, 'right')
    events.append((s, e, 'INITIATOR (WWBEAT)'))
    
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
    wavfile.write('data/01_spin_digiton_modem.wav', SAMPLE_RATE, (wav_data * 32767).astype(np.int16))
    print("\nSaved audio to data/01_spin_digiton_modem.wav")

    # --- VISUALIZATION ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # 1. Time Domain
    ax1.plot(t, rx_signal, 'k', alpha=0.7, linewidth=0.5)
    ax1.set_title("Spin Digiton Chat (Time Domain)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, duration)
    ax1.grid(True, alpha=0.3)
    
    # 2. Spectrogram
    f, t_spec, Sxx = signal.spectrogram(rx_signal, SAMPLE_RATE, nperseg=256, noverlap=128)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.xlim(0, duration)
    plt.ylim(1000, 2000) # Zoom in on 1500Hz +/- 500
    plt.title("Spectrogram (Notice Frequency Shifts)")
    plt.colorbar(label='dB')
    
    # 3. Instantaneous Frequency (Demodulated)
    # Calculate freq over time for the whole signal
    # Need to smooth it or it's too noisy
    inst_phase = np.unwrap(np.angle(rx_iq))
    inst_freq = np.diff(inst_phase) / (2 * np.pi * (1/SAMPLE_RATE))
    # Filter freq for plot
    b, a = signal.butter(2, 0.05)
    freq_smooth = signal.filtfilt(b, a, inst_freq)
    
    t_freq = t[1:]
    ax3.plot(t_freq, freq_smooth, 'b', linewidth=1)
    ax3.axhline(SPIN_OFFSET, color='g', linestyle='--', label='Right Spin (+200)')
    ax3.axhline(-SPIN_OFFSET, color='r', linestyle='--', label='Left Spin (-200)')
    ax3.axhline(0, color='k', alpha=0.3)
    ax3.set_ylim(-400, 400)
    ax3.set_ylabel("Freq Offset (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper right')
    ax3.set_title("Demodulated Frequency (SDR View)")
    
    plt.tight_layout()
    plt.savefig('data/01_spin_digiton_modem.png')
    print("Saved plot to data/01_spin_digiton_modem.png")

if __name__ == "__main__":
    simulate_spin_chat()
