import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
from hf_channel_simulator import HFChannelSimulator

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_tone_pulse(t, f_hz, sigma):
    """Generate a real Gaussian pulse at f_hz"""
    envelope = np.exp(-t**2 / (2 * sigma**2))
    return envelope * np.cos(2 * np.pi * f_hz * t)

def sdr_downconvert(real_signal, fs, center_freq):
    """
    Simulate an SDR receiver:
    1. Takes Real RF/Audio signal.
    2. Mixes with complex local oscillator e^(-j*2pi*fc*t).
    3. Low-pass filters to get Baseband I/Q.
    """
    t = np.arange(len(real_signal)) / fs
    lo = np.exp(-1j * 2 * np.pi * center_freq * t)
    mixed = real_signal * lo
    
    # Low-pass filter to remove the double-frequency component
    # Cutoff at 500Hz (assuming spins are within +/- 500Hz)
    sos = signal.butter(4, 500, 'low', fs=fs, output='sos')
    baseband_iq = signal.sosfilt(sos, mixed)
    
    return baseband_iq

def run_watterson_spin_test():
    print("--- DIGITON SPIN vs WATTERSON ---")
    print("Testing 'SDR Spin' robustness in HF Channel")
    
    fs = 8000 # Standard HF Audio
    duration = 2.0
    t = np.linspace(0, duration, int(duration*fs))
    
    # Setup
    center_freq = 1500  # Virtual Center
    spin_offset = 200   # +/- 200Hz
    sigma = 0.05        # 50ms pulse (longer for robustness)
    
    # 1. GENERATE SIGNAL SEQUENCE
    # 0.5s: Right Spin (+200Hz)
    # 1.0s: Left Spin (-200Hz)
    # 1.5s: Right Spin (+200Hz)
    
    sig_clean = np.zeros_like(t)
    
    # Pulse 1 (Right)
    t1 = 0.5
    sig_clean += generate_tone_pulse(t - t1, center_freq + spin_offset, sigma)
    
    # Pulse 2 (Left)
    t2 = 1.0
    sig_clean += generate_tone_pulse(t - t2, center_freq - spin_offset, sigma)
    
    # Pulse 3 (Right)
    t3 = 1.5
    sig_clean += generate_tone_pulse(t - t3, center_freq + spin_offset, sigma)
    
    # 2. APPLY WATTERSON CHANNEL
    snr_db = -15 # Very noisy!
    print(f"Applying Watterson Channel @ {snr_db}dB SNR...")
    
    simulator = HFChannelSimulator(sample_rate=fs, snr_db=snr_db, doppler_spread_hz=1.0, multipath_delay_ms=2.0)
    
    # The simulator returns complex analytic signal (SSB), we want the Real audio output
    # But wait, simulate_hf_channel returns complex if fading is on.
    # Real radio output is Real(Analytic).
    rx_analytic = simulator.simulate_hf_channel(sig_clean, include_fading=True, freq_offset=True)
    rx_audio = np.real(rx_analytic)
    
    # 3. SDR RECEIVER
    print("Demodulating with SDR Receiver...")
    iq_rx = sdr_downconvert(rx_audio, fs, center_freq)
    
    # 4. DETECT SPIN
    # Instantaneous Frequency = d(Phase)/dt
    phase = np.unwrap(np.angle(iq_rx))
    inst_freq = np.diff(phase) / (2 * np.pi * (1/fs))
    # Smooth it
    inst_freq_smooth = signal.medfilt(inst_freq, 101)
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(12, 12))
    
    # Plot 1: Noisy Audio (Time)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, rx_audio, 'k', alpha=0.5, linewidth=0.5)
    ax1.set_title(f"1. Received Audio (SNR {snr_db}dB) - Can you see the pulses?")
    ax1.set_ylabel("Amplitude")
    
    # Plot 2: Spectrogram
    ax2 = fig.add_subplot(3, 1, 2)
    f, t_spec, Sxx = signal.spectrogram(rx_audio, fs, nperseg=256, noverlap=240)
    ax2.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax2.set_title("2. Spectrogram - The Spins are visible as Frequency Shifts")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(1000, 2000)
    ax2.axhline(center_freq, color='w', linestyle='--', alpha=0.5)
    ax2.text(0.1, center_freq+spin_offset+50, "Right (+200)", color='cyan')
    ax2.text(0.1, center_freq-spin_offset-150, "Left (-200)", color='cyan')
    
    # Plot 3: Recovered Spin (Instantaneous Frequency)
    ax3 = fig.add_subplot(3, 1, 3)
    t_freq = t[:-1]
    ax3.plot(t_freq, inst_freq_smooth, 'b', linewidth=1.5)
    ax3.set_title("3. SDR Output: Instantaneous Frequency (The 'Spin')")
    ax3.set_ylabel("Freq Offset (Hz)")
    ax3.set_ylim(-400, 400)
    ax3.axhline(spin_offset, color='g', linestyle='--', label='Right Spin Target')
    ax3.axhline(-spin_offset, color='r', linestyle='--', label='Left Spin Target')
    ax3.axhline(0, color='k', alpha=0.3)
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/16_digiton_spin_watterson.png')
    print("Visualization saved to 'data/16_digiton_spin_watterson.png'")
    
    # Save Audio
    # Normalize
    rx_norm = rx_audio / np.max(np.abs(rx_audio)) * 0.9
    wavfile.write('data/16_digiton_spin_watterson_noisy.wav', fs, (rx_norm * 32767).astype(np.int16))
    
    # Save Clean for comparison
    clean_norm = sig_clean / np.max(np.abs(sig_clean)) * 0.9
    wavfile.write('data/16_digiton_spin_clean.wav', fs, (clean_norm * 32767).astype(np.int16))
    
    print("Audio saved to 'data/16_digiton_spin_watterson_noisy.wav'")

if __name__ == "__main__":
    run_watterson_spin_test()
