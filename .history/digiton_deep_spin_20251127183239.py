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
    
    # Low-pass filter
    sos = signal.butter(4, 500, 'low', fs=fs, output='sos')
    baseband_iq = signal.sosfilt(sos, mixed)
    
    return baseband_iq

def run_deep_spin_test():
    print("--- DIGITON DEEP SPIN (-60dB) ---")
    print("Combining 'Deep Mode' (Stacking) with 'SDR Spin'")
    
    fs = 8000
    pulse_len_s = 0.5
    pulse_interval_s = 1.0 # 1 second between pulses
    num_repeats = 1024     # Massive repetition for -60dB
    
    # Setup
    center_freq = 1500
    spin_offset = 200
    sigma = 0.05
    
    # Create ONE pulse template (Right Spin)
    t_pulse = np.linspace(-pulse_len_s/2, pulse_len_s/2, int(pulse_len_s*fs))
    pulse_right = generate_tone_pulse(t_pulse, center_freq + spin_offset, sigma)
    pulse_left = generate_tone_pulse(t_pulse, center_freq - spin_offset, sigma)
    
    # Generate the full sequence (Right Spin repeated N times)
    print(f"Generating {num_repeats} repeats of Right Spin...")
    total_duration = num_repeats * pulse_interval_s
    t_full = np.linspace(0, total_duration, int(total_duration*fs))
    sig_clean = np.zeros_like(t_full)
    
    # Place pulses
    samples_per_interval = int(pulse_interval_s * fs)
    pulse_samples = len(pulse_right)
    
    for i in range(num_repeats):
        start_idx = i * samples_per_interval
        # Center the pulse in the interval
        offset = int((samples_per_interval - pulse_samples) / 2)
        idx = start_idx + offset
        if idx + pulse_samples <= len(sig_clean):
            sig_clean[idx:idx+pulse_samples] = pulse_right
            
    # APPLY WATTERSON CHANNEL @ -60dB
    snr_db = -60
    print(f"Applying Watterson Channel @ {snr_db}dB SNR...")
    simulator = HFChannelSimulator(sample_rate=fs, snr_db=snr_db, doppler_spread_hz=0.5, multipath_delay_ms=2.0)
    
    # Simulate
    rx_analytic = simulator.simulate_hf_channel(sig_clean, include_fading=True, freq_offset=True)
    rx_audio = np.real(rx_analytic)
    
    # --- RECEIVER: COHERENT INTEGRATION ---
    print("Receiver: Stacking pulses...")
    
    # We assume we know the period (pulse_interval_s)
    # We stack the received audio into a single interval buffer
    stacked_audio = np.zeros(samples_per_interval)
    
    for i in range(num_repeats):
        start_idx = i * samples_per_interval
        end_idx = start_idx + samples_per_interval
        if end_idx <= len(rx_audio):
            chunk = rx_audio[start_idx:end_idx]
            stacked_audio += chunk
            
    # Normalize stacked audio
    stacked_audio /= num_repeats
    
    # --- SDR DECODING OF STACKED SIGNAL ---
    print("Receiver: Demodulating Stacked Signal...")
    iq_stacked = sdr_downconvert(stacked_audio, fs, center_freq)
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(12, 12))
    
    # 1. Raw Noise (Zoomed)
    ax1 = fig.add_subplot(3, 1, 1)
    zoom_s = 2.0
    zoom_samples = int(zoom_s * fs)
    ax1.plot(t_full[:zoom_samples], rx_audio[:zoom_samples], 'k', alpha=0.5, linewidth=0.5)
    ax1.set_title(f"1. Raw Input (-60dB) - First {zoom_s}s\nPure Noise. Signal is invisible.")
    
    # 2. Stacked Audio (Time)
    ax2 = fig.add_subplot(3, 1, 2)
    t_stack = np.linspace(0, pulse_interval_s, samples_per_interval)
    ax2.plot(t_stack, stacked_audio, 'b')
    ax2.set_title(f"2. Coherent Integration (N={num_repeats})\nThe Pulse emerges from the noise!")
    ax2.set_xlabel("Time (s)")
    
    # 3. SDR Spin Analysis of Stacked Signal
    ax3 = fig.add_subplot(3, 1, 3)
    # Calculate Instantaneous Frequency of the stacked pulse
    phase = np.unwrap(np.angle(iq_stacked))
    inst_freq = np.diff(phase) / (2 * np.pi * (1/fs))
    inst_freq_smooth = signal.medfilt(inst_freq, 51)
    
    t_freq = t_stack[:-1]
    # Only look at the center where the pulse is
    center_mask = (t_freq > 0.4) & (t_freq < 0.6)
    
    ax3.plot(t_freq, inst_freq_smooth, 'g', linewidth=1.5)
    ax3.set_title("3. SDR Spin Detection on Stacked Signal\nDoes it show +200Hz?")
    ax3.set_ylabel("Freq Offset (Hz)")
    ax3.set_ylim(-400, 400)
    ax3.axhline(spin_offset, color='g', linestyle='--', label='Target (+200Hz)')
    ax3.axhline(-spin_offset, color='r', linestyle='--', label='Opposite (-200Hz)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/17_digiton_deep_spin.png')
    print("Visualization saved to 'data/17_digiton_deep_spin.png'")
    
    # Save Audio
    # 1. The Noise (First 5 seconds)
    wavfile.write('data/17_digiton_deep_noise_sample.wav', fs, (rx_audio[:5*fs] / np.max(np.abs(rx_audio)) * 32767).astype(np.int16))
    
    # 2. The Recovered Stacked Pulse
    wavfile.write('data/17_digiton_deep_recovered.wav', fs, (stacked_audio / np.max(np.abs(stacked_audio)) * 32767).astype(np.int16))
    print("Audio saved.")

if __name__ == "__main__":
    run_deep_spin_test()
