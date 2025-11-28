import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
from spin_digiton_modem import SpinDigitonModem

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def generate_deep_train(modem, spin, count, interval_s):
    """
    Generates a coherent train of pulses for Deep Search Mode.
    """
    interval_samples = int(interval_s * modem.fs)
    total_samples = interval_samples * count
    train_sig = np.zeros(total_samples)
    
    pulse = modem.generate_pulse(spin=spin)
    pulse_len = len(pulse)
    
    for i in range(count):
        start_idx = i * interval_samples
        end_idx = start_idx + pulse_len
        if end_idx < total_samples:
            train_sig[start_idx:end_idx] += pulse
            
    return train_sig

def coherent_integration(signal_in, interval_samples, count):
    """
    Performs coherent integration (stacking) to pull signal out of noise.
    Assumes we know the start time (synchronized). In reality, we'd slide this window.
    """
    # Truncate to full cycles
    max_len = interval_samples * count
    if len(signal_in) < max_len:
        return np.zeros(interval_samples)
        
    sig_subset = signal_in[:max_len]
    
    # Reshape to (count, interval) and sum
    stacked = sig_subset.reshape((count, interval_samples))
    integrated = np.sum(stacked, axis=0) / count # Average to keep scale
    
    return integrated

def run_deep_ping_pong_test():
    print("--- DEEP SEARCH MODE: LOW SNR PING & PONG TEST ---")
    print("Verifying Complex Morlet Wavelet recovery at -60dB SNR")
    
    modem = SpinDigitonModem()
    fs = modem.fs
    
    # Parameters
    train_count = 1024 # Massive integration for -60dB
    pulse_interval = 0.1 # 100ms spacing
    interval_samples = int(pulse_interval * fs)
    
    # 1. Generate PING (Right Spin)
    print(f"Generating PING (Right Spin, {train_count} pulses)...")
    ping_sig = generate_deep_train(modem, 'right', train_count, pulse_interval)
    
    # 2. Generate PONG (Left Spin)
    print(f"Generating PONG (Left Spin, {train_count} pulses)...")
    pong_sig = generate_deep_train(modem, 'left', train_count, pulse_interval)
    
    # Combine into a sequence with a gap
    gap = np.zeros(int(0.5 * fs))
    full_clean_signal = np.concatenate([ping_sig, gap, pong_sig])
    
    # 3. Add Noise (-50dB)
    # -50dB => Amp = 316
    noise_amp = 316.0
    print(f"Adding AWGN Noise (Amplitude: {noise_amp}, SNR: ~-50dB)...")
    noise = np.random.normal(0, noise_amp, len(full_clean_signal))
    noisy_signal = full_clean_signal + noise
    
    # 4. Attempt Detection via Coherent Integration
    print("Attempting Recovery via Coherent Integration...")
    
    # Extract the chunks where we know the signals are (Perfect Sync Simulation)
    # In a real search, we would slide the window.
    ping_chunk = noisy_signal[0 : len(ping_sig)]
    pong_chunk = noisy_signal[len(ping_sig) + len(gap) : ]
    
    # Integrate
    integrated_ping = coherent_integration(ping_chunk, interval_samples, train_count)
    integrated_pong = coherent_integration(pong_chunk, interval_samples, train_count)
    
    # 5. Downconvert and Analyze Integrated Signals (Complex Morlet Analysis)
    # The modem's downconvert function creates the analytic signal (Complex)
    iq_ping = modem.sdr_downconvert(integrated_ping)
    iq_pong = modem.sdr_downconvert(integrated_pong)
    
    # --- MATCHED FILTER DETECTION (Complex Morlet) ---
    # Generate Reference Templates (The "Complex Morlet Wavelets")
    ref_pulse_right = modem.generate_pulse('right')
    ref_pulse_left = modem.generate_pulse('left')
    
    # Downconvert references to baseband I/Q
    ref_iq_right = modem.sdr_downconvert(ref_pulse_right)
    ref_iq_left = modem.sdr_downconvert(ref_pulse_left)
    
    # Normalize references
    ref_iq_right /= np.sum(np.abs(ref_iq_right)**2)
    ref_iq_left /= np.sum(np.abs(ref_iq_left)**2)
    
    def apply_matched_filter(signal_iq, ref_iq):
        # Correlate
        corr = np.correlate(signal_iq, ref_iq, mode='same')
        return np.max(np.abs(corr))

    # Analyze PING
    score_ping_right = apply_matched_filter(iq_ping, ref_iq_right)
    score_ping_left = apply_matched_filter(iq_ping, ref_iq_left)
    
    # Analyze PONG
    score_pong_right = apply_matched_filter(iq_pong, ref_iq_right)
    score_pong_left = apply_matched_filter(iq_pong, ref_iq_left)
    
    print("\n--- Matched Filter Results (Correlation Score) ---")
    print(f"PING Signal -> Right Score: {score_ping_right:.4f} | Left Score: {score_ping_left:.4f}")
    ping_result = "RIGHT" if score_ping_right > score_ping_left else "LEFT"
    print(f"PING Detection: {ping_result} (Expected: RIGHT)")
    
    print(f"PONG Signal -> Right Score: {score_pong_right:.4f} | Left Score: {score_pong_left:.4f}")
    pong_result = "LEFT" if score_pong_left > score_pong_right else "RIGHT"
    print(f"PONG Detection: {pong_result} (Expected: LEFT)")

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 10))
    
    pulse_len = int(0.08 * fs) # Define pulse_len for plotting
    plt.figure(figsize=(12, 10))
    
    # A. The Noisy Reality
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(noisy_signal[::10], 'k', linewidth=0.1)
    ax1.set_title("1. Raw Input Signal (-60dB SNR) - 'The Noise Floor'")
    ax1.set_ylim(-noise_amp*2, noise_amp*2)
    
    # B. Integrated PING
    ax2 = plt.subplot(4, 2, 3)
    ax2.plot(integrated_ping, 'b')
    ax2.set_title(f"2a. Integrated PING (N={train_count})")
    ax2.grid(True)
    
    # C. Integrated PONG
    ax3 = plt.subplot(4, 2, 4)
    ax3.plot(integrated_pong, 'r')
    ax3.set_title(f"2b. Integrated PONG (N={train_count})")
    ax3.grid(True)
    
    # D. I/Q Analysis (Complex Morlet) - PING
    ax4 = plt.subplot(4, 2, 5)
    ax4.plot(np.real(iq_ping[:pulse_len]), 'b', label='I')
    ax4.plot(np.imag(iq_ping[:pulse_len]), 'c', label='Q')
    ax4.set_title("3a. PING Complex Envelope (Right Spin)")
    ax4.legend()
    
    # E. I/Q Analysis (Complex Morlet) - PONG
    ax5 = plt.subplot(4, 2, 6)
    ax5.plot(np.real(iq_pong[:pulse_len]), 'r', label='I')
    ax5.plot(np.imag(iq_pong[:pulse_len]), 'm', label='Q')
    ax5.set_title("3b. PONG Complex Envelope (Left Spin)")
    ax5.legend()
    
    # F. Spectrogram of Integrated Signal (Concatenated for view)
    ax6 = plt.subplot(4, 1, 4)
    combined_integrated = np.concatenate([integrated_ping, integrated_pong])
    f, t_spec, Sxx = signal.spectrogram(combined_integrated, fs, nperseg=256, noverlap=128)
    ax6.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax6.set_title("4. Spectrogram of Recovered Signals")
    ax6.set_ylabel("Freq (Hz)")
    
    plt.tight_layout()
    plt.savefig('data/10_deep_ping_pong_test.png')
    print("Visualization saved to 'data/10_deep_ping_pong_test.png'")
    
    # Save Audio (Clean vs Noisy)
    wavfile.write('data/10_deep_ping_pong_clean.wav', fs, (full_clean_signal * 32000).astype(np.int16))
    # Normalize noisy for safety
    noisy_norm = noisy_signal / np.max(np.abs(noisy_signal)) * 0.5
    wavfile.write('data/10_deep_ping_pong_noisy.wav', fs, (noisy_norm * 32000).astype(np.int16))

if __name__ == "__main__":
    run_deep_ping_pong_test()
