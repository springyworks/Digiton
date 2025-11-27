import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
from hf_channel_simulator import HFChannelSimulator

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

SAMPLE_RATE = 8000
CENTER_FREQ = 1500
SPIN_OFFSET = 200

class SpinDigitonGenerator:
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

    def generate_sequence(self, spin='right', repeats=1, interval=0.2):
        """Generate a sequence of pulses for Deep Mode"""
        pulse = self.generate_pulse(spin)
        total_samples = int(repeats * interval * self.fs)
        sig = np.zeros(total_samples)
        
        samples_per_interval = int(interval * self.fs)
        pulse_len = len(pulse)
        
        for i in range(repeats):
            start = i * samples_per_interval
            # Center it
            offset = (samples_per_interval - pulse_len) // 2
            idx = start + offset
            if idx + pulse_len <= total_samples:
                sig[idx:idx+pulse_len] = pulse
                
        return sig

class SpinDigitonDetector:
    def __init__(self, fs=SAMPLE_RATE):
        self.fs = fs
        
    def sdr_downconvert(self, real_signal):
        """Downconvert to Baseband I/Q at CENTER_FREQ"""
        t = np.arange(len(real_signal)) / self.fs
        lo = np.exp(-1j * 2 * np.pi * CENTER_FREQ * t)
        mixed = real_signal * lo
        sos = signal.butter(4, 500, 'low', fs=self.fs, output='sos')
        return signal.sosfilt(sos, mixed)
        
    def detect_spin(self, rx_signal, repeats=1, interval=0.2):
        """
        Detect Spin using Coherent Integration + SDR Frequency Analysis.
        Returns: 'right', 'left', or None
        """
        # 1. Coherent Integration (Stacking)
        samples_per_interval = int(interval * self.fs)
        stacked = np.zeros(samples_per_interval)
        
        # Truncate to full intervals
        num_intervals = len(rx_signal) // samples_per_interval
        if num_intervals < repeats:
            # Not enough data
            return None, 0
            
        for i in range(repeats):
            start = i * samples_per_interval
            end = start + samples_per_interval
            stacked += rx_signal[start:end]
            
        stacked /= repeats
        
        # 2. SDR Downconvert
        iq = self.sdr_downconvert(stacked)
        
        # 3. Instantaneous Frequency
        # Only look at the center (where the pulse is expected)
        center_idx = samples_per_interval // 2
        window = int(0.05 * self.fs) # 50ms window
        iq_pulse = iq[center_idx-window : center_idx+window]
        
        if len(iq_pulse) == 0: return None, 0
        
        phase = np.unwrap(np.angle(iq_pulse))
        inst_freq = np.diff(phase) / (2 * np.pi * (1/self.fs))
        mean_freq = np.median(inst_freq)
        
        # 4. Decision
        threshold = 50 # Hz tolerance
        
        if mean_freq > SPIN_OFFSET - threshold:
            return 'right', mean_freq
        elif mean_freq < -SPIN_OFFSET + threshold:
            return 'left', mean_freq
        else:
            return None, mean_freq

class AdaptiveThrottle:
    """Manages the Speed vs Robustness trade-off"""
    def __init__(self):
        self.mode = 'FAST' # FAST or DEEP
        self.repeats = 1
        self.snr_history = []
        
    def update(self, measured_snr_db):
        self.snr_history.append(measured_snr_db)
        if len(self.snr_history) > 5:
            self.snr_history.pop(0)
            
        avg_snr = np.mean(self.snr_history)
        
        if self.mode == 'FAST':
            if avg_snr < 0: # Drop to Deep Mode if SNR < 0dB
                print(f"Throttle: SNR dropped to {avg_snr:.1f}dB. Engaging DEEP MODE.")
                self.mode = 'DEEP'
                self.repeats = 16 # Start with 16x
        elif self.mode == 'DEEP':
            if avg_snr > 10: # Return to Fast Mode if SNR > 10dB
                print(f"Throttle: SNR improved to {avg_snr:.1f}dB. Engaging FAST MODE.")
                self.mode = 'FAST'
                self.repeats = 1
            elif avg_snr < -10:
                # Go deeper
                self.repeats = min(self.repeats * 2, 256)
                
    def get_params(self):
        return self.mode, self.repeats

def run_protocol_simulation():
    print("--- DIGITON PROTOCOL V2: ADAPTIVE SPIN ---")
    
    # Setup
    gen = SpinDigitonGenerator()
    det = SpinDigitonDetector()
    throttle = AdaptiveThrottle()
    
    # Simulation Buffer
    full_audio = []
    
    # 1. FAST MODE PHASE (High SNR)
    print("\n[PHASE 1] High SNR (+10dB) - Fast Mode")
    throttle.update(15.0) # Force Fast
    mode, repeats = throttle.get_params()
    
    # Generate "Hello" (Right, Left, Right, Right, Left)
    sequence = ['right', 'left', 'right', 'right', 'left']
    
    for spin in sequence:
        sig = gen.generate_sequence(spin, repeats=repeats)
        # Add noise
        noisy = sig + np.random.normal(0, 0.1, len(sig))
        full_audio.append(noisy)
        
        # Detect
        detected, freq = det.detect_spin(noisy, repeats=repeats)
        print(f"  Sent: {spin.upper()} | Rx Freq: {freq:.1f}Hz | Detected: {str(detected).upper()}")
        
    # 2. DEEP MODE PHASE (Low SNR)
    print("\n[PHASE 2] Low SNR (-20dB) - Deep Mode")
    throttle.update(-20.0) # Force Deep
    mode, repeats = throttle.get_params()
    print(f"  Throttle set repeats to: {repeats}")
    
    # Generate "SOS" (Right, Right, Right)
    sequence = ['right', 'right', 'right']
    
    for spin in sequence:
        sig = gen.generate_sequence(spin, repeats=repeats)
        # Add HEAVY noise (-20dB is amplitude 0.1 vs noise 1.0)
        # Signal amp is ~1.0. Noise needs to be 10.0
        noise_amp = 10.0
        noisy = sig + np.random.normal(0, noise_amp, len(sig))
        full_audio.append(noisy)
        
        # Detect
        detected, freq = det.detect_spin(noisy, repeats=repeats)
        print(f"  Sent: {spin.upper()} | Rx Freq: {freq:.1f}Hz | Detected: {str(detected).upper()}")

    # Save Full Audio
    full_audio_concat = np.concatenate(full_audio)
    # Normalize
    full_audio_norm = full_audio_concat / np.max(np.abs(full_audio_concat)) * 0.9
    wavfile.write('data/18_digiton_protocol_v2.wav', SAMPLE_RATE, (full_audio_norm * 32767).astype(np.int16))
    print("\nAudio saved to 'data/18_digiton_protocol_v2.wav'")

if __name__ == "__main__":
    run_protocol_simulation()
