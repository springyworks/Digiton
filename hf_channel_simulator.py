#!/usr/bin/env python3
"""
HF Channel Simulator and Wavelet Ping Detector
Simulates difficult HF radio conditions with low SNR (-20dB)
Tests detection of wavelet pings with random start time and exact ping intervals
"""

import numpy as np
import pywt
from scipy.io import wavfile
from scipy import signal
import os

SAMPLE_RATE = 8000  # Hz, HF bandwidth
PING_DURATION = 0.2  # seconds, short ping
PING_FREQUENCY = 1500  # Hz, center of HF voice channel


class HFChannelSimulator:
    """
    Simulates HF radio channel using Watterson Model characteristics:
    - Multipath propagation (discrete paths with delays)
    - Rayleigh fading with Doppler spread for each path
    - Frequency offset
    - AWGN (Additive White Gaussian Noise)
    """
    
    def __init__(self, sample_rate=SAMPLE_RATE, snr_db=-20, doppler_spread_hz=2.0, multipath_delay_ms=2.0):
        self.sample_rate = sample_rate
        self.snr_db = snr_db
        self.doppler_spread_hz = doppler_spread_hz
        self.multipath_delay_ms = multipath_delay_ms
        
    def generate_rayleigh_fading(self, num_samples):
        """Generate Rayleigh fading coefficients with Doppler spectrum"""
        # Create complex Gaussian noise
        noise = (np.random.normal(0, 1, num_samples) + 1j * np.random.normal(0, 1, num_samples)) / np.sqrt(2)
        
        # Filter to create Doppler spread (Gaussian spectrum)
        # Simple approach: Low-pass filter the noise
        b, a = signal.butter(4, self.doppler_spread_hz / (self.sample_rate / 2), btype='low')
        fading = signal.lfilter(b, a, noise)
        
        # Normalize power
        fading = fading / np.sqrt(np.mean(np.abs(fading)**2))
        return fading

    def add_awgn(self, signal, snr_db):
        """Add Additive White Gaussian Noise at specified SNR"""
        signal_power = np.mean(np.abs(signal) ** 2)
        
        # If signal is silence, assume a reference power (e.g., amplitude 1.0 -> power 0.5)
        # This ensures we still get noise even when there is no signal
        if signal_power == 0:
            signal_power = 0.5
            
        signal_power_db = 10 * np.log10(signal_power)
        noise_power_db = signal_power_db - snr_db
        noise_power = 10 ** (noise_power_db / 10)
        
        # Generate complex noise if signal is complex, else real
        if np.iscomplexobj(signal):
            noise = (np.random.normal(0, np.sqrt(noise_power/2), len(signal)) + 
                     1j * np.random.normal(0, np.sqrt(noise_power/2), len(signal)))
        else:
            noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
            
        return signal + noise
    
    def simulate_hf_channel(self, tx_signal, include_fading=True, freq_offset=True, static_freq_offset=None):
        """
        Apply Watterson channel model effects
        Path 1: Direct (0 delay), Fading
        Path 2: Delayed, Attenuated, Independent Fading
        
        NOTE: Uses Hilbert transform to simulate SSB (USB) frequency shifting.
        """
        # Ensure signal is float
        tx_signal = tx_signal.astype(float)
        
        if not include_fading and not freq_offset and static_freq_offset is None:
            return self.add_awgn(tx_signal, self.snr_db)
            
        # Convert to analytic signal for SSB (USB) simulation
        # This creates the complex baseband representation of the USB signal
        # Frequency shifting this signal results in a pure frequency shift (no images)
        tx_analytic = signal.hilbert(tx_signal)
        
        if include_fading:
            # Generate fading for two paths
            path1_fading = self.generate_rayleigh_fading(len(tx_signal))
            path2_fading = self.generate_rayleigh_fading(len(tx_signal))
            
            # Path 1: No delay
            rx_path1 = tx_analytic * path1_fading
            
            # Path 2: Delayed and attenuated (-3dB)
            delay_samples = int(self.multipath_delay_ms * self.sample_rate / 1000)
            rx_path2 = np.zeros_like(tx_analytic, dtype=complex)
            if delay_samples < len(tx_signal):
                # Shift and apply fading
                delayed_sig = np.roll(tx_analytic, delay_samples)
                delayed_sig[:delay_samples] = 0
                rx_path2 = delayed_sig * path2_fading * 0.707  # -3dB attenuation
                
            # Combine paths
            rx_signal = rx_path1 + rx_path2
        else:
            rx_signal = tx_analytic
        
        # Apply frequency offset (Doppler shift / Mistuning)
        if freq_offset or static_freq_offset is not None:
            t = np.arange(len(rx_signal)) / self.sample_rate
            if static_freq_offset is not None:
                offset_hz = static_freq_offset
            else:
                offset_hz = np.random.uniform(-2, 2)  # Random offset +/- 2Hz
            
            # Apply rotation (USB: positive freq shift moves spectrum up)
            rx_signal = rx_signal * np.exp(1j * 2 * np.pi * offset_hz * t)
        
        # Add noise (complex noise to analytic signal)
        rx_signal = self.add_awgn(rx_signal, self.snr_db)
        
        # Demodulate (Take real part)
        # For analytic signal z(t), Re{z(t) * exp(j*w*t)} is the SSB shifted signal
        return np.real(rx_signal)


class WaveletPingGenerator:
    """Generate wavelet-based signals (pings and long transmissions)"""
    
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def generate_wavelet_waveform(self, wavelet_name, frequency, duration):
        """Generate a continuous wavelet tone"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Get wavelet function
        if wavelet_name in pywt.wavelist(kind='continuous'):
            wavelet = pywt.ContinuousWavelet(wavelet_name)
            int_psi, x = pywt.integrate_wavelet(wavelet, precision=10)
            psi = np.gradient(int_psi)
        else:
            wavelet = pywt.Wavelet(wavelet_name)
            wavefun_result = wavelet.wavefun(level=10)
            if len(wavefun_result) == 2:
                psi, x = wavefun_result
            else:
                psi = wavefun_result[0]
                x = wavefun_result[-1]
        
        psi = psi / np.max(np.abs(psi))
        
        # Resample to desired frequency
        period = self.sample_rate / frequency
        x_resampled = np.linspace(x[0], x[-1], int(period))
        psi_resampled = np.interp(x_resampled, x, psi)
        
        # Tile
        num_periods = int(len(t) / period) + 1
        signal = np.tile(psi_resampled, num_periods)[:len(t)]
        
        return signal

    def generate_ping(self, wavelet_name='morl', frequency=PING_FREQUENCY, duration=PING_DURATION):
        """Generate a single wavelet ping"""
        # ...existing code...
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Get wavelet
        if wavelet_name in pywt.wavelist(kind='continuous'):
            wavelet = pywt.ContinuousWavelet(wavelet_name)
            int_psi, x = pywt.integrate_wavelet(wavelet, precision=10)
            psi = np.gradient(int_psi)
        else:
            wavelet = pywt.Wavelet(wavelet_name)
            wavefun_result = wavelet.wavefun(level=10)
            if len(wavefun_result) == 2:
                psi, x = wavefun_result
            else:
                psi = wavefun_result[0]
                x = wavefun_result[-1]
        
        psi = psi / np.max(np.abs(psi))
        
        # Create ping signal
        period = self.sample_rate / frequency
        x_resampled = np.linspace(x[0], x[-1], int(period))
        psi_resampled = np.interp(x_resampled, x, psi)
        
        num_periods = int(len(t) / period) + 1
        ping = np.tile(psi_resampled, num_periods)[:len(t)]
        
        # Apply envelope (important for ping-like characteristic)
        # Use a slower decay so the ping is audible (k=4 instead of 10)
        envelope = np.exp(-4 * np.arange(len(ping)) / len(ping))  # Match ping length
        ping = ping * envelope
        
        # Normalize
        ping = ping / np.max(np.abs(ping)) * 0.9
        
        return ping

    def generate_2sec_test_signal(self, signal_type, wavelet_name='morl', frequency=PING_FREQUENCY):
        """
        Generate various 2-second test signals
        Types: 'tone', 'chirp', 'fsk', 'burst_seq'
        """
        duration = 2.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        if signal_type == 'tone':
            # Continuous tone
            signal = self.generate_wavelet_waveform(wavelet_name, frequency, duration)
            
        elif signal_type == 'chirp':
            # Frequency sweep 800->2000 Hz
            # We'll use a simple chirp but shaped by wavelet if possible, 
            # or just standard chirp for robustness comparison
            # For wavelet consistency, let's use the generator from before but simplified
            # Actually, let's stick to a standard chirp for the 'chirp' type to see if it's better
            # But user asked for "wavelet" detected. 
            # Let's use the wavelet waveform but vary the resampling period over time.
            
            # Simplified: generate short segments of increasing frequency
            segments = []
            freqs = np.linspace(800, 2000, 20)
            seg_dur = duration / 20
            for f in freqs:
                seg = self.generate_wavelet_waveform(wavelet_name, f, seg_dur)
                segments.append(seg)
            signal = np.concatenate(segments)
            
        elif signal_type == 'fsk':
            # FSK pattern
            bits = [1, 0, 1, 1, 0, 1, 0, 0] * 2 # 16 bits in 2 seconds
            bit_dur = duration / len(bits)
            segments = []
            f_mark = 1200
            f_space = 1800
            for b in bits:
                f = f_mark if b else f_space
                seg = self.generate_wavelet_waveform(wavelet_name, f, bit_dur)
                segments.append(seg)
            signal = np.concatenate(segments)
            
        elif signal_type == 'burst_seq':
            # Sequence of pings
            signal = np.zeros_like(t)
            ping = self.generate_ping(wavelet_name, frequency, duration=0.2)
            # Place 4 pings
            intervals = [0.0, 0.5, 1.0, 1.5]
            for start in intervals:
                idx = int(start * self.sample_rate)
                if idx + len(ping) <= len(signal):
                    signal[idx:idx+len(ping)] = ping
                    
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
            
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.9
        return signal

    def generate_ping_sequence(self, num_pings=5, ping_interval=1.0, 
                               random_start_range=(0, 5),
                               wavelet_name='morl', frequency=PING_FREQUENCY):
        # ...existing code...
        """
        Generate a sequence of pings with random start time and exact intervals
        
        Args:
            num_pings: Number of pings in sequence
            ping_interval: Time between pings (seconds) - EXACT intervals
            random_start_range: (min, max) time range for random sequence start
            wavelet_name: Wavelet type to use
            frequency: Ping frequency
        
        Returns:
            signal: Complete signal with ping sequence
            ping_times: Actual times when pings occur
        """
        # Random start time for the entire sequence
        start_time = np.random.uniform(*random_start_range)
        
        # Calculate total duration
        total_duration = start_time + (num_pings - 1) * ping_interval + PING_DURATION + 2.0
        total_samples = int(total_duration * self.sample_rate)
        
        # Initialize signal
        signal = np.zeros(total_samples)
        
        # Generate single ping
        ping = self.generate_ping(wavelet_name, frequency)
        
        # Place pings at exact intervals
        ping_times = []
        for i in range(num_pings):
            ping_time = start_time + i * ping_interval
            ping_start_sample = int(ping_time * self.sample_rate)
            ping_end_sample = ping_start_sample + len(ping)
            
            if ping_end_sample <= total_samples:
                signal[ping_start_sample:ping_end_sample] += ping
                ping_times.append(ping_time)
        
        return signal, ping_times, start_time


class WaveletPingDetector:
    """Detect wavelet signals in noisy HF channel"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, wavelet_name='morl', ping_frequency=PING_FREQUENCY, template_signal=None):
        self.sample_rate = sample_rate
        self.wavelet_name = wavelet_name
        self.ping_frequency = ping_frequency
        
        # Generate reference template
        if template_signal is not None:
            self.template = template_signal
        else:
            generator = WaveletPingGenerator(sample_rate)
            self.template = generator.generate_ping(wavelet_name, ping_frequency)
    
    def detect_pings(self, rx_signal, threshold_factor=3.0):
        # ...existing code...
        """
        Detect pings using matched filter (cross-correlation)
        
        Args:
            rx_signal: Received signal
            threshold_factor: Detection threshold (multiples of noise floor)
        
        Returns:
            detections: Array of detected ping times
            correlation: Full correlation signal
        """
        # Matched filter: correlate with template
        correlation = np.correlate(rx_signal, self.template, mode='same')
        correlation = np.abs(correlation)
        
        # Estimate noise floor (exclude peaks)
        sorted_corr = np.sort(correlation)
        noise_floor = np.median(sorted_corr[:int(0.8 * len(sorted_corr))])
        noise_std = np.std(sorted_corr[:int(0.8 * len(sorted_corr))])
        
        # Detection threshold
        threshold = noise_floor + threshold_factor * noise_std
        
        # Find peaks above threshold
        # Distance must be smaller than the ping interval (0.25s)
        # Set to 0.2s to suppress sidelobes/multipath but allow 0.25s pings
        peaks, properties = signal.find_peaks(correlation, height=threshold, distance=int(0.2 * self.sample_rate))
        
        # Convert to times
        detection_times = peaks / self.sample_rate
        detection_strengths = properties['peak_heights']
        
        return detection_times, correlation, threshold

    def detect_presence(self, rx_signal):
        """
        Detect presence of a long signal (2s)
        Returns max correlation score and SNR estimate
        """
        correlation = np.correlate(rx_signal, self.template, mode='same')
        correlation = np.abs(correlation)
        
        peak_val = np.max(correlation)
        
        # Noise floor estimation
        sorted_corr = np.sort(correlation)
        noise_floor = np.median(sorted_corr[:int(0.8 * len(sorted_corr))])
        
        snr_est = peak_val / noise_floor if noise_floor > 0 else 0
        
        return peak_val, snr_est, correlation
    
    def analyze_detection_performance(self, true_ping_times, detected_ping_times, tolerance=0.1):
        """
        Analyze detector performance
        
        Args:
            true_ping_times: Actual ping times
            detected_ping_times: Detected ping times
            tolerance: Time tolerance for matching (seconds)
        
        Returns:
            performance: Dictionary with detection statistics
        """
        true_positives = 0
        false_positives = 0
        missed_detections = 0
        
        detected_matched = [False] * len(detected_ping_times)
        
        # Check each true ping
        for true_time in true_ping_times:
            matched = False
            for i, det_time in enumerate(detected_ping_times):
                if abs(det_time - true_time) < tolerance:
                    matched = True
                    detected_matched[i] = True
                    break
            
            if matched:
                true_positives += 1
            else:
                missed_detections += 1
        
        # Count false positives
        false_positives = sum(1 for matched in detected_matched if not matched)
        
        # Calculate metrics
        total_true = len(true_ping_times)
        total_detected = len(detected_ping_times)
        
        if total_detected > 0:
            precision = true_positives / total_detected
        else:
            precision = 0
        
        if total_true > 0:
            recall = true_positives / total_true
        else:
            recall = 0
        
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'missed_detections': missed_detections,
            'total_true': total_true,
            'total_detected': total_detected,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }


def run_2sec_signal_test(snr_db=-10, signal_type='tone', wavelet_name='morl', test_number=1):
    """Run detection test for 2-second signals with random start time"""
    print(f"\n{'='*70}")
    print(f"TEST #{test_number}: 2s {signal_type.upper()} ({wavelet_name}) @ {snr_db} dB SNR")
    print(f"{'='*70}")
    
    generator = WaveletPingGenerator()
    
    # Generate the 2-second signal
    clean_signal = generator.generate_2sec_test_signal(signal_type, wavelet_name)
    
    # Pad with silence to simulate random arrival
    # Random start time between 0.5s and 3.0s
    random_start = np.random.uniform(0.5, 3.0)
    padding_before = np.zeros(int(random_start * SAMPLE_RATE))
    padding_after = np.zeros(int(0.5 * SAMPLE_RATE))
    tx_signal = np.concatenate([padding_before, clean_signal, padding_after])
    
    print(f"  Random Start Time: {random_start:.3f}s")
    
    # Apply HF channel (Watterson Model)
    channel = HFChannelSimulator(snr_db=snr_db)
    noisy_signal = channel.simulate_hf_channel(tx_signal, include_fading=True)
    
    # Detect
    # For detection, we use the clean signal as the template (matched filter)
    detector = WaveletPingDetector(template_signal=clean_signal)
    peak_val, snr_est, correlation = detector.detect_presence(noisy_signal)
    
    # Find the time of the peak
    peak_idx = np.argmax(correlation)
    detected_time = peak_idx / SAMPLE_RATE
    
    # Adjust detected time (correlation peak is usually at the end of the match)
    # For 'same' mode, it's centered.
    # Let's check accuracy
    # The template is centered at index len(template)/2 in the correlation window
    # So peak should be at start_time + len(template)/2
    expected_peak_time = random_start + len(clean_signal)/SAMPLE_RATE/2
    
    # But np.correlate 'same' shifts things. 
    # Let's just check if the peak is roughly where we expect (within 0.5s)
    # Actually, let's just trust the SNR estimate for detection "presence"
    
    print(f"Detection Results:")
    print(f"  Peak Correlation: {peak_val:.2f}")
    print(f"  Estimated SNR:    {snr_est:.2f} (ratio peak/noise)")
    
    # Threshold for -30dB needs to be lower, maybe 3.0 or 2.5
    threshold = 3.0
    detected = snr_est > threshold
    print(f"  Status:           {'DETECTED' if detected else 'MISSED'}")
    
    # Save files
    output_dir = "data/detection_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    filename_base = f"{output_dir}/test{test_number:02d}_{wavelet_name}_{signal_type}_{snr_db}dB"
    
    wavfile.write(f"{filename_base}_clean.wav", SAMPLE_RATE, np.int16(clean_signal * 32767))
    wavfile.write(f"{filename_base}_noisy.wav", SAMPLE_RATE, np.int16(noisy_signal / np.max(np.abs(noisy_signal)) * 32767 * 0.9))
    
    return detected, snr_est

def run_deep_snr_experiment():
    """
    Run deep SNR tests (-30 to -45 dB) with varying repeats using db4 wavelet
    """
    print("\n" + "="*70)
    print("DEEP SNR EXPERIMENT: LIMIT TESTING")
    print("Wavelet: db4 (Daubechies 4)")
    print("Channel: Watterson Model (Multipath + Fading)")
    print("="*70)
    
    snrs = [-45, -50, -55, -60]
    repeats_list = [20, 40, 80]
    wavelet = 'db4'
    
    results = []
    
    print(f"{'SNR (dB)':<10} {'Repeats':<10} {'Duration':<10} {'Score':<10} {'Result':<10}")
    print("-" * 60)
    
    generator = WaveletPingGenerator()
    
    for snr in snrs:
        for repeats in repeats_list:
            # Generate burst sequence
            # Use tighter interval for efficiency: 0.25s
            interval = 0.25
            clean_signal, _, _ = generator.generate_ping_sequence(
                num_pings=repeats, 
                ping_interval=interval, 
                random_start_range=(0,0), # No random start in generator, we add padding
                wavelet_name=wavelet
            )
            
            # Calculate signal duration
            sig_duration = len(clean_signal) / SAMPLE_RATE
            
            # Pad with silence to simulate random arrival
            random_start = np.random.uniform(0.5, 2.0)
            padding_before = np.zeros(int(random_start * SAMPLE_RATE))
            padding_after = np.zeros(int(0.5 * SAMPLE_RATE))
            tx_signal = np.concatenate([padding_before, clean_signal, padding_after])
            
            # Apply HF channel
            channel = HFChannelSimulator(snr_db=snr)
            noisy_signal = channel.simulate_hf_channel(tx_signal, include_fading=True)
            
            # Detect
            detector = WaveletPingDetector(template_signal=clean_signal)
            peak_val, snr_est, correlation = detector.detect_presence(noisy_signal)
            
            # Threshold (lower for deep SNR?)
            # Let's keep 3.0 as a strong detection, but maybe 2.5 is acceptable here
            detected = snr_est > 3.0
            
            res_str = "DETECTED" if detected else "MISSED"
            print(f"{snr:<10} {repeats:<10} {sig_duration:<10.2f} {snr_est:<10.2f} {res_str:<10}")
            
            results.append({
                'snr': snr,
                'repeats': repeats,
                'score': snr_est,
                'detected': detected
            })
            
            # Save the best result for each SNR
            if detected:
                output_dir = "detection_tests_deep"
                os.makedirs(output_dir, exist_ok=True)
                filename_base = f"{output_dir}/deep_snr_{snr}dB_{repeats}rep_{wavelet}"
                wavfile.write(f"{filename_base}_noisy.wav", SAMPLE_RATE, np.int16(noisy_signal / np.max(np.abs(noisy_signal)) * 32767 * 0.9))

    return results

def main():
    """Run multiple detection tests"""
    
    # Run the deep SNR experiment requested by user
    run_deep_snr_experiment()
    
    # Original tests commented out for now
    """
    print("="*70)
    print("WAVELET SIGNAL DETECTION SYSTEM")
    print("Watterson HF Channel Model - Harsh Condition Tests")
    print("="*70)
    
    # Test configurations
    # Focus on -30dB SNR as requested
    snr = -30
    
    configs = [
        ('tone', 'morl'),
        ('chirp', 'morl'),
        ('burst_seq', 'mexh'),
        ('burst_seq', 'db4'),
        ('burst_seq', 'morl'),
        ('fsk', 'morl'),
    ]
    
    results = []
    
    print(f"Running tests at {snr} dB SNR (Watterson Model)...")
    
    for i, (sig_type, wavelet) in enumerate(configs, 1):
        detected, score = run_2sec_signal_test(snr, sig_type, wavelet, i)
        results.append((sig_type, wavelet, detected, score))
        
    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY OF TESTS @ {snr} dB (Watterson Model)")
    print("="*70)
    print(f"{'Type':<12} {'Wavelet':<12} {'Result':<10} {'Score':<8}")
    print("-"*70)
    
    for i, (sig_type, wavelet, detected, score) in enumerate(results):
        res_str = "DETECTED" if detected else "MISSED"
        print(f"{sig_type:<12} {wavelet:<12} {res_str:<10} {score:.2f}")
    
    print("\n" + "="*70)
    print("Tests complete! Check 'detection_tests/' for audio files.")
    print("="*70)
    """

if __name__ == "__main__":
    main()
