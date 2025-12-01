import numpy as np
import pywt
from scipy import signal

SAMPLE_RATE = 48000  # Match app audio stream rate
PING_DURATION = 0.2  # seconds, protocol-friendly ping
PING_FREQUENCY = 1500  # Hz, center of HF voice channel

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
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        # Use simple sine wave instead of wavelet for better audibility
        ping = np.sin(2 * np.pi * frequency * t)
        
        # Apply smoother envelope with slower decay for audibility
        # Use Hann window for smooth attack/release
        window = np.hanning(len(ping))
        ping = ping * window
        
        # Normalize - loud volume for audibility
        ping = ping / np.max(np.abs(ping)) * 0.98
        
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
                               wavelet_name='morl', frequency=PING_FREQUENCY,
                               padding=2.0):
        """
        Generate a sequence of pings with random start time and exact intervals
        
        Args:
            num_pings: Number of pings in sequence
            ping_interval: Time between pings (seconds) - EXACT intervals
            random_start_range: (min, max) time range for random sequence start
            wavelet_name: Wavelet type to use
            frequency: Ping frequency
            padding: Extra silence at the end (seconds)
        
        Returns:
            signal: Complete signal with ping sequence
            ping_times: Actual times when pings occur
        """
        # Random start time for the entire sequence
        start_time = np.random.uniform(*random_start_range)
        
        # Calculate total duration
        total_duration = start_time + (num_pings - 1) * ping_interval + PING_DURATION + padding
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
        Returns max correlation score, SNR estimate, and peak index
        """
        # Use FFT-based convolution for speed (O(N log N) vs O(N*M))
        # correlate(x, y) == convolve(x, y[::-1])
        correlation = signal.fftconvolve(rx_signal, self.template[::-1], mode='same')
        correlation = np.abs(correlation)
        
        peak_idx = np.argmax(correlation)
        peak_val = correlation[peak_idx]
        
        # Noise floor estimation (fast approximation)
        # Sorting is O(N log N), can be slow for large buffers.
        # Use a random sample or just mean/std if distribution is assumed.
        # Or just take a slice.
        # For robustness, let's stick to median but on a decimated version if large.
        if len(correlation) > 10000:
            # Decimate for noise floor estimation
            sample = correlation[::10]
            noise_floor = np.median(sample)
        else:
            noise_floor = np.median(correlation)
        
        # Fix for perfect silence: if noise_floor is effectively zero, 
        # we must check if the peak is also effectively zero.
        if noise_floor < 1e-9:
            if peak_val < 1e-6:
                # Both are zero -> Silence -> SNR = 0
                snr_est = 0.0
            else:
                # Peak exists but noise is zero -> Infinite SNR (or very high)
                # But in practice, this is likely a bug or synthetic signal.
                # Let's cap it or treat it as valid only if peak is substantial.
                snr_est = 1000.0 # Cap at reasonable high value
        else:
            snr_est = peak_val / noise_floor
        
        return peak_val, snr_est, correlation, peak_idx
    
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
