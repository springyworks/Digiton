import numpy as np
from scipy import signal

SAMPLE_RATE = 8000  # Hz, HF bandwidth

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
