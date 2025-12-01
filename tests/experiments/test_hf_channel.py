import numpy as np
from scipy.io import wavfile
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from digiton.channel.simulator import HFChannelSimulator, SAMPLE_RATE
from digiton.core.ping import WaveletPingGenerator, WaveletPingDetector, PING_FREQUENCY

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

if __name__ == "__main__":
    main()
