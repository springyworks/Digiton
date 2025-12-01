import numpy as np
from scipy import signal

def calculate_link_budget(snr_input_db, target_snr_db=10.0):
    """
    Calculates the required integration time for a given input SNR.
    """
    print(f"--- DIGITON LINK BUDGET CALCULATOR ---")
    print(f"Input SNR: {snr_input_db} dB")
    print(f"Target SNR (Detector): {target_snr_db} dB")
    print(f"Required Gain: {target_snr_db - snr_input_db} dB")
    print("-" * 40)
    
    # 1. Pulse Compression Gain (Matched Filter)
    # Gain = 10 * log10(Time * Bandwidth)
    # For a Gaussian pulse, Time*BW is small (~1), so this formula is for Chirps.
    # For a simple pulse in noise, the Matched Filter Gain is 10*log10(Energy/NoisePSD).
    # Effectively, it's 10*log10(N_samples) if noise is white and bandwidth matches sample rate.
    # But noise is usually defined in the bandwidth of the signal.
    # Let's use the "Processing Gain" approach:
    # Gain = 10 * log10(SampleRate * PulseDuration)
    # Assuming noise is broadband across SampleRate.
    
    fs = 48000
    pulse_duration = 0.05 # 50ms
    n_samples = int(fs * pulse_duration)
    
    # If noise is defined as "Peak Amplitude Ratio" (which is how we generated it: amp 1 vs amp 300),
    # The "SNR" is -50dB.
    # The Matched Filter sums the signal energy coherently.
    # Signal Energy E = Sum(s^2) ~ N/2.
    # Noise Power out = N * sigma^2.
    # SNR_out = E / sigma^2 = (N/2) / sigma^2.
    # SNR_in (power) = 0.5 / sigma^2.
    # So SNR_out = N * SNR_in.
    # Gain = 10 * log10(N).
    
    pc_gain_db = 10 * np.log10(n_samples)
    print(f"Pulse Duration: {pulse_duration*1000} ms ({n_samples} samples)")
    print(f"Pulse Compression Gain: {pc_gain_db:.2f} dB")
    
    remaining_gain = (target_snr_db - snr_input_db) - pc_gain_db
    print(f"Remaining Gain Needed: {remaining_gain:.2f} dB")
    
    if remaining_gain <= 0:
        print("Status: DETECTABLE with Single Pulse!")
        repeats = 1
    else:
        # 2. Coherent Integration Gain
        # Gain = 10 * log10(Repeats)
        repeats = 10 ** (remaining_gain / 10)
        repeats = int(np.ceil(repeats))
        print(f"Status: Needs Integration")
        print(f"Required Repeats: {repeats}")
        print(f"Integration Gain: {10*np.log10(repeats):.2f} dB")
        
    total_time = repeats * pulse_duration
    print(f"Total Transmission Time: {total_time:.3f} s")
    
    # Data Rate
    # 1 bit per sequence (Ping vs Silence, or Left vs Right)
    bps = 1.0 / total_time
    print(f"Effective Data Rate: {bps:.4f} bps")
    print("-" * 40)
    return repeats, bps

if __name__ == "__main__":
    # Example: Deep Search Mode (-50dB)
    calculate_link_budget(-50.0)
    
    # Example: Turbo Mode (+10dB)
    calculate_link_budget(10.0)
