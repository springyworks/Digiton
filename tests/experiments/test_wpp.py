import numpy as np
import scipy.io.wavfile as wav
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from digiton.channel.simulator import HFChannelSimulator, SAMPLE_RATE
from digiton.protocol.wpp import InitiatorStation, ResponderStation, CQ_INTERVAL, SLOT_DURATION

def run_deep_snr_test():
    print("--- WAVELET PARTY PROTOCOL: DEEP SNR TEST ---")
    
    # Test Cases: (SNR, Repeats)
    # -20dB: 1 repeat (Standard)
    # -30dB: 4 repeats
    # -40dB: 16 repeats
    # -50dB: 64 repeats
    
    test_cases = [
        (-20, 1),
        (-30, 4),
        (-40, 16),
        (-50, 64),
        (-60, 256)
    ]
    
    for snr, repeats in test_cases:
        print(f"\n" + "="*50)
        print(f"TESTING SNR: {snr} dB | REPEATS: {repeats}")
        print("="*50)
        
        initiator = InitiatorStation("INITIATOR", repeats=repeats)
        responder = ResponderStation("RESPONDER", snr_profile=snr, repeats=repeats)
        
        # Channels
        chan_i2r = HFChannelSimulator(snr_db=snr)
        chan_r2i = HFChannelSimulator(snr_db=snr)
        
        # Run for 1 Cycle (Enough to Sync and Reply)
        # Cycle = CQ_INTERVAL slots
        # We run for CQ_INTERVAL + 4 slots to ensure we catch the reply
        total_slots = CQ_INTERVAL + 4
        
        for i in range(total_slots):
            # Initiator TX
            tx_initiator = initiator.get_tx_signal()
            
            # Propagate
            rx_at_responder = chan_i2r.simulate_hf_channel(tx_initiator)
            
            # Responder RX/TX
            tx_responder = responder.process_rx_and_get_tx(rx_at_responder, i)
            
            # Propagate Back
            rx_at_initiator = chan_r2i.simulate_hf_channel(tx_responder)
            
            # Initiator RX
            initiator.process_rx_signal(rx_at_initiator, i)
            
            # Check for success
            if len(initiator.known_friends) > 0:
                print(f"\n>>> SUCCESS at {snr}dB! Link Established.")
                break
        else:
            print(f"\n>>> FAILURE at {snr}dB. No link established.")



def run_party_simulation():
    print("--- WAVELET PARTY PROTOCOL SIMULATION ---")
    print(f"Initiator WWBEAT Interval: {CQ_INTERVAL} slots")
    print(f"Slot Duration: {SLOT_DURATION}s")
    
    # Create Stations
    initiator = InitiatorStation("INITIATOR")
    
    # Station B: Close, Strong (+10dB)
    station_b = ResponderStation("STATION_B", snr_profile=10)
    
    # Station C: Far, Weak (-20dB for this test)
    station_c = ResponderStation("STATION_C", snr_profile=-20)
    
    # Channels
    # Initiator -> B
    chan_i2b = HFChannelSimulator(snr_db=10)
    # Initiator -> C
    chan_i2c = HFChannelSimulator(snr_db=-20)
    # B -> Initiator
    chan_b2i = HFChannelSimulator(snr_db=10)
    # C -> Initiator
    chan_c2i = HFChannelSimulator(snr_db=-20)
    
    # WebSDR Channels (Observer)
    # We use High SNR (100dB) to get "Clean" faded signals, then add a common noise floor later.
    # This ensures the noise floor is constant while signal strengths vary.
    chan_i2w = HFChannelSimulator(snr_db=100) 
    chan_b2w = HFChannelSimulator(snr_db=100)
    chan_c2w = HFChannelSimulator(snr_db=100)
    
    # Audio Buffers
    audio_initiator = []
    audio_websdr = []
    
    # Noise Floor Configuration
    # We define a constant noise floor amplitude.
    # Signals are scaled relative to this to achieve desired SNR.
    NOISE_RMS = 0.05
    
    # Target SNRs (Voltage Ratios relative to Noise RMS)
    # SNR_dB = 20 * log10(Signal_RMS / Noise_RMS)
    # Signal_RMS = Noise_RMS * 10^(SNR_dB / 20)
    
    # Initiator: +20dB -> 10x Noise
    scale_i = NOISE_RMS * (10 ** (20/20))
    # Station B: +10dB -> 3.16x Noise
    scale_b = NOISE_RMS * (10 ** (10/20))
    # Station C: -10dB -> 0.316x Noise
    scale_c = NOISE_RMS * (10 ** (-10/20))
    
    # Run for 3 Cycles (24 Slots)
    total_slots = CQ_INTERVAL * 3
    
    for i in range(total_slots):
        print(f"\n--- SLOT {i} (Cycle {i // CQ_INTERVAL}, Step {i % CQ_INTERVAL}) ---")
        
        # 1. Initiator Transmits (or Listens)
        tx_initiator = initiator.get_tx_signal()
        if i % CQ_INTERVAL == 0:
            print(f"DEBUG: Slot {i} TX Initiator Max: {np.max(np.abs(tx_initiator))}")
        
        # 2. Propagate to Responders
        rx_at_b = chan_i2b.simulate_hf_channel(tx_initiator)
        if i % CQ_INTERVAL == 0:
            print(f"DEBUG: Slot {i} RX at B Max: {np.max(np.abs(rx_at_b))}")
        rx_at_c = chan_i2c.simulate_hf_channel(tx_initiator)
        
        # 3. Responders Process & Transmit
        tx_b = station_b.process_rx_and_get_tx(rx_at_b, i)
        tx_c = station_c.process_rx_and_get_tx(rx_at_c, i)
        
        # 4. Propagate back to Initiator (Sum of signals!)
        # This simulates the "Party Line" - Initiator hears everyone
        rx_from_b = chan_b2i.simulate_hf_channel(tx_b)
        rx_from_c = chan_c2i.simulate_hf_channel(tx_c)
        
        # Ensure lengths match (channel simulation might vary slightly due to multipath/resampling)
        min_len = min(len(rx_from_b), len(rx_from_c))
        rx_from_b = rx_from_b[:min_len]
        rx_from_c = rx_from_c[:min_len]
        
        rx_at_initiator = rx_from_b + rx_from_c
        
        # 5. Initiator Listens
        initiator.process_rx_signal(rx_at_initiator, i)
        
        # 6. WebSDR Reception (The "God View")
        # Get "Clean" faded signals (High SNR channel)
        rx_w_i = chan_i2w.simulate_hf_channel(tx_initiator)
        rx_w_b = chan_b2w.simulate_hf_channel(tx_b)
        rx_w_c = chan_c2w.simulate_hf_channel(tx_c)
        
        # Align lengths
        len_w = min(len(rx_w_i), len(rx_w_b), len(rx_w_c))
        
        # Normalize signals to unit amplitude before scaling
        # This ensures our SNR calculations are correct regardless of channel gain
        def normalize(sig):
            peak = np.max(np.abs(sig))
            if peak > 0:
                return sig / peak
            return sig

        # Only normalize if there is signal, otherwise it amplifies floating point noise
        # We know when they transmit based on the simulation step, but let's just check peak
        sig_i = normalize(rx_w_i[:len_w])
        sig_b = normalize(rx_w_b[:len_w])
        sig_c = normalize(rx_w_c[:len_w])
        
        # Sum scaled signals
        sig_sum = (sig_i * scale_i) + \
                  (sig_b * scale_b) + \
                  (sig_c * scale_c)
                  
        # Add Constant Noise Floor
        noise = np.random.normal(0, NOISE_RMS, len_w)
        rx_at_websdr = sig_sum + noise
        
        audio_websdr.append(rx_at_websdr)
        
        # 7. Audio Visualization (Initiator's perspective)
        # If Initiator TX, record TX. If Initiator RX, record RX.
        if np.max(np.abs(tx_initiator)) > 0:
            audio_initiator.append(tx_initiator)
        else:
            audio_initiator.append(rx_at_initiator)

    # Save Audio
    full_audio = np.concatenate(audio_initiator)
    full_audio = full_audio / np.max(np.abs(full_audio)) * 0.9
    wav.write("data/10_wavelet_party.wav", SAMPLE_RATE, (full_audio * 32767).astype(np.int16))
    print("\nSaved 'wavelet_party.wav'")
    
    # Save WebSDR Audio
    full_audio_w = np.concatenate(audio_websdr)
    # Normalize carefully to avoid clipping from sum
    full_audio_w = full_audio_w / np.max(np.abs(full_audio_w)) * 0.9
    wav.write("data/10_wavelet_party_websdr.wav", SAMPLE_RATE, (full_audio_w * 32767).astype(np.int16))
    print("Saved 'wavelet_party_websdr.wav' (The Sound of the Party!)")
    
    print("\n--- PARTY STATUS ---")
    print(f"Initiator found {len(initiator.known_friends)} friends:")
    for slot, snr in initiator.known_friends.items():
        print(f" - Friend in Slot {slot}: SNR ~ {snr:.1f} dB")

if __name__ == "__main__":
    # Run the Deep SNR Test as requested
    run_deep_snr_test()
    # run_party_simulation() # Optional: Run the audio generation sim
