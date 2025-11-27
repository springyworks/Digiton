"""
WAVELET PARTY PROTOCOL (WPP)
============================
A robust, decentralized HF protocol for "finding friends" in the noise floor (-60dB).

DESIGN PHILOSOPHY
-----------------
1.  **Simplicity over Bandwidth:** The primary goal is *connection*, not throughput.
2.  **Master-Slave Timing:** No GPS/Wall-clock required. The "Net Control" (CQ Station) 
    sets the heartbeat. Everyone else syncs to the Master.
3.  **Physics-First:** Uses the robust Wavelet Ping (0.2s) which is proven to work 
    at -50dB to -60dB.
4.  **Adaptive:** Starts slow (Discovery), speeds up only when safe (Chat).

PROTOCOL STATES
---------------
1.  **CQ-PING (Heartbeat):** 
    -   Master sends a periodic Ping every T_slot (e.g., 1.0s).
    -   Pattern: PING ... Silence ... Silence ... PING ...
    -   Purpose: Allow listeners to synchronize their clocks and measure SNR.

2.  **DISCOVERY (Join):**
    -   Listener hears CQ-Ping.
    -   Listener picks a random "Response Slot" (Slotted Aloha) to avoid collisions.
    -   Listener sends a PONG in that slot.
    -   Master listens between Pings. If it detects a PONG, it notes the time offset.

3.  **HANDSHAKE (Ack):**
    -   Master changes pattern to acknowledge the specific slot where it heard a PONG.
    -   "I heard someone in Slot 3".
    -   Listener in Slot 3 confirms: "That's me!" -> Link Established.

4.  **PARTY (Round Robin):**
    -   Master polls connected stations one by one.
    -   "Station A -> Station B: Go ahead."
    -   Station B sends data.
    -   Master ACKs.
    -   "Station A -> Station C: Go ahead."

SIMULATION SETUP
----------------
-   **Station A (Master):** Sends CQ.
-   **Station B (Close):** High SNR (+10dB), Short Propagation Delay.
-   **Station C (Far):** Deep Noise (-40dB), Long Propagation Delay.
-   **Goal:** A establishes a "Party" with both B and C simultaneously.
"""

import numpy as np
import scipy.io.wavfile as wav
from hf_channel_simulator import HFChannelSimulator, WaveletPingGenerator, WaveletPingDetector, SAMPLE_RATE

# --- CONSTANTS ---
SLOT_DURATION = 0.4        # Seconds (Long enough for Ping + Propagation)
PING_DURATION = 0.2        # Seconds (Actual Wavelet Pulse)
CQ_INTERVAL = 8            # Slots (Master sends CQ-Ping every 8 slots)
                           # Slots 1-7 are for Responses (Slotted Aloha)

# --- STATION CLASSES ---

class MasterStation:
    def __init__(self, name, repeats=1):
        self.name = name
        self.repeats = repeats
        self.generator = WaveletPingGenerator(SAMPLE_RATE)
        self.detector = WaveletPingDetector(SAMPLE_RATE)
        self.known_friends = {} # {slot_index: snr}
        self.current_slot = 0
        
    def get_tx_signal(self):
        """
        Master Logic:
        - Slot 0: Send CQ-PING.
        - Slots 1-7: Listen (Silence).
        """
        cycle_idx = self.current_slot % CQ_INTERVAL
        
        # Calculate total slot samples
        slot_samples = int(SLOT_DURATION * SAMPLE_RATE)
        
        if cycle_idx == 0:
            # CQ-PING SLOT
            if self.repeats == 1:
                # Single ping
                sig = self.generator.generate_ping(wavelet_name='morl')
            else:
                # Multiple pings for deep SNR
                sig, _, _ = self.generator.generate_ping_sequence(
                    num_pings=self.repeats, 
                    ping_interval=PING_DURATION, # Tight packing
                    random_start_range=(0, 0),   # Start immediately
                    wavelet_name='morl'
                )
            
            # Pad with silence to fill the slot
            if len(sig) < slot_samples:
                padding = np.zeros(slot_samples - len(sig))
                sig = np.concatenate([sig, padding])
            elif len(sig) > slot_samples:
                # This shouldn't happen in normal config, but truncate if so
                sig = sig[:slot_samples]
                
            return sig
        else:
            # LISTEN SLOT
            return np.zeros(slot_samples)

    def process_rx_signal(self, rx_signal, slot_idx):
        """
        Listen for responses in the empty slots.
        """
        cycle_idx = self.current_slot % CQ_INTERVAL
        
        # We don't listen while transmitting (Slot 0)
        if cycle_idx == 0:
            self.current_slot += 1
            return
            
        # Detect (Coherent Integration)
        # Generate template with same repeats
        if self.repeats == 1:
            template = self.generator.generate_ping(wavelet_name='morl')
        else:
            template, _, _ = self.generator.generate_ping_sequence(
                num_pings=self.repeats,
                ping_interval=PING_DURATION,
                random_start_range=(0, 0),
                wavelet_name='morl'
            )
        detector = WaveletPingDetector(template_signal=template)
        
        _, _, corr = detector.detect_presence(rx_signal)
        peak = np.max(np.abs(corr))
        noise = np.median(np.abs(corr))
        snr = peak / noise if noise > 0 else 0
        snr_db = 20 * np.log10(snr) if snr > 0 else 0
        
        # Threshold for "Hearing a Friend"
        if snr_db > 10.0: # Lower threshold for deep noise
            print(f"[{self.name}] Heard response in Slot {cycle_idx} (SNR={snr_db:.1f}dB)!")
            if cycle_idx not in self.known_friends:
                self.known_friends[cycle_idx] = snr_db
                print(f"[{self.name}] *** NEW FRIEND ADDED IN SLOT {cycle_idx} ***")
        
        self.current_slot += 1

class ResponderStation:
    def __init__(self, name, snr_profile, repeats=1):
        self.name = name
        self.snr_profile = snr_profile
        self.repeats = repeats
        self.generator = WaveletPingGenerator(SAMPLE_RATE)
        self.detector = WaveletPingDetector(SAMPLE_RATE)
        
        self.synchronized = False
        self.master_slot_offset = 0
        self.my_response_slot = None 
        
    def process_rx_and_get_tx(self, rx_signal, global_slot_idx):
        """
        Responder Logic:
        1. Listen for CQ-Ping.
        2. Sync internal clock.
        3. Transmit PONG in assigned slot.
        """
        # 1. Listen (Coherent)
        if self.repeats == 1:
            template = self.generator.generate_ping(wavelet_name='morl')
        else:
            template, _, _ = self.generator.generate_ping_sequence(
                num_pings=self.repeats,
                ping_interval=PING_DURATION,
                random_start_range=(0, 0),
                wavelet_name='morl'
            )
        detector = WaveletPingDetector(template_signal=template)
        
        _, _, corr = detector.detect_presence(rx_signal)
        peak = np.max(np.abs(corr))
        noise = np.median(np.abs(corr))
        snr_db = 20 * np.log10(peak/noise) if noise > 0 else 0
        
        # Debug print for Slot 0 (Master CQ)
        if global_slot_idx % CQ_INTERVAL == 0:
             print(f"[{self.name}] Slot {global_slot_idx} RX SNR: {snr_db:.1f}dB (Peak: {peak:.4f}, Noise: {noise:.4f})")
             pass
        
        # 2. Sync Logic
        if snr_db > 10.0 and not self.synchronized:
            print(f"[{self.name}] CQ-PING DETECTED (SNR={snr_db:.1f}dB). Synchronizing...")
            self.synchronized = True
            self.master_slot_offset = global_slot_idx % CQ_INTERVAL
            
        # 3. Transmit Logic
        if self.synchronized:
            relative_slot = (global_slot_idx - self.master_slot_offset) % CQ_INTERVAL
            
            if relative_slot == 1: 
                 self.my_response_slot = np.random.randint(1, CQ_INTERVAL)

            if relative_slot == self.my_response_slot:
                print(f"[{self.name}] Transmitting PONG in Slot {relative_slot}...")
                if self.repeats == 1:
                    sig = self.generator.generate_ping(wavelet_name='morl')
                else:
                    sig, _, _ = self.generator.generate_ping_sequence(
                        num_pings=self.repeats,
                        ping_interval=PING_DURATION,
                        random_start_range=(0, 0),
                        wavelet_name='morl'
                    )
                
                # Pad to slot duration
                slot_samples = int(SLOT_DURATION * SAMPLE_RATE)
                if len(sig) < slot_samples:
                    padding = np.zeros(slot_samples - len(sig))
                    sig = np.concatenate([sig, padding])
                elif len(sig) > slot_samples:
                    sig = sig[:slot_samples]
                    
                return sig
                
        # Return silence of correct length
        slot_samples = int(SLOT_DURATION * SAMPLE_RATE)
        return np.zeros(slot_samples)

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
        
        master = MasterStation("MASTER", repeats=repeats)
        responder = ResponderStation("RESPONDER", snr_profile=snr, repeats=repeats)
        
        # Channels
        chan_m2r = HFChannelSimulator(snr_db=snr)
        chan_r2m = HFChannelSimulator(snr_db=snr)
        
        # Run for 1 Cycle (Enough to Sync and Reply)
        # Cycle = CQ_INTERVAL slots
        # We run for CQ_INTERVAL + 4 slots to ensure we catch the reply
        total_slots = CQ_INTERVAL + 4
        
        for i in range(total_slots):
            # Master TX
            tx_master = master.get_tx_signal()
            
            # Propagate
            rx_at_responder = chan_m2r.simulate_hf_channel(tx_master)
            
            # Responder RX/TX
            tx_responder = responder.process_rx_and_get_tx(rx_at_responder, i)
            
            # Propagate Back
            rx_at_master = chan_r2m.simulate_hf_channel(tx_responder)
            
            # Master RX
            master.process_rx_signal(rx_at_master, i)
            
            # Check for success
            if len(master.known_friends) > 0:
                print(f"\n>>> SUCCESS at {snr}dB! Link Established.")
                break
        else:
            print(f"\n>>> FAILURE at {snr}dB. No link established.")



def run_party_simulation():
    print("--- WAVELET PARTY PROTOCOL SIMULATION ---")
    print(f"Master CQ Interval: {CQ_INTERVAL} slots")
    print(f"Slot Duration: {SLOT_DURATION}s")
    
    # Create Stations
    master = MasterStation("MASTER")
    
    # Station B: Close, Strong (+10dB)
    station_b = ResponderStation("STATION_B", snr_profile=10)
    
    # Station C: Far, Weak (-20dB for this test)
    station_c = ResponderStation("STATION_C", snr_profile=-20)
    
    # Channels
    # Master -> B
    chan_m2b = HFChannelSimulator(snr_db=10)
    # Master -> C
    chan_m2c = HFChannelSimulator(snr_db=-20)
    # B -> Master
    chan_b2m = HFChannelSimulator(snr_db=10)
    # C -> Master
    chan_c2m = HFChannelSimulator(snr_db=-20)
    
    # WebSDR Channels (Observer)
    # We use High SNR (100dB) to get "Clean" faded signals, then add a common noise floor later.
    # This ensures the noise floor is constant while signal strengths vary.
    chan_m2w = HFChannelSimulator(snr_db=100) 
    chan_b2w = HFChannelSimulator(snr_db=100)
    chan_c2w = HFChannelSimulator(snr_db=100)
    
    # Audio Buffers
    audio_master = []
    audio_websdr = []
    
    # Noise Floor Configuration
    # We define a constant noise floor amplitude.
    # Signals are scaled relative to this to achieve desired SNR.
    NOISE_RMS = 0.05
    
    # Target SNRs (Voltage Ratios relative to Noise RMS)
    # SNR_dB = 20 * log10(Signal_RMS / Noise_RMS)
    # Signal_RMS = Noise_RMS * 10^(SNR_dB / 20)
    
    # Master: +20dB -> 10x Noise
    scale_m = NOISE_RMS * (10 ** (20/20))
    # Station B: +10dB -> 3.16x Noise
    scale_b = NOISE_RMS * (10 ** (10/20))
    # Station C: -10dB -> 0.316x Noise
    scale_c = NOISE_RMS * (10 ** (-10/20))
    
    # Run for 3 Cycles (24 Slots)
    total_slots = CQ_INTERVAL * 3
    
    for i in range(total_slots):
        print(f"\n--- SLOT {i} (Cycle {i // CQ_INTERVAL}, Step {i % CQ_INTERVAL}) ---")
        
        # 1. Master Transmits (or Listens)
        tx_master = master.get_tx_signal()
        if i % CQ_INTERVAL == 0:
            print(f"DEBUG: Slot {i} TX Master Max: {np.max(np.abs(tx_master))}")
        
        # 2. Propagate to Responders
        rx_at_b = chan_m2b.simulate_hf_channel(tx_master)
        if i % CQ_INTERVAL == 0:
            print(f"DEBUG: Slot {i} RX at B Max: {np.max(np.abs(rx_at_b))}")
        rx_at_c = chan_m2c.simulate_hf_channel(tx_master)
        
        # 3. Responders Process & Transmit
        tx_b = station_b.process_rx_and_get_tx(rx_at_b, i)
        tx_c = station_c.process_rx_and_get_tx(rx_at_c, i)
        
        # 4. Propagate back to Master (Sum of signals!)
        # This simulates the "Party Line" - Master hears everyone
        rx_from_b = chan_b2m.simulate_hf_channel(tx_b)
        rx_from_c = chan_c2m.simulate_hf_channel(tx_c)
        
        # Ensure lengths match (channel simulation might vary slightly due to multipath/resampling)
        min_len = min(len(rx_from_b), len(rx_from_c))
        rx_from_b = rx_from_b[:min_len]
        rx_from_c = rx_from_c[:min_len]
        
        rx_at_master = rx_from_b + rx_from_c
        
        # 5. Master Listens
        master.process_rx_signal(rx_at_master, i)
        
        # 6. WebSDR Reception (The "God View")
        # Get "Clean" faded signals (High SNR channel)
        rx_w_m = chan_m2w.simulate_hf_channel(tx_master)
        rx_w_b = chan_b2w.simulate_hf_channel(tx_b)
        rx_w_c = chan_c2w.simulate_hf_channel(tx_c)
        
        # Align lengths
        len_w = min(len(rx_w_m), len(rx_w_b), len(rx_w_c))
        
        # Normalize signals to unit amplitude before scaling
        # This ensures our SNR calculations are correct regardless of channel gain
        def normalize(sig):
            peak = np.max(np.abs(sig))
            if peak > 0:
                return sig / peak
            return sig

        # Only normalize if there is signal, otherwise it amplifies floating point noise
        # We know when they transmit based on the simulation step, but let's just check peak
        sig_m = normalize(rx_w_m[:len_w])
        sig_b = normalize(rx_w_b[:len_w])
        sig_c = normalize(rx_w_c[:len_w])
        
        # Sum scaled signals
        sig_sum = (sig_m * scale_m) + \
                  (sig_b * scale_b) + \
                  (sig_c * scale_c)
                  
        # Add Constant Noise Floor
        noise = np.random.normal(0, NOISE_RMS, len_w)
        rx_at_websdr = sig_sum + noise
        
        audio_websdr.append(rx_at_websdr)
        
        # 7. Audio Visualization (Master's perspective)
        # If Master TX, record TX. If Master RX, record RX.
        if np.max(np.abs(tx_master)) > 0:
            audio_master.append(tx_master)
        else:
            audio_master.append(rx_at_master)

    # Save Audio
    full_audio = np.concatenate(audio_master)
    full_audio = full_audio / np.max(np.abs(full_audio)) * 0.9
    wav.write("data/10_wavelet_party.wav", SAMPLE_RATE, (full_audio * 32767).astype(np.int16))
    print("\nSaved 'wavelet_party.wav'")
    
    # Save WebSDR Audio
    full_audio_w = np.concatenate(audio_websdr)
    # Normalize carefully to avoid clipping from sum
    full_audio_w = full_audio_w / np.max(np.abs(full_audio_w)) * 0.9
    wav.write("wavelet_party_websdr.wav", SAMPLE_RATE, (full_audio_w * 32767).astype(np.int16))
    print("Saved 'wavelet_party_websdr.wav' (The Sound of the Party!)")
    
    print("\n--- PARTY STATUS ---")
    print(f"Master found {len(master.known_friends)} friends:")
    for slot, snr in master.known_friends.items():
        print(f" - Friend in Slot {slot}: SNR ~ {snr:.1f} dB")

if __name__ == "__main__":
    # Run the Deep SNR Test as requested
    run_deep_snr_test()
    # run_party_simulation() # Optional: Run the audio generation sim
