import time
import numpy as np
import sounddevice as sd
import queue
import sys
import os
import threading
import random
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.core.ping import WaveletPingGenerator, WaveletPingDetector, SAMPLE_RATE
from digiton.sdr import iq_downconvert, detect_spin, detect_spin_robust

# --- PROTOCOL CONSTANTS ---
SLOT_DURATION = 1.5        # Seconds (Extended for Deep Sequence)
SLOTS_PER_CYCLE = 8        # Slot 0 = CQ, Slots 1-7 = Response
CQ_SLOT_INDEX = 0
PING_FREQ = 1500           # Hz
TX_GAIN = 0.8
BLOCK_SIZE = 2048

# Deep Mode Constants
DEEP_REPEATS = 4
DEEP_INTERVAL = 0.25       # Seconds

class RealTimeProtocolTester:
    def __init__(self):
        self.fs = SAMPLE_RATE
        self.generator = WaveletPingGenerator(self.fs)
        
        # Standard Detector (Single Ping)
        self.detector = WaveletPingDetector(self.fs)
        
        # Deep Detector (Sequence)
        # Generate the sequence template (Center)
        self.deep_seq_center, self.ping_times, _ = self.generator.generate_ping_sequence(
            num_pings=DEEP_REPEATS,
            ping_interval=DEEP_INTERVAL,
            random_start_range=(0,0),
            wavelet_name='morl',
            frequency=PING_FREQ,
            padding=0.0
        )
        self.deep_seq_center /= np.max(np.abs(self.deep_seq_center))
        
        # Generate Left Spin (-200Hz)
        self.deep_seq_left, _, _ = self.generator.generate_ping_sequence(
            num_pings=DEEP_REPEATS,
            ping_interval=DEEP_INTERVAL,
            random_start_range=(0,0),
            wavelet_name='morl',
            frequency=PING_FREQ - 200,
            padding=0.0
        )
        self.deep_seq_left /= np.max(np.abs(self.deep_seq_left))
        
        # Generate Right Spin (+200Hz)
        self.deep_seq_right, _, _ = self.generator.generate_ping_sequence(
            num_pings=DEEP_REPEATS,
            ping_interval=DEEP_INTERVAL,
            random_start_range=(0,0),
            wavelet_name='morl',
            frequency=PING_FREQ + 200,
            padding=0.0
        )
        self.deep_seq_right /= np.max(np.abs(self.deep_seq_right))
        
        # Detector uses Center template (Matched Filter is robust enough to detect +/- 200Hz usually, 
        # or we rely on the fact that the envelope matches. 
        # Ideally we'd have a bank of filters, but for now we detect with Center and measure Spin with I/Q).
        self.deep_detector = WaveletPingDetector(self.fs, template_signal=self.deep_seq_center)
        
        # Queues
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()
        
        self.running = False
        
        # Protocol State
        self.current_slot = 0
        self.cycle_start_time = 0
        self.next_slot_time = 0
        
        # Use Deep Sequence for CQ
        # We will pick one dynamically
        self.tx_spin = 'center' 
        
        # Detection State
        self.overlap_buffer = np.zeros(0)
        self.last_detection_ts = 0
        
        print(f"Initialized Protocol Tester at {self.fs}Hz")
        print(f"Slot Duration: {SLOT_DURATION}s | Cycle: {SLOT_DURATION*SLOTS_PER_CYCLE}s")
        print(f"Deep Mode: {DEEP_REPEATS} pings @ {DEEP_INTERVAL}s interval")

    def duplex_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"\nStream Status: {status}", file=sys.stderr)
            
        # 1. Capture Input (RX)
        self.rx_queue.put(indata.copy())
        
        # 2. Handle Output (TX)
        try:
            data = self.tx_queue.get_nowait()
            if len(data) < frames:
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:frames].reshape(-1, 1)
        except queue.Empty:
            outdata.fill(0)

    def draw_status(self, rms):
        # Simple VU and Slot Indicator
        if rms < 1e-6: db = -100
        else: db = 20 * np.log10(rms)
        
        # Slot Indicator
        slots_vis = ["."] * SLOTS_PER_CYCLE
        slots_vis[self.current_slot] = "X" if self.current_slot == CQ_SLOT_INDEX else "R"
        slots_str = "".join(slots_vis)
        
        sys.stdout.write(f"\r[{slots_str}] Slot {self.current_slot} | RX: {db:5.1f} dB   ")
        sys.stdout.flush()

    def run(self):
        self.running = True
        print("Starting Real-Time Protocol Test (Initiator Mode)...")
        print("Patch 'ALSA plug-in [python3]' in Pavucontrol.")
        print("Press Ctrl+C to stop.\n")
        
        # Initialize Clock
        self.cycle_start_time = time.time()
        self.next_slot_time = self.cycle_start_time + SLOT_DURATION
        
        with sd.Stream(samplerate=self.fs,
                       blocksize=BLOCK_SIZE,
                       channels=1,
                       callback=self.duplex_callback):
            
            while self.running:
                now = time.time()
                
                # --- CLOCK & TX LOGIC ---
                if now >= self.next_slot_time:
                    # Advance Slot
                    self.current_slot = (self.current_slot + 1) % SLOTS_PER_CYCLE
                    self.next_slot_time += SLOT_DURATION
                    
                    # New Cycle?
                    if self.current_slot == 0:
                        self.cycle_start_time = now # Reset cycle reference roughly
                        
                        # Pick a random spin for this cycle
                        spin_choice = random.choice(['center', 'left', 'right'])
                        if spin_choice == 'center':
                            sig = self.deep_seq_center * TX_GAIN
                            spin_label = "AMBIGUOUS (1500Hz)"
                        elif spin_choice == 'left':
                            sig = self.deep_seq_left * TX_GAIN
                            spin_label = "LEFT (1300Hz)"
                        else:
                            sig = self.deep_seq_right * TX_GAIN
                            spin_label = "RIGHT (1700Hz)"
                            
                        # TX CQ PING
                        sys.stdout.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> TX CQ (Slot 0) [{spin_label}]\n")
                        
                        # Queue the ping
                        sig = sig.astype(np.float32)
                        # Split into blocks for the callback
                        for i in range(0, len(sig), BLOCK_SIZE):
                            chunk = sig[i:i+BLOCK_SIZE]
                            self.tx_queue.put(chunk)

                # --- RX LOGIC ---
                try:
                    while not self.rx_queue.empty():
                        audio_chunk = self.rx_queue.get_nowait().flatten()
                        
                        # VU Meter
                        rms = np.sqrt(np.mean(audio_chunk**2))
                        self.draw_status(rms)
                        
                        # Buffer & Detect
                        combined = np.concatenate((self.overlap_buffer, audio_chunk))
                        # Keep enough for Deep Sequence (approx 1.0s) + margin -> 2.0s
                        keep_samples = int(self.fs * 2.0)
                        if len(combined) > keep_samples:
                            self.overlap_buffer = combined[-keep_samples:]
                        else:
                            self.overlap_buffer = combined
                            
                        # Run Deep Detector
                        # Note: detect_presence returns peak, snr, correlation, peak_idx
                        peak, snr, _, peak_idx = self.deep_detector.detect_presence(combined)
                        snr_db = 20 * np.log10(snr) if snr > 0 else 0
                        
                        # Thresholds (Adjusted for Deep Mode)
                        # Deep Mode correlation peak will be higher (sum of 4 pings)
                        # But noise floor also calculated differently? No, noise floor is median of correlation.
                        # SNR should be higher if signal is present.
                        if snr > 12.0 and peak > 1e-3 and (now - self.last_detection_ts > 1.0):
                            # Valid Detection
                            
                            # Determine Arrival Slot
                            # We need to map 'now' back to the slot index relative to cycle_start
                            # Note: 'now' is when we processed the chunk, which is roughly when it arrived.
                            # Cycle time:
                            time_in_cycle = (now - self.cycle_start_time) % (SLOT_DURATION * SLOTS_PER_CYCLE)
                            arrival_slot = int(time_in_cycle / SLOT_DURATION)
                            
                            # Spin Analysis (Robust Accumulation)
                            # We know the relative offsets of the pings from self.ping_times
                            # The template starts at index 0.
                            # The detected peak corresponds to the center of the template match?
                            # np.correlate 'same' aligns the center of the template to the peak.
                            # Template length:
                            tpl_len = len(self.deep_seq_center)
                            seq_start_idx = peak_idx - tpl_len // 2
                            
                            iq_chunks = []
                            ping_dur_samples = int(0.2 * self.fs) # 0.2s ping duration
                            
                            for pt in self.ping_times:
                                # Calculate start/end of this ping in the combined buffer
                                p_start = seq_start_idx + int(pt * self.fs)
                                p_end = p_start + ping_dur_samples
                                
                                # Bounds check
                                if p_start >= 0 and p_end <= len(combined):
                                    chunk = combined[p_start:p_end]
                                    # Downconvert this chunk
                                    iq_chunk = iq_downconvert(chunk, PING_FREQ, self.fs)
                                    iq_chunks.append(iq_chunk)
                            
                            if iq_chunks:
                                spin_dir, spin_freq = detect_spin_robust(iq_chunks, self.fs)
                                spin_msg = f"{spin_dir} ({spin_freq:.1f}Hz)" if spin_dir else "?"
                            else:
                                spin_msg = "?"
                            
                            sys.stdout.write(f"\n   <<< RX DEEP SEQ! Slot {arrival_slot} | SNR={snr_db:.1f}dB | Spin: {spin_msg}\n")
                            
                            self.last_detection_ts = now
                            
                except queue.Empty:
                    pass
                
                time.sleep(0.005)

if __name__ == "__main__":
    try:
        tester = RealTimeProtocolTester()
        tester.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
