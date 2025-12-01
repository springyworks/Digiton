import time
import numpy as np
import sounddevice as sd
import queue
import sys
import os
import threading
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.core.ping import WaveletPingGenerator, WaveletPingDetector, SAMPLE_RATE
from digiton.sdr import iq_downconvert, detect_spin

# Configuration
BLOCK_SIZE = 2048
PING_INTERVAL = 3.0  # Seconds between pings
TX_GAIN = 0.8
PING_FREQ = 1500 # Center frequency of the ping

class RealTimeEchoTester:
    def __init__(self):
        self.fs = SAMPLE_RATE
        self.generator = WaveletPingGenerator(self.fs)
        self.detector = WaveletPingDetector(self.fs)
        
        # Queues
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()
        
        self.running = False
        self.last_tx_time = 0
        self.tx_times = [] # Keep track of TX times for RTT
        
        # Generate the standard ping for transmission
        self.ping_signal = self.generator.generate_ping(wavelet_name='morl') * TX_GAIN
        
        # Detection State
        self.overlap_buffer = np.zeros(0)
        self.last_detection_ts = 0
        
        print(f"Initialized Echo Tester at {self.fs}Hz")
        print(f"Ping Duration: {len(self.ping_signal)/self.fs:.3f}s")

    def duplex_callback(self, indata, outdata, frames, time_info, status):
        """
        Callback for full-duplex audio.
        """
        if status:
            print(f"\nStream Status: {status}", file=sys.stderr)
            
        # 1. Capture Input (RX)
        self.rx_queue.put(indata.copy())
        
        # 2. Handle Output (TX)
        try:
            data = self.tx_queue.get_nowait()
            # If data is smaller than block size, pad with zeros
            if len(data) < frames:
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:frames].reshape(-1, 1)
        except queue.Empty:
            outdata.fill(0)

    def draw_vu_meter(self, rms):
        # Log scale for VU
        if rms < 1e-6:
            db = -100
        else:
            db = 20 * np.log10(rms)
            
        # Range: -60dB to 0dB
        min_db = -60
        max_db = 0
        width = 40
        
        normalized = (db - min_db) / (max_db - min_db)
        normalized = max(0.0, min(1.0, normalized))
        
        bars = int(normalized * width)
        meter = "|" * bars + "." * (width - bars)
        
        sys.stdout.write(f"\rRX Level: [{meter}] {db:6.1f} dB")
        sys.stdout.flush()

    def start(self):
        self.running = True
        print("Starting Real-Time Echo Test (Continuous TX/RX)...")
        print("Patch 'ALSA plug-in [python3]' in Pavucontrol.")
        print("Press Ctrl+C to stop.\n")
        
        with sd.Stream(samplerate=self.fs,
                       blocksize=BLOCK_SIZE,
                       channels=1,
                       callback=self.duplex_callback):
            
            while self.running:
                current_time = time.time()
                
                # --- TX LOGIC ---
                if current_time - self.last_tx_time > PING_INTERVAL:
                    # Queue Ping
                    # We chunk it to fit block size logic if needed, but here we rely on 
                    # the callback consuming one block at a time. 
                    # Wait, the callback consumes ONE block. If we put a huge array, 
                    # the callback logic above (get_nowait) only takes ONE item.
                    # We need to split the ping into blocks.
                    
                    sys.stdout.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> TX PING\n")
                    
                    sig = self.ping_signal.astype(np.float32)
                    for i in range(0, len(sig), BLOCK_SIZE):
                        chunk = sig[i:i+BLOCK_SIZE]
                        self.tx_queue.put(chunk)
                        
                    self.last_tx_time = current_time
                    self.tx_times.append(current_time)
                    # Keep only last 10 TX times
                    if len(self.tx_times) > 10:
                        self.tx_times.pop(0)

                # --- RX LOGIC ---
                try:
                    while not self.rx_queue.empty():
                        audio_chunk = self.rx_queue.get_nowait().flatten()
                        
                        # VU Meter Update
                        rms = np.sqrt(np.mean(audio_chunk**2))
                        self.draw_vu_meter(rms)
                        
                        # Robust Detection (Overlap-Save)
                        combined = np.concatenate((self.overlap_buffer, audio_chunk))
                        
                        # Update overlap buffer (keep last 0.3s)
                        keep_samples = int(self.fs * 0.3)
                        if len(combined) > keep_samples:
                            self.overlap_buffer = combined[-keep_samples:]
                        else:
                            self.overlap_buffer = combined
                            
                        # Run Detector
                        peak, snr, _, peak_idx = self.detector.detect_presence(combined)
                        
                        # Calculate dB for display and logic
                        # snr from detector is Linear Ratio (Peak / Median)
                        snr_db = 20 * np.log10(snr) if snr > 0 else 0
                        
                        # Debounce & Sanity Check
                        # 1. SNR > 15.0 (Linear) -> ~23.5dB. Raised to reject impulsive HF noise.
                        # 2. Peak > 1e-3 (Absolute amplitude check)
                        # 3. Time debounce
                        if snr > 15.0 and peak > 1e-3 and (current_time - self.last_detection_ts > 1.0):
                            # Calculate RTT
                            rtt_msg = ""
                            if self.tx_times:
                                # Find closest TX time
                                # Since we expect RTT > 0, look for TX times in the past
                                valid_tx = [t for t in self.tx_times if current_time - t > 0.1] # Ignore immediate self-hearing < 100ms
                                if valid_tx:
                                    last_tx = valid_tx[-1]
                                    rtt = current_time - last_tx
                                    rtt_msg = f" | RTT: {rtt*1000:.0f}ms"
                            
                            # Analyze Spin (I/Q)
                            # Extract the signal around the peak for spin analysis
                            # Peak index is where the correlation is max, which corresponds to the *end* of the template match?
                            # Or center? np.correlate 'same' mode centers it.
                            # Let's take a window around the peak.
                            window_size = int(0.2 * self.fs) # 0.2s window
                            start_idx = max(0, peak_idx - window_size // 2)
                            end_idx = min(len(combined), peak_idx + window_size // 2)
                            spin_chunk = combined[start_idx:end_idx]
                            
                            iq = iq_downconvert(spin_chunk, PING_FREQ, self.fs)
                            spin_dir, spin_freq = detect_spin(iq, self.fs)
                            
                            spin_msg = f" | Spin: {spin_dir} ({spin_freq:.1f}Hz)" if spin_dir else " | Spin: ?"
                            
                            # Calculate Noise Floor for debug
                            nf = peak / snr if snr > 0 else 0
                            
                            sys.stdout.write(f"\n   <<< CONFIRMED PING! SNR={snr_db:.1f}dB (Ratio={snr:.1f}, NF={nf:.4f}){rtt_msg}{spin_msg}\n")
                            self.last_detection_ts = current_time
                            
                except queue.Empty:
                    pass
                
                time.sleep(0.01)

if __name__ == "__main__":
    try:
        tester = RealTimeEchoTester()
        tester.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    print(sd.query_devices())
    print("\nUsing default device (PipeWire usually handles this).")
    
    tester = RealTimeEchoTester()
    tester.run()
