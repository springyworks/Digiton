import time
import numpy as np
import sounddevice as sd
import queue
import threading
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.core.ping import WaveletPingGenerator, WaveletPingDetector, SAMPLE_RATE

class PersistentEchoTester:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.fs = sample_rate
        self.block_size = 2048
        
        # Audio Queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Signal Generation
        self.generator = WaveletPingGenerator(self.fs)
        self.detector = WaveletPingDetector(self.fs)
        self.ping_signal = self.generator.generate_ping(wavelet_name='morl')
        
        # State
        self.running = False
        self.tx_active = False
        
        # Detection State
        self.overlap_buffer = np.zeros(0)
        self.last_detection_ts = 0
        
    def duplex_callback(self, indata, outdata, frames, time, status):
        """Combined Duplex Callback"""
        if status:
            print(f"Stream Status: {status}", file=sys.stderr)
            
        # 1. Handle Input (RX)
        self.input_queue.put(indata.copy())
        
        # 2. Handle Output (TX)
        try:
            # Try to get data from the output queue (non-blocking)
            data = self.output_queue.get_nowait()
            
            # If data is smaller than block size, pad with zeros
            if len(data) < frames:
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:frames].reshape(-1, 1)
                
        except queue.Empty:
            # If no signal to send, send SILENCE (Zeros)
            outdata.fill(0)

    def start(self):
        self.running = True
        
        print("Starting Persistent Audio Streams...")
        print("1. Open Pavucontrol NOW.")
        print("2. Look for 'ALSA plug-in [python3]' in Playback and Recording tabs.")
        print("3. Patch 'Playback' to your Transceiver (TX).")
        print("4. Patch 'Recording' from your WebSDR (RX).")
        print("Streams will remain active indefinitely.\n")

        # We use two separate streams to allow independent patching if needed,
        # or a single duplex stream. Duplex is usually easier to manage sync.
        # Let's use a Stream (Duplex)
        
        with sd.Stream(samplerate=self.fs,
                       blocksize=self.block_size,
                       channels=1,
                       callback=self.duplex_callback):
            
            print(">>> STREAMS ACTIVE. Press Ctrl+C to stop. <<<")
            
            # Main Control Loop
            last_ping_time = 0
            ping_interval = 5.0 # Send a ping every 5 seconds
            
            while self.running:
                current_time = time.time()
                
                # 1. Trigger Ping
                if current_time - last_ping_time > ping_interval:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Transmitting PING...")
                    # Queue the ping signal to be played by the callback
                    # We might need to chunk it if it's larger than block_size, 
                    # but for simplicity in this test, let's just put the whole thing 
                    # and let the callback handle it (simplified) or just put chunks.
                    
                    # Better approach for the callback: Put chunks
                    sig = self.ping_signal.astype(np.float32)
                    # Split into blocks
                    for i in range(0, len(sig), self.block_size):
                        chunk = sig[i:i+self.block_size]
                        self.output_queue.put(chunk)
                        
                    last_ping_time = current_time
                
                # 2. Process RX
                # We use an overlapping buffer to ensure we don't miss pings that straddle chunk boundaries.
                try:
                    while not self.input_queue.empty():
                        audio_chunk = self.input_queue.get_nowait().flatten()
                        
                        # Combine with previous data (Overlap-Save)
                        # We keep enough history to cover the ping duration (~0.2s)
                        combined = np.concatenate((self.overlap_buffer, audio_chunk))
                        
                        # Update overlap buffer for NEXT iteration
                        # Keep the last 0.3s of data
                        keep_samples = int(self.fs * 0.3)
                        if len(combined) > keep_samples:
                            self.overlap_buffer = combined[-keep_samples:]
                        else:
                            self.overlap_buffer = combined
                            
                        # Run detector on the COMBINED signal
                        # This effectively scans the "seam" between chunks
                        peak, snr, _ = self.detector.detect_presence(combined)
                        
                        # Debounce: Only report if enough time has passed since last detection
                        # This prevents reporting the same ping multiple times as it slides through the buffer
                        now = time.time()
                        if snr > 3.0 and (now - self.last_detection_ts > 1.0): 
                            rms = np.sqrt(np.mean(audio_chunk**2))
                            print(f"   > RX Event: SNR={snr:.1f}dB (RMS={rms:.4f}) [Robust Detect]")
                            self.last_detection_ts = now
                                
                except queue.Empty:
                    pass
                
                time.sleep(0.1)

if __name__ == "__main__":
    from datetime import datetime
    try:
        tester = PersistentEchoTester()
        tester.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
