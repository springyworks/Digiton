#!/usr/bin/env python3
"""
WAVELET MODEM - LISTEN MODE (Hybrid Rust/Python)
================================================
A passive receiver for monitoring Wavelet Party Protocol traffic.
Uses the Rust 'stream_decode' binary for high-performance DSP.

Usage:
    python3 listener.py <wav_file>

The modem will:
1. Stream audio to the Rust subprocess
2. Parse JSON detection events
3. Visualize the party grid
"""

import numpy as np
import scipy.io.wavfile as wav
import sys
import subprocess
import json
import threading
import time
import os

# Constants (must match protocol)
SLOT_DURATION = 0.4        # Seconds
CQ_INTERVAL = 8            # Initiator WWBEAT every 8 slots
RUST_BINARY_PATH = "/home/rustuser/projects/rust/MorletModemRust/target/debug/stream_decode"

class RustDecoder:
    def __init__(self, sample_rate=8000):
        if not os.path.exists(RUST_BINARY_PATH):
            print(f"Error: Rust binary not found at {RUST_BINARY_PATH}")
            print("Please build the Rust project first: cargo build")
            sys.exit(1)
            
        self.process = subprocess.Popen(
            [RUST_BINARY_PATH, "--fs", str(sample_rate), "--threshold", "0.05"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0 # Unbuffered
        )
        self.events = []
        self.running = True
        self.thread = threading.Thread(target=self._read_stdout)
        self.thread.daemon = True
        self.thread.start()

    def _read_stdout(self):
        while self.running:
            line = self.process.stdout.readline()
            if not line:
                break
            try:
                # Parse JSON line
                # Expected format: {"time": 1.23, "freq": 1500, "snr": 12.5, "spin": "right"}
                data = json.loads(line.decode('utf-8').strip())
                self.events.append(data)
                print(f"DETECTION: {data}")
            except json.JSONDecodeError:
                pass # Ignore non-JSON debug output

    def write_audio(self, chunk_bytes):
        try:
            self.process.stdin.write(chunk_bytes)
            self.process.stdin.flush()
        except BrokenPipeError:
            self.running = False

    def close(self):
        self.running = False
        self.process.terminate()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 listener.py <wav_file>")
        sys.exit(1)
        
    wav_file = sys.argv[1]
    fs, audio = wav.read(wav_file)
    
    # Convert to float32 -1..1
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
        
    print(f"Streaming {wav_file} ({len(audio)/fs:.1f}s) to Rust decoder...")
    
    decoder = RustDecoder(fs)
    
    # Stream in chunks
    chunk_size = 1024
    audio_bytes = audio.tobytes()
    
    for i in range(0, len(audio_bytes), chunk_size * 4): # 4 bytes per float32
        chunk = audio_bytes[i:i + chunk_size * 4]
        decoder.write_audio(chunk)
        time.sleep(chunk_size / fs) # Real-time simulation
        
    decoder.close()
    print("Done.")

if __name__ == "__main__":
    main()
