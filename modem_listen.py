#!/usr/bin/env python3
"""
WAVELET MODEM - LISTEN MODE (Hybrid Rust/Python)
================================================
A passive receiver for monitoring Wavelet Party Protocol traffic.
Uses the Rust 'stream_decode' binary for high-performance DSP.

Usage:
    python3 modem_listen.py <wav_file>

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
CQ_INTERVAL = 8            # Master CQ every 8 slots
RUST_BINARY_PATH = "/home/rustuser/projects/rust/MorletModemRust/target/debug/stream_decode"

class RustDecoder:
    def __init__(self, sample_rate=8000):
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
                event = json.loads(line.decode('utf-8'))
                self.events.append(event)
            except json.JSONDecodeError:
                pass # Ignore partial lines or debug output

    def feed_audio(self, audio_chunk):
        """
        Feed float32 audio chunk to Rust process.
        """
        # Convert to float32 bytes
        bytes_data = audio_chunk.astype(np.float32).tobytes()
        try:
            self.process.stdin.write(bytes_data)
            self.process.stdin.flush()
        except BrokenPipeError:
            print("Error: Rust process died")
            self.running = False

    def close(self):
        self.running = False
        if self.process.stdin:
            self.process.stdin.close()
        self.process.terminate()
        self.process.wait()

    def get_events(self):
        """Return and clear new events"""
        # For this simple visualizer, we just return all events that match a time window?
        # Or just return all and let the caller filter.
        # Let's just return the list reference, caller handles logic.
        return self.events

def visualize_grid(detections, cycle_num):
    """
    Print a visual grid showing slot activity.
    """
    print(f"\n┌─ CYCLE {cycle_num} " + "─" * 60 + "┐")
    
    slots_display = []
    for slot_idx, (detected, snr_db, peak) in enumerate(detections):
        if slot_idx == 0:
            # CQ-Ping slot
            if detected:
                slots_display.append(f"CQ({snr_db:4.1f}dB)")
            else:
                slots_display.append("  CQ(--) ")
        else:
            if detected:
                slots_display.append(f"S{slot_idx}({snr_db:4.1f}dB)")
            else:
                slots_display.append("  --    ")
    
    # Print slots in a row
    print("│ ", end="")
    for disp in slots_display:
        print(f"{disp:>10}", end=" ")
    print("│")
    print("└" + "─" * 73 + "┘")

def listen_to_wav(input_file):
    """
    Main listening function: decode a WAV file and display WPP activity.
    """
    print("=" * 75)
    print("WAVELET MODEM - LISTEN MODE (Hybrid Rust/Python)")
    print("=" * 75)
    print(f"Input File: {input_file}")
    
    if not os.path.exists(RUST_BINARY_PATH):
        print(f"ERROR: Rust binary not found at {RUST_BINARY_PATH}")
        print("Please run 'cargo build -p apps --bin stream_decode' first.")
        return

    # Load WAV
    try:
        sr, audio = wav.read(input_file)
    except Exception as e:
        print(f"ERROR: Could not read WAV file: {e}")
        return
        
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        
    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    
    # Start Rust Decoder
    decoder = RustDecoder(sample_rate=sr)
    
    # Feed all audio at once (for file mode)
    # In a real stream, we'd chunk this.
    print("Feeding audio to Rust DSP engine...")
    decoder.feed_audio(audio)
    
    # Wait a bit for processing to finish
    time.sleep(0.5) 
    decoder.close()
    
    events = decoder.get_events()
    print(f"Received {len(events)} detection events from Rust.")
    
    # Process events into slots
    slot_samples = int(SLOT_DURATION * sr)
    num_slots = len(audio) // slot_samples
    num_cycles = num_slots // CQ_INTERVAL
    
    print(f"Total Slots: {num_slots}")
    print(f"Complete Cycles: {num_cycles}\n")
    
    master_detections = []
    station_activity = {i: [] for i in range(1, CQ_INTERVAL)}
    
    for cycle in range(num_cycles):
        cycle_detections = []
        
        for slot in range(CQ_INTERVAL):
            global_slot = cycle * CQ_INTERVAL + slot
            start_time = global_slot * SLOT_DURATION
            end_time = start_time + SLOT_DURATION
            
            # Check for events in this time window
            slot_events = [e for e in events if start_time <= e['timestamp'] < end_time]
            
            detected = len(slot_events) > 0
            if detected:
                # Take best event
                best = max(slot_events, key=lambda x: x['score'])
                snr_db = 20 * np.log10(best['score'] * 100) # Rough approx since score is raw correlation
                peak = best['score']
                spin = best['spin']
            else:
                snr_db = 0
                peak = 0
                spin = "none"
            
            cycle_detections.append((detected, snr_db, peak))
            
            # Track activity
            if slot == 0 and detected and spin == "right": # Master usually Right spin? Or just any?
                master_detections.append((cycle, snr_db))
            elif slot > 0 and detected:
                station_activity[slot].append((cycle, snr_db))
        
        visualize_grid(cycle_detections, cycle)
    
    # Summary Report (Same as before)
    print("\n" + "=" * 75)
    print("DETECTION SUMMARY")
    print("=" * 75)
    
    print(f"\n🎯 MASTER (CQ-Ping) Detected in {len(master_detections)}/{num_cycles} cycles:")
    if master_detections:
        avg_snr = np.mean([snr for _, snr in master_detections])
        print(f"   Average SNR: {avg_snr:.1f} dB")
    
    print("\n📡 STATION ACTIVITY:")
    active_slots = 0
    for slot in range(1, CQ_INTERVAL):
        activity = station_activity[slot]
        if activity:
            active_slots += 1
            avg_snr = np.mean([snr for _, snr in activity])
            print(f"   Slot {slot}: {len(activity)}/{num_cycles} cycles, Avg SNR: {avg_snr:.1f} dB")

def listen_to_stream():
    """
    Live streaming mode: Reads raw float32 audio from stdin and processes in real-time.
    Assumes 8000 Hz, Mono, Float32 LE.
    """
    print("=" * 75)
    print("WAVELET MODEM - LIVE STREAM MODE")
    print("=" * 75)
    print("Listening on stdin (Format: 8000Hz, Mono, Float32)...")
    
    if not os.path.exists(RUST_BINARY_PATH):
        print(f"ERROR: Rust binary not found at {RUST_BINARY_PATH}")
        return

    decoder = RustDecoder(sample_rate=8000)
    
    # Buffer for calculating slots
    chunk_size = 1024
    
    try:
        while True:
            # Read raw bytes from stdin
            raw_data = sys.stdin.buffer.read(chunk_size * 4) # 4 bytes per float
            if not raw_data:
                break
                
            # Feed to Rust
            try:
                decoder.process.stdin.write(raw_data)
                decoder.process.stdin.flush()
            except BrokenPipeError:
                break
            
            # Check for events
            while decoder.events:
                event = decoder.events.pop(0)
                # Real-time print
                t = event['timestamp']
                spin = event['spin']
                score = event['score']
                print(f"[{t:8.3f}s] DETECT: {spin.upper()} (Score: {score:.4f})")
                
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        decoder.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 modem_listen.py <wav_file>")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    
    if wav_file == "stream":
        # Live streaming mode (reads raw f32 LE 8000Hz from stdin)
        listen_to_stream()
    else:
        listen_to_wav(wav_file)