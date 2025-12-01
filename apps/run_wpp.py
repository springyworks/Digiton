import time
import numpy as np
import sounddevice as sd
import queue
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.protocol.wpp import WaveletPartyProtocol, StationConfig, ProtocolMode

# Configuration
BLOCK_SIZE = 2048

def main():
    print("--- Wavelet Party Protocol (WPP) Runner ---")
    
    config = StationConfig(
        callsign="INITIATOR",
        slot_duration=1.5,
        slots_per_cycle=8
    )
    
    protocol = WaveletPartyProtocol(config)
    protocol.mode = ProtocolMode.DEEP
    
    # Audio Callback
    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Stream Status: {status}", file=sys.stderr)
            
        # RX
        protocol.process_audio(indata.flatten())
        
        # TX
        try:
            data = protocol.tx_queue.get_nowait()
            # Handle block size mismatch
            # If data is larger than frames, we need to buffer it?
            # The protocol puts the WHOLE ping in the queue.
            # We need a TX buffer here in the runner to stream it out.
            # This is a common issue. Let's fix it by using a persistent buffer.
            # But for now, let's assume the protocol puts chunks or we handle it.
            # Actually, the protocol puts the whole array.
            # We need a tx_buffer in the runner.
            pass 
        except queue.Empty:
            pass
            
        # We need a proper TX buffer mechanism in the callback
        # Let's use a simple global/class buffer
        
    # Re-implementing callback with state
    tx_buffer = np.zeros(0)
    
    def callback_with_state(indata, outdata, frames, time_info, status):
        nonlocal tx_buffer
        if status:
            print(f"Stream Status: {status}", file=sys.stderr)
            
        # RX
        protocol.process_audio(indata.flatten())
        
        # TX Logic
        # Check if we have data in the queue to add to buffer
        try:
            while True:
                new_data = protocol.tx_queue.get_nowait()
                tx_buffer = np.concatenate((tx_buffer, new_data))
        except queue.Empty:
            pass
            
        # Write to output
        if len(tx_buffer) >= frames:
            outdata[:] = tx_buffer[:frames].reshape(-1, 1)
            tx_buffer = tx_buffer[frames:]
        else:
            # Partial write
            valid = len(tx_buffer)
            outdata[:valid] = tx_buffer.reshape(-1, 1)
            outdata[valid:] = 0
            tx_buffer = np.zeros(0)

    print("Starting Audio Stream...")
    with sd.Stream(samplerate=config.sample_rate,
                   blocksize=BLOCK_SIZE,
                   channels=1,
                   callback=callback_with_state):
        
        protocol.cycle_start_time = time.time()
        
        while True:
            protocol.tick()
            time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")
