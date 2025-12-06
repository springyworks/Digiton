import numpy as np
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from digiton.channel.simulator import HFChannelSimulator, SAMPLE_RATE
from digiton.protocol.wpp import WaveletPartyProtocol, StationConfig, ProtocolMode, ProtocolState, MODE_PARAMS

def run_deep_snr_test():
    print("--- WAVELET PARTY PROTOCOL: DEEP SNR TEST ---")
    
    # Test Cases: (SNR, Mode)
    # -20dB: NORMAL
    # -40dB: DEEP
    # -60dB: DEEP (Extreme)
    
    test_cases = [
        (-20, ProtocolMode.NORMAL),
        (-40, ProtocolMode.DEEP),
        (-60, ProtocolMode.DEEP)
    ]
    
    for snr, mode in test_cases:
        print(f"\n" + "="*50)
        print(f"TESTING SNR: {snr} dB | MODE: {mode.name}")
        print("="*50)
        
        # Configure Stations
        # Initiator: Auto-mode OFF to force specific test mode
        cfg_init = StationConfig(
            callsign="INIT", 
            auto_mode=False, 
            monitor_self=False,
            slot_duration=MODE_PARAMS[mode]['slot_dur']
        )
        initiator = WaveletPartyProtocol(cfg_init)
        initiator.mode = mode
        initiator.state = ProtocolState.DISCOVERY_TX # Force TX state
        
        # Responder: Auto-mode OFF, starts in DEEP (safe)
        cfg_resp = StationConfig(
            callsign="RESP", 
            auto_mode=False, 
            monitor_self=False,
            slot_duration=MODE_PARAMS[mode]['slot_dur']
        )
        responder = WaveletPartyProtocol(cfg_resp)
        responder.mode = ProtocolMode.DEEP # Always listen in DEEP initially
        responder.state = ProtocolState.LISTENING
        
        # Channels
        chan_i2r = HFChannelSimulator(snr_db=snr)
        chan_r2i = HFChannelSimulator(snr_db=snr)
        
        # Run for 2 Cycles
        slots_per_cycle = 8
        total_slots = slots_per_cycle * 2
        
        success = False
        
        print(f"Simulating {total_slots} slots...")
        
        for i in range(total_slots):
            slot_idx = i % slots_per_cycle
            
            # 1. Initiator Processing (Tick -> TX Queue)
            initiator.tick() # Updates slot, generates TX if slot 0
            
            # Get TX audio from Initiator
            tx_audio_i = np.zeros(0, dtype=np.float32)
            while not initiator.tx_queue.empty():
                chunk = initiator.tx_queue.get()
                tx_audio_i = np.concatenate([tx_audio_i, chunk])
            
            # If no TX, generate silence for slot duration
            if len(tx_audio_i) == 0:
                samples = int(cfg_init.sample_rate * cfg_init.slot_duration)
                tx_audio_i = np.zeros(samples, dtype=np.float32)
            
            # 2. Propagate I -> R
            rx_audio_r = chan_i2r.simulate_hf_channel(tx_audio_i)
            
            # 3. Responder Processing
            # Need to feed audio in chunks to simulate real-time
            chunk_size = 1024
            for j in range(0, len(rx_audio_r), chunk_size):
                chunk = rx_audio_r[j:j+chunk_size]
                responder.process_audio(chunk)
            
            responder.tick() # Updates slot, generates TX if slot 1 (and detected)
            
            # Get TX audio from Responder
            tx_audio_r = np.zeros(0, dtype=np.float32)
            while not responder.tx_queue.empty():
                chunk = responder.tx_queue.get()
                tx_audio_r = np.concatenate([tx_audio_r, chunk])
                
            # 4. Propagate R -> I
            if len(tx_audio_r) > 0:
                rx_audio_i = chan_r2i.simulate_hf_channel(tx_audio_r)
                
                # 5. Initiator Processing (RX)
                for j in range(0, len(rx_audio_i), chunk_size):
                    chunk = rx_audio_i[j:j+chunk_size]
                    initiator.process_audio(chunk)
            
            # Check Status
            # Initiator should see Responder in Slot 1
            if 1 in initiator.peers:
                peer = initiator.peers[1]
                if peer['rx_count'] > 0:
                    print(f"Slot {i}: Initiator detected Responder! SNR={peer['snr'][-1]:.1f}dB")
                    success = True
                    break
            
            # Responder should see Initiator in Slot 0
            if 0 in responder.peers:
                 peer = responder.peers[0]
                 # print(f"Slot {i}: Responder sees Initiator (SNR={peer['snr'][-1]:.1f}dB)")

        if success:
            print(f"\n>>> SUCCESS at {snr}dB! Link Established.")
        else:
            print(f"\n>>> FAILURE at {snr}dB. No link established.")

if __name__ == "__main__":
    run_deep_snr_test()
