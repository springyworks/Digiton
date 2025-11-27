#!/usr/bin/env python3
"""
WAVELET MODEM - LISTEN MODE
============================
A passive receiver for monitoring Wavelet Party Protocol traffic.

Usage:
    python3 modem_listen.py <wav_file>

The modem will:
1. Detect the Master's CQ-Ping rhythm
2. Identify response slots (potential stations)
3. Display a visual grid of activity
4. Report detected stations and their signal strengths

This is the "WebSDR Listener" mode - you hear the party, you don't join.
"""

import numpy as np
import scipy.io.wavfile as wav
import sys
from hf_channel_simulator import WaveletPingDetector, WaveletPingGenerator, SAMPLE_RATE

# Constants (must match protocol)
SLOT_DURATION = 0.4        # Seconds
PING_DURATION = 0.2        # Seconds
CQ_INTERVAL = 8            # Master CQ every 8 slots

def analyze_slot(audio_chunk, template, slot_idx):
    """
    Analyze a single time slot for wavelet ping presence.
    
    Returns:
        detected: Boolean
        snr_db: SNR in dB
        peak: Raw correlation peak
    """
    detector = WaveletPingDetector(template_signal=template)
    _, _, corr = detector.detect_presence(audio_chunk)
    
    peak = np.max(np.abs(corr))
    noise = np.median(np.abs(corr))
    
    # Avoid division by zero
    if noise > 0:
        snr_db = 20 * np.log10(peak / noise)
    else:
        snr_db = 0
        
    # Detection threshold
    detected = snr_db > 10.0
    
    return detected, snr_db, peak

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
                slots_display.append(f"  --    ")
    
    # Print slots in a row
    print("│ ", end="")
    for i, disp in enumerate(slots_display):
        print(f"{disp:>10}", end=" ")
    print("│")
    print("└" + "─" * 73 + "┘")

def listen_to_wav(wav_file):
    """
    Main listening function: decode a WAV file and display WPP activity.
    """
    print("=" * 75)
    print("WAVELET MODEM - LISTEN MODE")
    print("=" * 75)
    print(f"Input File: {wav_file}")
    
    # Load WAV
    try:
        sr, audio = wav.read(wav_file)
    except Exception as e:
        print(f"ERROR: Could not read WAV file: {e}")
        return
        
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")
    print(f"Channels: {'Stereo' if len(audio.shape) > 1 else 'Mono'}")
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        
    # Convert to float
    audio = audio.astype(float) / 32768.0
    
    # Generate template (single ping)
    generator = WaveletPingGenerator(sr)
    template = generator.generate_ping(wavelet_name='morl')
    
    print(f"\nTemplate Length: {len(template)} samples ({len(template)/sr:.3f}s)")
    print(f"Slot Duration: {SLOT_DURATION}s")
    print(f"CQ Interval: {CQ_INTERVAL} slots\n")
    
    # Calculate number of complete cycles
    slot_samples = int(SLOT_DURATION * sr)
    num_slots = len(audio) // slot_samples
    num_cycles = num_slots // CQ_INTERVAL
    
    print(f"Total Slots: {num_slots}")
    print(f"Complete Cycles: {num_cycles}\n")
    
    # Analysis loop
    master_detections = []
    station_activity = {i: [] for i in range(1, CQ_INTERVAL)}  # Slots 1-7
    
    for cycle in range(num_cycles):
        cycle_detections = []
        
        for slot in range(CQ_INTERVAL):
            global_slot = cycle * CQ_INTERVAL + slot
            start_sample = global_slot * slot_samples
            end_sample = start_sample + slot_samples
            
            if end_sample > len(audio):
                break
                
            chunk = audio[start_sample:end_sample]
            detected, snr_db, peak = analyze_slot(chunk, template, slot)
            
            cycle_detections.append((detected, snr_db, peak))
            
            # Track activity
            if slot == 0 and detected:
                master_detections.append((cycle, snr_db))
            elif slot > 0 and detected:
                station_activity[slot].append((cycle, snr_db))
        
        # Visualize this cycle
        visualize_grid(cycle_detections, cycle)
    
    # Summary Report
    print("\n" + "=" * 75)
    print("DETECTION SUMMARY")
    print("=" * 75)
    
    print(f"\n🎯 MASTER (CQ-Ping) Detected in {len(master_detections)}/{num_cycles} cycles:")
    if master_detections:
        avg_snr = np.mean([snr for _, snr in master_detections])
        print(f"   Average SNR: {avg_snr:.1f} dB")
        print(f"   Cycles: {[cycle for cycle, _ in master_detections]}")
    
    print(f"\n📡 STATION ACTIVITY:")
    active_slots = 0
    real_stations = []
    for slot in range(1, CQ_INTERVAL):
        activity = station_activity[slot]
        if activity:
            active_slots += 1
            avg_snr = np.mean([snr for _, snr in activity])
            std_snr = np.std([snr for _, snr in activity]) if len(activity) > 1 else 0
            
            # Classify: Real station vs Artifact
            # Real stations: consistent SNR across cycles (low variance)
            # Artifacts: sporadic or highly variable SNR
            is_real = len(activity) >= num_cycles * 0.5 and std_snr < 3.0
            
            marker = "⭐" if is_real else "💭"
            station_type = "STATION" if is_real else "artifact"
            
            print(f"   {marker} Slot {slot}: {len(activity)}/{num_cycles} cycles, Avg SNR: {avg_snr:.1f}±{std_snr:.1f} dB [{station_type}]")
            print(f"           Cycles: {[cycle for cycle, _ in activity]}")
            
            if is_real:
                real_stations.append(slot)
    
    if active_slots == 0:
        print("   (No station activity detected)")
    
    print("\n" + "=" * 75)
    print(f"Total Detections: {active_slots} slots")
    print(f"Real Stations (consistent): {len(real_stations)} → Slots {real_stations}")
    print("=" * 75)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 modem_listen.py <wav_file>")
        print("\nExample:")
        print("  python3 modem_listen.py wavelet_party_websdr.wav")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    listen_to_wav(wav_file)
