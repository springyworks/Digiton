#!/usr/bin/env python3
"""
Real Echo Test V2 - Finegrained Slotting with Simulated WebSDR Delay

This app simulates a remote WebSDR receiver scenario:
1. TX at a configurable slot/frequency
2. Simulated receiver delay (0-1000ms) to mimic internet latency
3. Fine-grained frequency slots to detect mistuned SSB transceivers
4. I/Q processing over mono SSB audio channel

== How SSB I/Q Works Over Mono Audio ==
In SSB (Single Sideband), only ONE sideband is transmitted. The radio's BFO (Beat Frequency Oscillator)
mixes the incoming RF with a local carrier to produce audio. The KEY insight:

    - USB (Upper Sideband): Audio frequency = RF frequency - Carrier frequency
    - LSB (Lower Sideband): Audio frequency = Carrier frequency - RF frequency

When a signal at 1500Hz audio is transmitted via USB, it means the RF signal is 1500Hz ABOVE the
suppressed carrier. When received by a WebSDR tuned to the SAME frequency, the audio is also 1500Hz.

If the WebSDR is MISTUNED by +200Hz, the received audio will be at 1500-200 = 1300Hz (for USB).
This is why we need a BANK OF FREQUENCY DETECTORS to handle mistuning.

The I/Q analysis is done in SOFTWARE by:
1. Taking the mono audio from the SDR
2. Multiplying by a complex LO at the expected center frequency (e.g., 1500Hz) -> I/Q downconvert
3. Low-pass filtering to isolate the baseband signal
4. Analyzing the phase rotation for "spin" detection

DEV CONTROLS: Keyboard controls are for development only. In production, all parameters are automated.
"""

import time
import numpy as np
import sounddevice as sd
import queue
import sys
import os
import curses
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.core.ping import WaveletPingGenerator, WaveletPingDetector, SAMPLE_RATE
from digiton.sdr import iq_downconvert, detect_spin, detect_spin_robust

# ============================================================================
# Configuration
# ============================================================================
BLOCK_SIZE = 1024
TX_GAIN = 0.8

@dataclass
class SlotConfig:
    """Fine-grained slot and frequency configuration."""
    slot_duration_ms: int = 500        # Each slot is 500ms
    num_slots: int = 16                # 16 slots = 8 second cycle
    base_freq_hz: float = 1500.0       # Center frequency in audio band
    freq_step_hz: float = 100.0        # Frequency slot step
    num_freq_slots: int = 5            # -200, -100, 0, +100, +200 Hz
    
    # Deep Mode (4-ping sequence)
    deep_pings: int = 4
    deep_interval_ms: int = 80         # 80ms between pings in deep sequence

@dataclass
class SimConfig:
    """Simulated delay and channel parameters."""
    rx_delay_ms: float = 0.0           # Simulated WebSDR delay (0-1000ms)
    auto_delay: bool = True            # Auto-calibrate delay from self-echo
    
    # Detection
    snr_threshold_db: float = 18.0     # Min SNR for valid detection
    debounce_ms: float = 2000.0        # Min time between detections (increased to prevent noise spam)

# ============================================================================
# Frequency Bank
# ============================================================================
class FrequencyBank:
    """
    A bank of detectors for fine-grained frequency detection.
    This handles mistuned SSB receivers by searching multiple frequency offsets.
    """
    def __init__(self, config: SlotConfig, sample_rate: int):
        self.config = config
        self.fs = sample_rate
        self.generator = WaveletPingGenerator(sample_rate)
        
        # Generate templates for each frequency slot
        self.freq_offsets = []
        self.templates = {}
        self.detectors = {}
        
        half = config.num_freq_slots // 2
        for i in range(-half, half + 1):
            offset = i * config.freq_step_hz
            freq = config.base_freq_hz + offset
            self.freq_offsets.append(offset)
            
            # Generate deep sequence template
            seq, times, _ = self.generator.generate_ping_sequence(
                num_pings=config.deep_pings,
                ping_interval=config.deep_interval_ms / 1000.0,
                frequency=freq,
                padding=0.0
            )
            seq /= np.max(np.abs(seq))
            
            self.templates[offset] = seq
            self.detectors[offset] = WaveletPingDetector(sample_rate, template_signal=seq)
            
        self.ping_times = times  # Same for all
        
    def detect_best(self, audio: np.ndarray) -> Optional[Dict]:
        """
        Run all detectors and return the best match.
        Returns: {offset, snr_db, peak, peak_idx} or None
        """
        best = None
        best_snr = 0
        
        for offset, detector in self.detectors.items():
            try:
                peak, snr, _, peak_idx = detector.detect_presence(audio)
            except ValueError:
                peak, snr, _ = detector.detect_presence(audio)
                peak_idx = 0
                
            if snr > best_snr:
                best_snr = snr
                best = {
                    'offset': offset,
                    'freq': self.config.base_freq_hz + offset,
                    'snr': snr,
                    'snr_db': 20 * np.log10(snr) if snr > 0 else -100,
                    'peak': peak,
                    'peak_idx': peak_idx
                }
                
        return best

    def analyze_spin(self, audio: np.ndarray, peak_idx: int, freq: float) -> tuple:
        """
        Fine-grained spin analysis with accumulation across deep pings.
        """
        tpl_len = len(self.templates[0])
        seq_start = peak_idx - tpl_len // 2
        
        chunks = []
        ping_len = int(0.15 * self.fs)  # 150ms per ping
        
        for pt in self.ping_times:
            p_start = seq_start + int(pt * self.fs)
            p_end = p_start + ping_len
            if p_start >= 0 and p_end <= len(audio):
                chunk = audio[p_start:p_end]
                iq = iq_downconvert(chunk, freq, self.fs)
                chunks.append(iq)
                
        if chunks:
            return detect_spin_robust(chunks, self.fs)
        return None, 0.0

# ============================================================================
# Delay Line (Simulated WebSDR Latency)
# ============================================================================
class DelayLine:
    """
    Ring buffer delay line to simulate WebSDR/internet latency.
    Delay is adjustable in real-time.
    """
    def __init__(self, max_delay_ms: float, sample_rate: int):
        self.fs = sample_rate
        self.max_samples = int(max_delay_ms * sample_rate / 1000.0)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_ptr = 0
        self.delay_samples = 0
        
    def set_delay_ms(self, delay_ms: float):
        self.delay_samples = int(delay_ms * self.fs / 1000.0)
        self.delay_samples = min(self.delay_samples, self.max_samples - 1)
        
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Push samples in, get delayed samples out."""
        out = np.zeros_like(samples)
        
        for i, s in enumerate(samples):
            # Write
            self.buffer[self.write_ptr] = s
            
            # Read (delayed)
            read_ptr = (self.write_ptr - self.delay_samples) % self.max_samples
            out[i] = self.buffer[read_ptr]
            
            # Advance
            self.write_ptr = (self.write_ptr + 1) % self.max_samples
            
        return out

# ============================================================================
# Main Application
# ============================================================================
class RealEchoTestV2:
    def __init__(self, slot_cfg: SlotConfig, sim_cfg: SimConfig):
        self.slot_cfg = slot_cfg
        self.sim_cfg = sim_cfg
        self.fs = SAMPLE_RATE
        
        # DSP
        self.freq_bank = FrequencyBank(slot_cfg, self.fs)
        self.delay_line = DelayLine(1000.0, self.fs)  # Max 1s delay
        self.delay_line.set_delay_ms(sim_cfg.rx_delay_ms)
        
        # TX Generator
        self.generator = WaveletPingGenerator(self.fs)
        
        # Queues
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()
        
        # State
        self.rx_buffer = np.zeros(0, dtype=np.float32)
        self.last_detection_ts = 0.0
        self.last_tx_ts = 0.0
        self.cycle_start = 0.0
        self.current_slot = 0
        self.current_tx_freq_offset = 0
        
        # Statistics
        self.stats = {
            'tx_count': 0,
            'rx_count': 0,
            'success_count': 0,
            'detections': deque(maxlen=20),  # Last 20 detections
            'rms_history': deque(maxlen=100),
        }
        self.cycle_verified = False
        
        # Log buffer for TUI
        self.log_lines = deque(maxlen=15)
        
        # Setup Logging
        self._setup_logging()

    def _setup_logging(self):
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'echo_test.log')
        
        self.file_logger = logging.getLogger("EchoTest")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.handlers = []
        
        # 5MB limit, keep 1 backup
        handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=1)
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        
        self.file_logger.addHandler(handler)
        self.file_logger.info("=== Echo Test Session Started ===")
        
    def log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.log_lines.append(f"{ts} {msg}")
        if hasattr(self, 'file_logger'):
            self.file_logger.info(msg)
        
    def _generate_tx_signal(self, freq_offset: float) -> np.ndarray:
        """Generate a deep-mode ping sequence at given frequency offset."""
        freq = self.slot_cfg.base_freq_hz + freq_offset
        seq, _, _ = self.generator.generate_ping_sequence(
            num_pings=self.slot_cfg.deep_pings,
            ping_interval=self.slot_cfg.deep_interval_ms / 1000.0,
            frequency=freq,
            padding=0.0
        )
        return (seq / np.max(np.abs(seq)) * TX_GAIN).astype(np.float32)
        
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.log(f"Stream: {status}")
            
        # RX: Apply simulated delay ONLY if configured
        raw_in = indata.flatten()
        if self.sim_cfg.rx_delay_ms > 0:
            delayed = self.delay_line.process(raw_in)
            self.rx_queue.put(delayed)
        else:
            # Direct pass-through for Real World Testing
            self.rx_queue.put(raw_in)
        
        # TX
        try:
            data = self.tx_queue.get_nowait()
            if len(data) < frames:
                outdata[:len(data)] = data.reshape(-1, 1)
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:frames].reshape(-1, 1)
        except queue.Empty:
            outdata.fill(0)
            
    def _process_rx(self):
        """Process received audio."""
        while not self.rx_queue.empty():
            try:
                chunk = self.rx_queue.get_nowait()
            except queue.Empty:
                break
                
            # Update RMS
            rms = np.sqrt(np.mean(chunk**2))
            self.stats['rms_history'].append(rms)
            
            # Buffer
            self.rx_buffer = np.concatenate((self.rx_buffer, chunk))
            
            # Keep 2 seconds max
            keep = int(self.fs * 2.0)
            if len(self.rx_buffer) > keep:
                self.rx_buffer = self.rx_buffer[-keep:]
                
        # Detection (every tick)
        if len(self.rx_buffer) < self.fs * 0.5:
            return
            
        now = time.time()
        debounce_sec = self.sim_cfg.debounce_ms / 1000.0
        
        if now - self.last_detection_ts < debounce_sec:
            return
            
        det = self.freq_bank.detect_best(self.rx_buffer)
        if det is None:
            return
            
        snr_thresh = 10 ** (self.sim_cfg.snr_threshold_db / 20.0)
        if det['snr'] < snr_thresh or det['peak'] < 1e-3:
            return
            
        # Valid detection
        self.last_detection_ts = now
        self.stats['rx_count'] += 1
        
        if not self.cycle_verified:
            self.stats['success_count'] += 1
            self.cycle_verified = True
        
        # Spin analysis
        spin_dir, spin_freq = self.freq_bank.analyze_spin(
            self.rx_buffer, det['peak_idx'], det['freq']
        )
        
        # Slot calculation (accounting for delay)
        cycle_dur = (self.slot_cfg.slot_duration_ms * self.slot_cfg.num_slots) / 1000.0
        adjusted_time = now - (self.sim_cfg.rx_delay_ms / 1000.0)
        time_in_cycle = (adjusted_time - self.cycle_start) % cycle_dur
        arrival_slot = int(time_in_cycle / (self.slot_cfg.slot_duration_ms / 1000.0))
        
        # Is this our own echo?
        is_self = (arrival_slot == 0)
        
        # Auto-calibrate delay from self-echo
        if is_self and self.sim_cfg.auto_delay and self.stats['tx_count'] > 0:
            # Measure actual RTT
            raw_time = (now - self.cycle_start) % cycle_dur
            measured_delay_ms = raw_time * 1000.0
            
            # Smooth update
            old = self.sim_cfg.rx_delay_ms
            new = 0.8 * old + 0.2 * measured_delay_ms
            if abs(new - old) > 5.0:
                self.sim_cfg.rx_delay_ms = new
                self.delay_line.set_delay_ms(new)
                self.log(f"Auto-Delay: {old:.0f} -> {new:.0f}ms")
        
        # Log detection
        tag = "SELF" if is_self else "RX"
        spin_str = f"{spin_dir}({spin_freq:.0f}Hz)" if spin_dir else "?"
        self.log(f"{tag} Slot{arrival_slot:02d} | Δf={det['offset']:+.0f}Hz SNR={det['snr_db']:.1f}dB | Spin={spin_str}")
        
        self.stats['detections'].append({
            'time': now,
            'slot': arrival_slot,
            'offset': det['offset'],
            'snr_db': det['snr_db'],
            'spin': spin_dir,
            'is_self': is_self
        })
        
    def _tick(self):
        """Slot timing and TX logic."""
        now = time.time()
        
        cycle_dur = (self.slot_cfg.slot_duration_ms * self.slot_cfg.num_slots) / 1000.0
        time_in_cycle = (now - self.cycle_start) % cycle_dur
        new_slot = int(time_in_cycle / (self.slot_cfg.slot_duration_ms / 1000.0))
        
        if new_slot != self.current_slot:
            self.current_slot = new_slot
            
            # TX in Slot 0
            if new_slot == 0:
                self.cycle_verified = False
                # Choose frequency offset (cycle through them for testing)
                offsets = self.freq_bank.freq_offsets
                self.current_tx_freq_offset = offsets[self.stats['tx_count'] % len(offsets)]
                
                sig = self._generate_tx_signal(self.current_tx_freq_offset)
                for i in range(0, len(sig), BLOCK_SIZE):
                    self.tx_queue.put(sig[i:i+BLOCK_SIZE])
                    
                self.last_tx_ts = now
                self.stats['tx_count'] += 1
                self.log(f"TX Slot00 | Δf={self.current_tx_freq_offset:+.0f}Hz")
                
    def run_tui(self, stdscr):
        """Curses TUI main loop."""
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_CYAN, -1)
        stdscr.nodelay(True)
        
        self.cycle_start = time.time()
        
        with sd.Stream(samplerate=self.fs, blocksize=BLOCK_SIZE, channels=1,
                       callback=self._audio_callback):
            
            while True:
                self._tick()
                self._process_rx()
                
                # Input (DEV CONTROLS)
                key = stdscr.getch()
                if key == ord('q'):
                    break
                elif key == ord('+'):
                    self.sim_cfg.rx_delay_ms = min(1000, self.sim_cfg.rx_delay_ms + 50)
                    self.delay_line.set_delay_ms(self.sim_cfg.rx_delay_ms)
                    self.sim_cfg.auto_delay = False
                    self.log(f"[DEV] Delay: {self.sim_cfg.rx_delay_ms:.0f}ms (Auto OFF)")
                elif key == ord('-'):
                    self.sim_cfg.rx_delay_ms = max(0, self.sim_cfg.rx_delay_ms - 50)
                    self.delay_line.set_delay_ms(self.sim_cfg.rx_delay_ms)
                    self.sim_cfg.auto_delay = False
                    self.log(f"[DEV] Delay: {self.sim_cfg.rx_delay_ms:.0f}ms (Auto OFF)")
                elif key == ord('a'):
                    self.sim_cfg.auto_delay = True
                    self.log(f"[DEV] Auto-Delay ON")
                elif key == ord('t'):
                    # Threshold adjust
                    self.sim_cfg.snr_threshold_db = (self.sim_cfg.snr_threshold_db + 3) % 36
                    self.log(f"[DEV] SNR Thresh: {self.sim_cfg.snr_threshold_db:.0f}dB")
                elif key == ord('p'):
                    # Increase pings per sequence (Nasty SNR mode)
                    self.slot_cfg.deep_pings = min(16, self.slot_cfg.deep_pings + 1)
                    self.freq_bank = FrequencyBank(self.slot_cfg, self.fs) # Rebuild templates
                    self.log(f"[CFG] Pings/Seq -> {self.slot_cfg.deep_pings} (Better SNR)")
                elif key == ord('P'):
                    # Decrease pings per sequence
                    self.slot_cfg.deep_pings = max(1, self.slot_cfg.deep_pings - 1)
                    self.freq_bank = FrequencyBank(self.slot_cfg, self.fs) # Rebuild templates
                    self.log(f"[CFG] Pings/Seq -> {self.slot_cfg.deep_pings}")
                    
                # Draw
                self._draw_tui(stdscr)
                time.sleep(0.02)
                
    def _draw_tui(self, stdscr):
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        
        # Header
        auto_str = "AUTO" if self.sim_cfg.auto_delay else "MANUAL"
        header = f" ECHO TEST V2 | Delay: {self.sim_cfg.rx_delay_ms:.0f}ms ({auto_str}) | SNR>{self.sim_cfg.snr_threshold_db:.0f}dB | Slot: {self.current_slot:02d}/{self.slot_cfg.num_slots} "
        stdscr.addstr(0, 0, header[:w-1], curses.color_pair(1) | curses.A_REVERSE)
        
        # VU Meter
        if self.stats['rms_history']:
            rms = self.stats['rms_history'][-1]
            db = 20 * np.log10(rms) if rms > 1e-6 else -100
            db = max(-60, min(0, db))
            pct = (db + 60) / 60.0
            bar_w = w - 20
            fill = int(pct * bar_w)
            bar = "█" * fill + "-" * (bar_w - fill)
            color = curses.color_pair(2) if pct < 0.7 else (curses.color_pair(4) if pct < 0.9 else curses.color_pair(3))
            stdscr.addstr(2, 2, f"RX [{bar}] {db:5.1f}dB", color)
        
        # Frequency Bank Status
        stdscr.addstr(4, 2, "Freq Slots: ", curses.color_pair(5))
        for i, offset in enumerate(self.freq_bank.freq_offsets):
            freq_str = f"{offset:+.0f}"
            # Highlight if recent detection at this offset
            recent = [d for d in self.stats['detections'] if time.time() - d['time'] < 2.0 and d['offset'] == offset]
            if recent:
                stdscr.addstr(f"[{freq_str}]", curses.color_pair(2) | curses.A_BOLD)
            else:
                stdscr.addstr(f" {freq_str} ", curses.color_pair(1))
                
        # Stats
        tx = self.stats['tx_count']
        rx = self.stats['rx_count']
        success = self.stats['success_count']
        rate = (success / tx * 100.0) if tx > 0 else 0.0
        stdscr.addstr(6, 2, f"TX: {tx} | RX: {rx} | Success: {rate:.1f}% | Pings/Seq: {self.slot_cfg.deep_pings}", curses.color_pair(5))
        
        # Recent Detections
        stdscr.addstr(8, 2, "Recent:", curses.color_pair(5))
        for i, det in enumerate(list(self.stats['detections'])[-5:]):
            age = time.time() - det['time']
            tag = "SELF" if det['is_self'] else "RX"
            line = f"  {tag} S{det['slot']:02d} Δf={det['offset']:+.0f} SNR={det['snr_db']:.0f}dB {det['spin'] or '?'} ({age:.1f}s ago)"
            stdscr.addstr(9 + i, 2, line[:w-3])
        
        # Logs
        log_y = h - len(self.log_lines) - 2
        if log_y - 1 >= 0 and log_y - 1 < h:
            try:
                stdscr.addstr(log_y - 1, 0, "─" * (w - 1), curses.color_pair(1))
            except curses.error:
                pass
                
        for i, line in enumerate(self.log_lines):
            if log_y + i < h - 1 and log_y + i >= 0:
                try:
                    stdscr.addstr(log_y + i, 1, line[:w-2])
                except curses.error:
                    pass
                
        # Help
        stdscr.addstr(h-1, 0, " q:Quit  +/-:Delay  a:AutoDelay  t:Threshold  p/P:Pings(SNR) ", curses.color_pair(1) | curses.A_DIM)
        
        stdscr.refresh()

# ============================================================================
# Entry Point
# ============================================================================
def main():
    slot_cfg = SlotConfig()
    # Default to NO SIMULATION for real-world testing
    sim_cfg = SimConfig(rx_delay_ms=0.0, auto_delay=False)
    
    app = RealEchoTestV2(slot_cfg, sim_cfg)
    curses.wrapper(app.run_tui)

if __name__ == "__main__":
    main()
