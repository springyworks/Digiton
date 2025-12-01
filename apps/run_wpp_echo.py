#!/usr/bin/env python3
"""
WPP Protocol Echo Test Runner

Runs the Wavelet Party Protocol with self-echo detection for testing:
- Automatic speed throttling based on measured SNR
- Simulated delay line to mimic WebSDR latency
- Fine-grained frequency slot detection for mistuned receivers
- TUI with VU meter and status display

Usage:
    python3 apps/run_wpp_echo.py [--delay 500]

Add noise via pavucontrol to test robustness.

DEV CONTROLS: Keyboard controls are for development/debugging only.
In production, all parameters are automated based on channel conditions.
"""

import time
import numpy as np
import sounddevice as sd
import queue
import sys
import os
import curses
import argparse
from collections import deque
import signal

# Force CPU thread usage to 10 cores (must be set before numpy import)
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.protocol.wpp import (
    WaveletPartyProtocol, StationConfig, ProtocolMode, MODE_PARAMS
)

# Configuration
BLOCK_SIZE = 1024

def draw_vu_meter(stdscr, y, x, width, rms_val):
    """Draw a horizontal VU meter with color."""
    if rms_val < 1e-6:
        db = -100
    else:
        db = 20 * np.log10(rms_val)
    
    db = max(-60, min(0, db))
    pct = (db + 60) / 60.0
    
    fill_len = int(width * pct)
    bar = "█" * fill_len + "-" * (width - fill_len)
    
    color = curses.color_pair(2)  # Green
    if pct > 0.9:
        color = curses.color_pair(3)  # Red
    elif pct > 0.7:
        color = curses.color_pair(4)  # Yellow
        
    stdscr.addstr(y, x, f"[{bar}] {db:5.1f}dB", color)

def draw_mode_indicator(stdscr, y, x, mode):
    """Draw speed mode indicator with appropriate color."""
    mode_colors = {
        ProtocolMode.TURBO: curses.color_pair(5),   # Cyan (fastest)
        ProtocolMode.FAST: curses.color_pair(2),    # Green
        ProtocolMode.NORMAL: curses.color_pair(4),  # Yellow
        ProtocolMode.DEEP: curses.color_pair(6),    # Magenta
        ProtocolMode.RESCUE: curses.color_pair(3),  # Red (slowest)
    }
    color = mode_colors.get(mode, curses.color_pair(1))
    stdscr.addstr(y, x, f" {mode.name:^8} ", color | curses.A_BOLD)

def main(stdscr, args):
    # Curses setup
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_CYAN, -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    stdscr.nodelay(True)
    
    # Protocol Setup
    config = StationConfig(
        callsign="ECHO_TEST",
        sample_rate=48000,
        slot_duration=1.5,
        slots_per_cycle=8,
        remote_latency_ms=0.0,
        sim_delay_ms=args.delay,
        monitor_self=True,
        auto_mode=True,
        auto_latency=True,
    )
    
    protocol = WaveletPartyProtocol(config)
    
    # TX buffer for audio callback
    tx_buffer = np.zeros(0, dtype=np.float32)
    rx_queue = queue.Queue()
    
    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal tx_buffer
        if status:
            pass  # Log handled by protocol
            
        # RX: Queue for main thread
        rx_queue.put(indata.copy())
        
        # TX: Drain queue to buffer
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
            valid = len(tx_buffer)
            outdata[:valid] = tx_buffer.reshape(-1, 1)
            outdata[valid:] = 0
            tx_buffer = np.zeros(0, dtype=np.float32)
    
    # Log buffer
    log_lines = deque(maxlen=12)
    
    # Capture protocol logs (disable console output, TUI only)
    import logging
    class TUILogHandler(logging.Handler):
        def emit(self, record):
            msg = f"{time.strftime('%H:%M:%S')} {record.getMessage()}"
            log_lines.append(msg)
    
    wpp_logger = logging.getLogger("WPP")
    wpp_logger.handlers.clear()  # Remove default console handler
    wpp_logger.addHandler(TUILogHandler())
    wpp_logger.setLevel(logging.INFO)
    wpp_logger.propagate = False  # Don't propagate to root logger
    
    # Start audio
    stream = sd.Stream(
        samplerate=config.sample_rate,
        blocksize=BLOCK_SIZE,
        channels=1,
        callback=audio_callback
    )
    stream.start()
    protocol.cycle_start_time = time.time()

    running = True
    try:
        while running:
            # Process Audio from Queue (limit to 5 chunks per cycle to prevent hang)
            chunks_processed = 0
            try:
                while chunks_processed < 5:
                    chunk = rx_queue.get_nowait()
                    protocol.process_audio(chunk.flatten())
                    chunks_processed += 1
            except queue.Empty:
                pass

            protocol.tick()

            # Handle input (DEV CONTROLS)
            key = stdscr.getch()
            
            # Sleep to prevent busy loop and reduce CPU usage
            time.sleep(0.02)
            
            if key in (ord('q'), ord('Q'), 27):  # q, Q, or ESC
                running = False
            elif key == ord('+'):
                config.sim_delay_ms = min(1000, config.sim_delay_ms + 50)
                protocol.delay_line.set_delay_ms(config.sim_delay_ms)
                config.auto_latency = False
                log_lines.append(f"[DEV] Sim Delay: {config.sim_delay_ms:.0f}ms (Auto OFF)")
            elif key == ord('-'):
                config.sim_delay_ms = max(0, config.sim_delay_ms - 50)
                protocol.delay_line.set_delay_ms(config.sim_delay_ms)
                config.auto_latency = False
                log_lines.append(f"[DEV] Sim Delay: {config.sim_delay_ms:.0f}ms (Auto OFF)")
            elif key == ord('a'):
                config.auto_mode = not config.auto_mode
                log_lines.append(f"[DEV] Auto-Mode: {'ON' if config.auto_mode else 'OFF'}")
            elif key == ord('m'):
                config.monitor_self = not config.monitor_self
                log_lines.append(f"[DEV] Monitor Self: {'ON' if config.monitor_self else 'OFF'}")
            elif key == ord('1'):
                protocol.mode = ProtocolMode.TURBO
                config.auto_mode = False
                log_lines.append("[DEV] Forced TURBO mode")
            elif key == ord('2'):
                protocol.mode = ProtocolMode.FAST
                config.auto_mode = False
                log_lines.append("[DEV] Forced FAST mode")
            elif key == ord('3'):
                protocol.mode = ProtocolMode.NORMAL
                config.auto_mode = False
                log_lines.append("[DEV] Forced NORMAL mode")
            elif key == ord('4'):
                protocol.mode = ProtocolMode.DEEP
                config.auto_mode = False
                log_lines.append("[DEV] Forced DEEP mode")
            elif key == ord('5'):
                protocol.mode = ProtocolMode.RESCUE
                config.auto_mode = False
                log_lines.append("[DEV] Forced RESCUE mode")
            
            # Draw UI (throttled to ~20 FPS)
            draw_ui(stdscr, protocol, config, log_lines)

    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        pass
    finally:
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        # Clear screen and show goodbye message
        stdscr.nodelay(False)
        stdscr.erase()
        stdscr.addstr(0, 0, "Exiting WPP Echo Test...", curses.A_BOLD)
        stdscr.refresh()

def draw_ui(stdscr, protocol, config, log_lines):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    
    # Header
    auto_str = "AUTO" if config.auto_mode else "MANUAL"
    header = " WPP ECHO TEST | Mode: "
    try:
        stdscr.addstr(0, 0, header[:max(0, w-1)], curses.color_pair(1) | curses.A_REVERSE)
    except curses.error:
        pass
    draw_mode_indicator(stdscr, 0, len(header), protocol.mode)
    
    mode_end = len(header) + 10
    status = f" ({auto_str}) | SimDelay: {config.sim_delay_ms:.0f}ms | Latency: {config.remote_latency_ms:.0f}ms "
    try:
        stdscr.addstr(0, mode_end, status[:max(0, w - mode_end - 1)], curses.color_pair(1) | curses.A_REVERSE)
    except curses.error:
        pass
    
    # Slot indicator
    cycle_dur = config.slot_duration * config.slots_per_cycle
    time_in_cycle = (time.time() - protocol.cycle_start_time) % cycle_dur
    slot_progress = (time_in_cycle % config.slot_duration) / config.slot_duration
    
    slot_str = f"Slot: {protocol.current_slot:02d}/{config.slots_per_cycle} "
    try:
        stdscr.addstr(1, 2, slot_str[:max(0, w-3)])
    except curses.error:
        pass
    
    # Slot progress bar
    slot_bar_w = 20
    slot_fill = int(slot_progress * slot_bar_w)
    slot_bar = "▓" * slot_fill + "░" * (slot_bar_w - slot_fill)
    try:
        stdscr.addstr(1, len(slot_str) + 3, f"[{slot_bar}]"[:max(0, w - (len(slot_str) + 4))], curses.color_pair(5))
    except curses.error:
        pass
    
    # VU Meter
    rms = protocol.get_current_rms()
    try:
        draw_vu_meter(stdscr, 3, 2, max(10, w - 20), rms)
    except curses.error:
        pass
    
    # Self SNR History
    if protocol.self_snr_history:
        avg_snr = np.mean(protocol.self_snr_history[-5:])
        snr_str = f"Self SNR: {avg_snr:.1f}dB (n={len(protocol.self_snr_history)})"
        try:
            stdscr.addstr(4, 2, snr_str[:max(0, w-3)], curses.color_pair(2))
        except curses.error:
            pass
    else:
        try:
            stdscr.addstr(4, 2, "Self SNR: Waiting for echo..."[:max(0, w-3)], curses.color_pair(4))
        except curses.error:
            pass
    
    # Speed Mode Thresholds
    stdscr.addstr(6, 2, "Speed Modes:", curses.color_pair(5))
    for i, mode in enumerate([ProtocolMode.TURBO, ProtocolMode.FAST, ProtocolMode.NORMAL, 
                              ProtocolMode.DEEP, ProtocolMode.RESCUE]):
        params = MODE_PARAMS[mode]
        is_current = (mode == protocol.mode)
        marker = "►" if is_current else " "
        line = f"{marker} {mode.name:7} SNR>{params['snr_min']:4.0f}dB  {params['pings']}p/{params['interval']*1000:.0f}ms"
        color = curses.color_pair(2) if is_current else curses.color_pair(1)
        try:
            stdscr.addstr(7 + i, 4, line[:max(0, w-5)], color | (curses.A_BOLD if is_current else 0))
        except curses.error:
            pass
    
    # Peers
    peers = protocol.get_peer_summary()
    try:
        stdscr.addstr(13, 2, f"Peers ({len(peers)}):"[:max(0, w-3)], curses.color_pair(5))
    except curses.error:
        pass
    for i, p in enumerate(peers[:3]):
        p_str = f"  Slot {p['slot']:02d}: SNR={p['avg_snr']:.1f}dB Δf={p['avg_freq_offset']:.0f}Hz Conf={p['confidence']:.2f}"
        try:
            stdscr.addstr(14 + i, 2, p_str[:max(0, w-3)])
        except curses.error:
            pass
    
    # Logs
    log_y = h - len(log_lines) - 2
    try:
        stdscr.addstr(log_y - 1, 0, ("─" * max(0, w-1)))
    except curses.error:
        pass
    for i, line in enumerate(log_lines):
        if log_y + i < h - 1:
            # Truncate and display
            try:
                stdscr.addstr(log_y + i, 1, line[:max(0, w-2)])
            except curses.error:
                pass
    
    # Help (truncate to terminal width to avoid curses ERR)
    help_str = " q:Quit  +/-:Delay  a:AutoMode  m:Monitor  1-5:ForceMode "
    try:
        if h > 0:
            stdscr.addstr(h-1, 0, help_str[:max(0, w-1)], curses.color_pair(1) | curses.A_DIM)
    except curses.error:
        pass
    
    stdscr.refresh()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WPP Echo Test with Speed Throttling")
    parser.add_argument('--delay', type=float, default=0.0, 
                        help='Initial simulated delay in ms (0-1000)')
    args = parser.parse_args()
    
    curses.wrapper(lambda stdscr: main(stdscr, args))
