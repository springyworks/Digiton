import curses
import time
import numpy as np
import sounddevice as sd
import queue
import sys
import os
import logging
from logging.handlers import QueueHandler

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.protocol.wpp import WaveletPartyProtocol, StationConfig, ProtocolMode
import random
import json
from datetime import datetime

# Configuration
BLOCK_SIZE = 4096
SAMPLE_RATE = 48000

# Setup Logging Queue - suppress console output, route to TUI only
log_queue = queue.Queue()
queue_handler = QueueHandler(log_queue)

# Configure WPP logger
logger = logging.getLogger("WPP")
logger.handlers.clear()  # Remove any existing handlers
logger.addHandler(queue_handler)
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger

# Configure root logger to suppress console output
root_logger = logging.getLogger()
root_logger.handlers.clear()  # Remove default handlers (console output)
root_logger.addHandler(queue_handler)
root_logger.setLevel(logging.INFO)

def draw_vu_meter(stdscr, y, x, width, rms_val):
    """Draw a horizontal VU meter."""
    # Log scale for VU
    if rms_val < 1e-6:
        db = -100
    else:
        db = 20 * np.log10(rms_val)
    
    # Range: -60dB to 0dB
    min_db = -60
    max_db = 0
    
    percent = (db - min_db) / (max_db - min_db)
    percent = max(0.0, min(1.0, percent))
    
    fill_len = int(width * percent)
    
    bar = "█" * fill_len + "-" * (width - fill_len)
    
    color = curses.color_pair(2) # Green
    if percent > 0.8:
        color = curses.color_pair(3) # Red/Warning
    elif percent > 0.6:
        color = curses.color_pair(4) # Yellow
        
    stdscr.addstr(y, x, f"[{bar}] {db:.1f} dB", color)

def main(stdscr):
    # Curses Setup
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_CYAN, -1)
    
    stdscr.nodelay(True) # Non-blocking input
    stdscr.scrollok(False) # Disable scrolling
    stdscr.idlok(False)

    # Protocol Setup
    # Generate a random callsign at startup (e.g., STN-4821)
    rand_callsign = f"STN-{random.randint(1000, 9999)}"

    # Aggregated logging setup (JSON lines)
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass
    agg_log_path = os.path.abspath(os.path.join(logs_dir, 'wpp_sessions.log'))

    def write_agg_log(event, payload=None):
        entry = {
            'ts': datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
            'id': rand_callsign,
            'event': event,
        }
        if payload:
            try:
                entry.update(payload)
            except Exception:
                entry['payload'] = str(payload)
        try:
            with open(agg_log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Fallback: ignore file write errors
            pass

    # Provide structured emitter to protocol for autonomous events
    def emit_structured(payload):
        base = {
            'ts': datetime.utcnow().isoformat(timespec='milliseconds') + 'Z',
            'id': rand_callsign,
            'slot': getattr(protocol, 'current_slot', 0),
            'state': protocol.state.name,
            'mode': protocol.mode.name,
            'lat_ms': float(config.remote_latency_ms),
        }
        try:
            base.update(payload)
        except Exception:
            base['payload'] = str(payload)
        try:
            with open(agg_log_path, 'a') as f:
                f.write(json.dumps(base) + "\n")
        except Exception:
            pass
    # Fully autonomous defaults; no end-user settings
    config = StationConfig(
        callsign=rand_callsign,
        sample_rate=SAMPLE_RATE,
        slot_duration=1.5,
        slots_per_cycle=8,
        remote_latency_ms=0.0,
        monitor_self=True,
        auto_mode=True,
        auto_latency=True
    )
    
    protocol = WaveletPartyProtocol(config)
    protocol.mode = ProtocolMode.DEEP
    # Attach structured emitter so protocol can write events autonomously
    setattr(protocol, 'agg_log_writer', emit_structured)
    
    # Audio State
    tx_buffer = np.zeros(0)
    tune_phase = 0.0  # Track phase for continuous tone
    # UI dynamic state
    log_lines = []
    max_log_lines = 10
    tune_mode = False
    tune_frequency = 1000  # Hz
    noise_level = 0.0  # Injected Gaussian noise amplitude into RX path
    spin_cycle = ['center', 'left', 'right']
    spin_idx = 0

    def callback(indata, outdata, frames, time_info, status):
        nonlocal tx_buffer, tune_phase, noise_level
        # RX with optional noise injection
        rx = indata.flatten().astype(np.float32)
        if noise_level > 0.0:
            rx += np.random.normal(0.0, noise_level, size=rx.shape).astype(np.float32)
        protocol.process_audio(rx)
        
        # TX
        if tune_mode:
            # Generate continuous test tone with phase tracking
            t = np.arange(frames) / SAMPLE_RATE
            phase = 2 * np.pi * tune_frequency * t + tune_phase
            tone = 0.15 * np.sin(phase)  # 15% amplitude
            tune_phase = (tune_phase + 2 * np.pi * tune_frequency * frames / SAMPLE_RATE) % (2 * np.pi)
            outdata[:] = tone.reshape(-1, 1)
        else:
            # Normal protocol operation
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
                tx_buffer = np.zeros(0)

    # Start Audio Stream
    stream = sd.Stream(samplerate=SAMPLE_RATE,
                       blocksize=BLOCK_SIZE,
                       channels=1,
                       dtype='float32',
                       latency='high',
                       callback=callback)
    stream.start()
    protocol.cycle_start_time = int(time.time())  # satisfy type checker (expects int)
    
    chat_mode = False
    chat_buffer = ""

    try:
        while True:
            protocol.tick()
            
            # Handle Input
            key = stdscr.getch()
            if key == ord('q'):
                break
            # DEV CONTROLS (Overrides Automation)
            elif key == ord('t'):
                tune_mode = not tune_mode
                if tune_mode:
                    tx_buffer = np.zeros(0)  # Clear buffer when entering tune mode
                logger.info(f"[DEV] Tune Mode: {'ON' if tune_mode else 'OFF'} @ {tune_frequency}Hz")
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                presets = {
                    ord('1'): 800,
                    ord('2'): 1000,
                    ord('3'): 1200,
                    ord('4'): 1500,
                    ord('5'): 1800,
                }
                tune_frequency = presets.get(key, tune_frequency)
                tune_phase = 0.0  # reset phase to avoid clicks
                logger.info(f"[DEV] Tune Frequency set to {tune_frequency}Hz")
            elif key == ord('c'):
                chat_mode = not chat_mode
                if chat_mode:
                    chat_buffer = ""
            elif chat_mode and key != -1:
                if key in (10, 13):  # Enter
                    if chat_buffer.strip():
                        protocol.tx_chat_queue.put(chat_buffer.strip())
                        chat_buffer = ""
                    chat_mode = False
                elif key == 27:  # ESC
                    chat_mode = False
                    chat_buffer = ""
                elif key in (8, 127):  # Backspace
                    chat_buffer = chat_buffer[:-1]
                else:
                    try:
                        chat_buffer += chr(key)
                    except ValueError:
                        pass
            elif key == ord('m'):
                protocol.config.monitor_self = not protocol.config.monitor_self
                logger.info(f"[DEV] Monitor Self: {protocol.config.monitor_self}")
            elif key == ord('s'):
                spin_idx = (spin_idx + 1) % len(spin_cycle)
                protocol.forced_spin = spin_cycle[spin_idx]  # type: ignore[attr-defined]
                logger.info(f"[DEV] Forced spin: {protocol.forced_spin}")
            elif key == ord('n'):
                noise_level = min(0.5, noise_level + 0.005)
                logger.info(f"[DEV] Noise level: {noise_level:.3f}")
            elif key == ord('N'):
                noise_level = max(0.0, noise_level - 0.005)
                logger.info(f"[DEV] Noise level: {noise_level:.3f}")
            elif key == ord(']'):
                modes_order = [ProtocolMode.RESCUE, ProtocolMode.DEEP, ProtocolMode.NORMAL, ProtocolMode.FAST, ProtocolMode.TURBO]
                cur_i = modes_order.index(protocol.mode)
                if cur_i < len(modes_order) - 1:
                    protocol.mode = modes_order[cur_i + 1]
                    protocol.mode_lock_until = 0
                    protocol.config.auto_mode = False
                    logger.info(f"[DEV] Mode -> {protocol.mode.name}")
            elif key == ord('['):
                modes_order = [ProtocolMode.RESCUE, ProtocolMode.DEEP, ProtocolMode.NORMAL, ProtocolMode.FAST, ProtocolMode.TURBO]
                cur_i = modes_order.index(protocol.mode)
                if cur_i > 0:
                    protocol.mode = modes_order[cur_i - 1]
                    protocol.mode_lock_until = 0
                    protocol.config.auto_mode = False
                    logger.info(f"[DEV] Mode -> {protocol.mode.name}")
            elif key == ord('d'):
                # Toggle Auto-Mode off if manual override used
                protocol.config.auto_mode = False
                if protocol.mode == ProtocolMode.DEEP:
                    protocol.mode = ProtocolMode.FAST
                else:
                    protocol.mode = ProtocolMode.DEEP
                logger.info(f"[DEV] Mode switched to: {protocol.mode.name} (Auto-Mode Disabled)")
            elif key == ord('a'):
                protocol.config.auto_mode = True
                logger.info(f"Auto-Mode Re-enabled")
            elif key == ord('+') or key == ord('='):
                protocol.config.auto_latency = False
                protocol.config.remote_latency_ms += 100
                logger.info(f"[DEV] Latency: {protocol.config.remote_latency_ms}ms (Auto-Latency Disabled)")
            elif key == ord('-') or key == ord('_'):
                protocol.config.auto_latency = False
                protocol.config.remote_latency_ms = max(0, protocol.config.remote_latency_ms - 100)
                logger.info(f"[DEV] Latency: {protocol.config.remote_latency_ms}ms (Auto-Latency Disabled)")

            # Process Logs (and mirror to aggregated logfile)
            try:
                while True:
                    record = log_queue.get_nowait()
                    msg = f"{time.strftime('%H:%M:%S')} {record.getMessage()}"
                    log_lines.append(msg)
                    # Write structured log line with best-effort parsing
                    snr_val = getattr(protocol, 'current_snr_estimate', None)
                    snr_out = None
                    try:
                        if snr_val is not None:
                            snr_out = float(snr_val)
                    except Exception:
                        snr_out = None
                    write_agg_log('log', {
                        'message': record.getMessage(),
                        'level': getattr(record, 'levelname', 'INFO'),
                        'state': protocol.state.name,
                        'mode': protocol.mode.name,
                        'lat_ms': float(config.remote_latency_ms),
                        'snr_db': snr_out,
                    })
                    if len(log_lines) > max_log_lines:
                        log_lines.pop(0)
            except queue.Empty:
                pass

            # Draw UI
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            
            # Header
            auto_status = "AUTO" if config.auto_mode else "MANUAL"
            header = f" WPP TERMINAL | Callsign: {config.callsign} | State: {protocol.state.name} | Mode: {protocol.mode.name} ({auto_status}) | Latency: {config.remote_latency_ms:.0f}ms "
            # Periodically mirror header state to agg log (every UI tick)
            emit_structured({'event': 'ui_tick'})
            stdscr.addstr(0, 0, header[:w-1], curses.color_pair(1) | curses.A_REVERSE)
            
            # Status
            snr_display = f"SNR={protocol.current_snr_estimate:.1f}dB" if getattr(protocol, 'current_snr_estimate', None) is not None else "SNR=?"
            status_line = (
                f"Slot {protocol.current_slot}/{config.slots_per_cycle} | Mode {protocol.mode.name} | Spin {(protocol.forced_spin or 'auto'):>6} | "
                f"{snr_display} | Noise {noise_level:.3f} | Mon {'Y' if config.monitor_self else 'N'} | AutoMode {'Y' if config.auto_mode else 'N'} | LatAuto {'Y' if config.auto_latency else 'N'}"
            )
            if tune_mode:
                status_line += f" | TUNE {tune_frequency}Hz"
            if chat_mode:
                status_line += " | CHAT: type and Enter"
            stdscr.addstr(1, 0, status_line[:w-1])
            
            # VU Meter
            rms = protocol.get_current_rms()
            draw_vu_meter(stdscr, 3, 2, w - 20, rms)
            
            # Peers
            stdscr.addstr(5, 0, "--- PEERS ---", curses.color_pair(5))
            peers = protocol.get_peer_summary()
            for i, p in enumerate(peers):
                if 6 + i >= h - max_log_lines - 2: break
                p_str = f"Slot {p['slot']}: SNR={p['avg_snr']:.1f}dB Δf={p['avg_freq_offset']:.1f}Hz Conf={p['confidence']:.2f}"
                stdscr.addstr(6 + i, 2, p_str[:w-3])
                
            # Logs
            log_y = h - max_log_lines - 1
            if log_y > 6:
                stdscr.addstr(log_y, 0, "--- LOGS ---", curses.color_pair(5))
                for i, line in enumerate(log_lines):
                    if log_y + 1 + i < h - 2:  # Reserve bottom 2 lines for help bar
                        try:
                            stdscr.addstr(log_y + 1 + i, 0, line[:w-1])
                        except curses.error:
                            pass # Ignore error at bottom-right corner
                # Show last received chat messages
                chat_display_start = log_y - 3
                if chat_display_start > 6:
                    stdscr.addstr(chat_display_start, 0, "--- CHAT ---", curses.color_pair(5))
                    # drain without blocking; only peek last 3
                    chats = []
                    try:
                        while True:
                            chats.append(protocol.rx_chat_queue.get_nowait())
                            if len(chats) > 20:
                                chats.pop(0)
                    except queue.Empty:
                        pass
                    for i, msg in enumerate(chats[-3:]):
                        try:
                            stdscr.addstr(chat_display_start + 1 + i, 0, str(msg)[:w-1])
                        except curses.error:
                            pass
            
            # Help bar at bottom (2 rows)
            if h > 2:
                help_text1 = "q quit | t tune | 1-5 tone | s spin | [/] mode +/- | m monitor | a auto | c chat"
                help_text2 = "+/- latency | n/N noise +/- | pipewire multi-instance | state: LISTEN/DISCOVER/CONNECT"
                try:
                    stdscr.addstr(h-2, 0, help_text1.ljust(w-1)[:w-1], curses.color_pair(1) | curses.A_DIM)
                    stdscr.addstr(h-1, 0, help_text2.ljust(w-1)[:w-1], curses.color_pair(1) | curses.A_DIM)
                except curses.error:
                    pass
            
            stdscr.refresh()
            time.sleep(0.05)
            
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    curses.wrapper(main)
