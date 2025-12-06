import time
import numpy as np
import queue
import random
import logging
from enum import Enum
from dataclasses import dataclass

from digiton.core.ping import WaveletPingGenerator, WaveletPingDetector, SAMPLE_RATE
from digiton.sdr import iq_downconvert, detect_spin_robust

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("WPP")

class ProtocolState(Enum):
    IDLE = 0
    LISTENING = 1
    DISCOVERY_TX = 2  # Sending CQ
    DISCOVERY_RX = 3  # Listening for replies
    CONNECTED = 4

class ProtocolMode(Enum):
    TURBO = 0   # Ultra-fast single ping, short slots (very high SNR, close range)
    FAST = 1    # Single Ping (High SNR)
    NORMAL = 2  # 2-ping sequence (Moderate SNR)
    DEEP = 3    # 4-ping sequence (Low SNR, far away)
    RESCUE = 4  # 8-ping sequence (Very low SNR, extreme conditions)

@dataclass
class StationConfig:
    callsign: str
    sample_rate: int = SAMPLE_RATE
    base_freq: float = 1500.0
    slot_duration: float = 1.5
    slots_per_cycle: int = 8
    remote_latency_ms: float = 0.0  # Latency of remote receiver (e.g. WebSDR)
    monitor_self: bool = True       # Whether to detect own transmissions (for echo test)
    auto_mode: bool = True          # Automatically switch modes based on SNR
    auto_latency: bool = True       # Automatically calibrate latency via self-monitoring
    sim_delay_ms: float = 0.0       # Simulated delay line (for testing without real channel)
    freq_slots: int = 5             # Number of frequency slots (-200, -100, 0, +100, +200)
    freq_step_hz: float = 100.0     # Frequency step between slots
    sim_delay_jitter_ms: float = 0.0  # Optional delay jitter to simulate spread

# Speed Mode Parameters
MODE_PARAMS = {
    ProtocolMode.TURBO:  {'pings': 1, 'interval': 0.05, 'slot_dur': 0.3, 'snr_min': 35.0},
    ProtocolMode.FAST:   {'pings': 1, 'interval': 0.10, 'slot_dur': 0.5, 'snr_min': 28.0},
    ProtocolMode.NORMAL: {'pings': 2, 'interval': 0.15, 'slot_dur': 0.8, 'snr_min': 20.0},
    ProtocolMode.DEEP:   {'pings': 4, 'interval': 0.20, 'slot_dur': 1.5, 'snr_min': 12.0},
    ProtocolMode.RESCUE: {'pings': 8, 'interval': 0.25, 'slot_dur': 3.0, 'snr_min': 5.0},
}

# ============================================================================
# Delay Line (Simulated WebSDR Latency)
# ============================================================================
class DelayLine:
    """Ring buffer delay line to simulate WebSDR/internet latency."""
    def __init__(self, max_delay_ms: float, sample_rate: int):
        self.fs = sample_rate
        self.max_samples = int(max_delay_ms * sample_rate / 1000.0) + 1
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_ptr = 0
        self.delay_samples = 0
        self.jitter_samples = 0
        
    def set_delay_ms(self, delay_ms: float):
        self.delay_samples = int(delay_ms * self.fs / 1000.0)
        self.delay_samples = min(self.delay_samples, self.max_samples - 1)
    
    def set_jitter_ms(self, jitter_ms: float):
        self.jitter_samples = int(abs(jitter_ms) * self.fs / 1000.0)
        
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Push samples in, get delayed samples out."""
        out = np.zeros(len(samples), dtype=np.float32)
        for i, s in enumerate(samples):
            self.buffer[self.write_ptr] = s
            # Apply simple per-sample jitter to simulate delay spread
            if self.jitter_samples > 0:
                jitter = random.randint(-self.jitter_samples, self.jitter_samples)
            else:
                jitter = 0
            eff_delay = self.delay_samples + jitter
            if eff_delay < 0:
                eff_delay = 0
            if eff_delay >= self.max_samples:
                eff_delay = self.max_samples - 1
            read_ptr = (self.write_ptr - eff_delay) % self.max_samples
            out[i] = self.buffer[read_ptr]
            self.write_ptr = (self.write_ptr + 1) % self.max_samples
        return out

class WaveletPartyProtocol:
    def __init__(self, config: StationConfig):
        self.config = config
        self.state = ProtocolState.IDLE
        self.mode = ProtocolMode.DEEP # Start safe
        self.forced_spin = None  # Optional override: 'center' | 'left' | 'right'
        self.current_snr_estimate = None  # Last auto-mode SNR estimate
        self.last_activity_ts = time.time()
        self.peer_linked = False
        self.tx_chat_queue = queue.Queue()
        self.rx_chat_queue = queue.Queue()
        self.last_ack_ts = 0.0
        self.last_ack_rx_ts = 0.0  # When we last detected an ACK from peer
        # Autonomous backoff/jitter to avoid discovery collisions
        self._silence_timeout_s = 2.0
        self._listen_jitter_s = random.uniform(0.3, 1.1)
        self._backoff_until = 0.0
        self._discovery_window_end = 0.0
        # Heartbeat scheduling for autonomous link validation
        self._next_heartbeat_ts = time.time() + random.uniform(5.0, 12.0)
        # No-pong streak for fallback to deeper modes
        self._no_pong_streak = 0
        self._fallback_deep_sweep = False  # Temporary DEEP sweep flag
        self._fallback_return_mode = None
        # Latency history for outlier rejection
        self._latency_history = []
        
        # DSP Components
        self.generator = WaveletPingGenerator(config.sample_rate)
        self.detector_fast = WaveletPingDetector(config.sample_rate)
        
        # Build single-ping templates for fast detection (center, left, right)
        self._fast_templates = {}
        self._fast_template_len = 0
        for name, offset in {'center': 0, 'left': -200, 'right': +200}.items():
            freq = config.base_freq + offset
            ping = self.generator.generate_ping(frequency=freq, wavelet_name='morl')
            ping = ping / np.max(np.abs(ping))
            self._fast_templates[name] = ping.astype(np.float32)
            self._fast_template_len = max(self._fast_template_len, len(ping))
        
        # ACK pattern template (3x 50ms 1000Hz tone with 50ms gaps)
        fs = config.sample_rate
        tone_dur = 0.05
        gap_dur = 0.05
        t = np.linspace(0, tone_dur, int(fs * tone_dur), endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        gap = np.zeros(int(fs * gap_dur), dtype=np.float32)
        self._ack_template = np.concatenate([tone, gap, tone, gap, tone])
        self._ack_template_len = len(self._ack_template)
        
        # Delay Line for simulation
        self.delay_line = DelayLine(1000.0, config.sample_rate)
        self.delay_line.set_delay_ms(config.sim_delay_ms)
        self.delay_line.set_jitter_ms(getattr(config, 'sim_delay_jitter_ms', 0.0))
        
        # Speed/SNR tracking for auto-throttle
        self.self_snr_history = []  # Track our own echo SNR
        self.target_mode = ProtocolMode.DEEP
        self.mode_lock_until = 0.0  # Prevent rapid mode switching

        # Auto-latency calibration support
        self.last_tx_template = None
        self.last_tx_time = 0.0
        self._last_latency_try_ts = 0.0
        self._last_latency_update_ts = 0.0
        
        # ALE-style handshake role tracking
        self._ale_role: str = "none"  # "initiator", "responder", or "none"
        self._ale_cq_sent: bool = False
        self._ale_de_sent: bool = False
        self._ale_exchange_complete: bool = False
        
        # Deep Mode Templates - Multi-Frequency Bank
        self.deep_repeats = MODE_PARAMS[ProtocolMode.DEEP]['pings']
        self.deep_interval = MODE_PARAMS[ProtocolMode.DEEP]['interval']
        
        # Generate templates for Center, Left, Right spins
        self.freq_offsets = {
            'center': 0,
            'left': -200,
            'right': +200
        }
        self.templates = {}
        self.detectors = {}
        self.ping_times = {}
        
        for name, offset in self.freq_offsets.items():
            freq = config.base_freq + offset
            seq, times, _ = self.generator.generate_ping_sequence(
                num_pings=self.deep_repeats,
                ping_interval=self.deep_interval,
                frequency=freq,
                padding=0.0
            )
            seq /= np.max(np.abs(seq))
            self.templates[name] = seq
            self.detectors[name] = WaveletPingDetector(config.sample_rate, template_signal=seq)
            self.ping_times[name] = times
        
        # Reference for spin analysis
        self.deep_times = self.ping_times['center']
        
        # Queues
        self.tx_queue = queue.Queue()
        self.rx_buffer = np.zeros(0)
        
        # Protocol Timing
        self.cycle_start_time = 0
        self.current_slot = 0
        self.last_detection_ts = 0
        self.slot_history = [] # Track slot timing for clock sync
        
        # Peers - Fine-grained tracking
        # {slot_id: {'snr': [], 'spin': [], 'freq_offset': [], 'last_seen': ts, 'confidence': float}}
        self.peers = {}
        
        # Detection History (for correlation across cycles)
        self.detection_log = []
        
        # Lighthouse Beacon Counter
        self._cycle_count = 0

    def process_audio(self, audio_chunk: np.ndarray):
        """
        Main processing loop called by audio callback.
        1. Apply simulated delay (if configured)
        2. Buffer Audio
        3. Detect Signals
        4. Update State Machine
        """
        # 1. Apply simulated delay
        if self.config.sim_delay_ms > 0:
            audio_chunk = self.delay_line.process(audio_chunk.flatten())
        else:
            audio_chunk = audio_chunk.flatten()
            
        # 2. Buffer
        self.rx_buffer = np.concatenate((self.rx_buffer, audio_chunk))
        # Keep buffer size manageable (2 cycles max? or just enough for detection)
        # For Deep Mode, we need ~1.5s. Let's keep 3.0s.
        keep_samples = int(self.config.sample_rate * 3.0)
        if len(self.rx_buffer) > keep_samples:
            self.rx_buffer = self.rx_buffer[-keep_samples:]
            
        # 2. Detect
        # We only run detection periodically to avoid CPU overload
        # Run every 200ms (5 Hz) - sufficient for slot timing
        now = time.time()
        if now - self.last_detection_ts < 0.2:
            # Still try latency calibration occasionally, independent of detection cadence
            if self.config.auto_latency and (now - self._last_latency_try_ts) > 0.5:
                self._try_auto_latency_calibration(now)
            return

        if self.mode == ProtocolMode.DEEP:
            self._detect_deep()
        else:
            self._detect_fast()
        # Attempt latency calibration after detection step as well
        if self.config.auto_latency and (now - self._last_latency_try_ts) > 0.5:
            self._try_auto_latency_calibration(now)
            
    def _detect_deep(self):
        """Multi-frequency bank detection for the HF jungle."""
        now = time.time()
        
        # Run all 3 detectors in parallel (Center, Left, Right)
        detections = []
        
        for spin_name, detector in self.detectors.items():
            try:
                peak, snr, _, peak_idx = detector.detect_presence(self.rx_buffer)
            except ValueError:
                peak, snr, _ = detector.detect_presence(self.rx_buffer)
                peak_idx = 0
            
            if snr > 12.0 and peak > 1e-3:
                detections.append({
                    'spin_name': spin_name,
                    'peak': peak,
                    'snr': snr,
                    'snr_db': 20 * np.log10(snr),
                    'peak_idx': peak_idx,
                    'freq': self.config.base_freq + self.freq_offsets[spin_name]
                })
        
        # Pick the strongest detection (highest SNR)
        if not detections:
            # No peer detected; keep last_activity_ts unchanged so LISTENING jitter/backoff applies
            return
            
        best = max(detections, key=lambda d: d['snr'])
        
        # Debounce
        if now - self.last_detection_ts < 0.8:  # Reduced to 0.8s for finer granularity
            return
        self.last_detection_ts = now
        
        # Fine-grained Spin Analysis (verify with I/Q even if template matched)
        spin_iq = self._analyze_spin(self.rx_buffer, best['peak_idx'])
        
        # Determine arrival slot
        # Adjust for remote latency if configured
        latency_sec = self.config.remote_latency_ms / 1000.0
        adjusted_time = now - latency_sec
        
        cycle_dur = self.config.slot_duration * self.config.slots_per_cycle
        time_in_cycle = (adjusted_time - self.cycle_start_time) % cycle_dur
        arrival_slot = int(time_in_cycle / self.config.slot_duration)
        
        # Self-Monitoring Check
        # If we are in our own TX slot (Slot 0 for Initiator), flag it
        is_self = (arrival_slot == 0) # Assuming we are Slot 0 for now
        
        if is_self:
            # Track self-echo SNR for speed throttling
            self.self_snr_history.append(best['snr_db'])
            if len(self.self_snr_history) > 20:
                self.self_snr_history.pop(0)
                
            if not self.config.monitor_self:
                # Ignore own echo if not monitoring (but still used for auto-throttle above)
                return

        # Log detection
        tag = "SELF" if is_self else "RX"
        logger.info(f"{tag} [{best['spin_name'].upper():^6}] Slot {arrival_slot} | "
                   f"SNR={best['snr_db']:.1f}dB @ {best['freq']:.0f}Hz | "
                   f"IQ-Spin={spin_iq}")
        
        # Update peer tracking (only if not self)
        if not is_self:
            self._update_peer(arrival_slot, best['snr_db'], spin_iq, best['freq'])
            # ALE-lite: if we hear peer consistently, mark activity
            self.last_activity_ts = now
            if self.state in [ProtocolState.IDLE, ProtocolState.LISTENING, ProtocolState.DISCOVERY_TX]:
                # Peer present -> switch to responder role and try to connect
                old = self.state
                self.state = ProtocolState.DISCOVERY_RX
                logger.info(f"STATE: {old.name} -> {self.state.name} reason=peer_detect slot={arrival_slot} snr={best['snr_db']:.1f}dB")

            # If we recently sent an ACK and now observe activity shortly after, consider link up
            if self.last_ack_ts and (now - self.last_ack_ts) < 2.0 and self.state == ProtocolState.DISCOVERY_RX:
                old = self.state
                self.state = ProtocolState.CONNECTED
                self.peer_linked = True
                self._ale_role = "responder"  # We sent ACK, so we're responder
                self._ale_de_sent = False
                self._ale_exchange_complete = False
                logger.info("ACK RECEIVED: transition to CONNECTED (role=responder)")
            # Reset no-pong streak upon peer activity
            if hasattr(self, '_no_pong_streak'):
                self._no_pong_streak = 0
        
        # Log to history
        self.detection_log.append({
            'timestamp': now,
            'adjusted_timestamp': adjusted_time,
            'slot': arrival_slot,
            'is_self': is_self,
            'snr_db': best['snr_db'],
            'spin': spin_iq,
            'freq': best['freq'],
            'template_match': best['spin_name']
        })
        
        # Keep log bounded (last 100 detections)
        if len(self.detection_log) > 100:
            self.detection_log.pop(0)
            
    def _detect_fast(self):
        """Fast single-ping matched filter detection for TURBO/FAST/NORMAL modes.
        Uses normalized cross-correlation against pre-built templates.
        Also checks for ACK pattern to confirm pong responses.
        """
        now = time.time()
        fs = self.config.sample_rate
        
        # Need enough buffer for at least one ping + margin
        min_buf = self._fast_template_len + int(0.1 * fs)
        if len(self.rx_buffer) < min_buf:
            return
        
        # Use recent portion of buffer (last ~0.8s for fast modes)
        analysis_dur = 0.8
        analysis_samples = int(analysis_dur * fs)
        if len(self.rx_buffer) > analysis_samples:
            buf = self.rx_buffer[-analysis_samples:]
        else:
            buf = self.rx_buffer
        
        # === 1. Detect single ping (center/left/right) ===
        best_ping = None
        best_snr_db = -np.inf
        
        for spin_name, tpl in self._fast_templates.items():
            # Normalized cross-correlation
            tpl_norm = tpl / (np.linalg.norm(tpl) + 1e-12)
            
            # Sliding window correlation
            if len(buf) < len(tpl):
                continue
            
            # FFT-based correlation for speed
            nfft = 1
            while nfft < len(buf) + len(tpl):
                nfft <<= 1
            BUF = np.fft.rfft(buf, nfft)
            TPL = np.fft.rfft(tpl_norm[::-1], nfft)
            corr = np.fft.irfft(BUF * TPL, nfft)
            corr = corr[:len(buf) - len(tpl) + 1]
            
            # Compute local RMS for normalization
            buf_sq = buf ** 2
            cumsum = np.cumsum(np.concatenate([[0], buf_sq]))
            window_energy = cumsum[len(tpl):] - cumsum[:-len(tpl)]
            local_rms = np.sqrt(window_energy / len(tpl) + 1e-12)
            
            # Normalized correlation
            norm_corr = corr / (local_rms * np.linalg.norm(tpl) + 1e-12)
            
            peak_idx = int(np.argmax(np.abs(norm_corr)))
            peak_val = np.abs(norm_corr[peak_idx])
            
            # Estimate SNR: peak vs median of correlation
            noise_floor = np.median(np.abs(norm_corr))
            if noise_floor > 1e-6:
                snr_linear = peak_val / noise_floor
                snr_db = 20 * np.log10(snr_linear + 1e-12)
            else:
                snr_db = 60.0 if peak_val > 0.1 else 0.0
            
            # Threshold: peak > 0.3 and SNR > mode minimum
            mode_snr_min = MODE_PARAMS[self.mode]['snr_min']
            if peak_val > 0.25 and snr_db > mode_snr_min - 5.0:
                if snr_db > best_snr_db:
                    best_snr_db = snr_db
                    best_ping = {
                        'spin_name': spin_name,
                        'peak_val': peak_val,
                        'snr_db': snr_db,
                        'peak_idx': peak_idx,
                        'freq': self.config.base_freq + {'center': 0, 'left': -200, 'right': +200}[spin_name]
                    }
        
        # === 2. Detect ACK pattern (pong) ===
        ack_detected = False
        ack_snr_db = 0.0
        if len(buf) >= self._ack_template_len:
            ack_tpl = self._ack_template / (np.linalg.norm(self._ack_template) + 1e-12)
            nfft = 1
            while nfft < len(buf) + len(ack_tpl):
                nfft <<= 1
            BUF = np.fft.rfft(buf, nfft)
            ACK = np.fft.rfft(ack_tpl[::-1], nfft)
            ack_corr = np.fft.irfft(BUF * ACK, nfft)
            ack_corr = ack_corr[:len(buf) - len(ack_tpl) + 1]
            
            ack_peak_idx = int(np.argmax(np.abs(ack_corr)))
            ack_peak_val = np.abs(ack_corr[ack_peak_idx])
            ack_noise = np.median(np.abs(ack_corr))
            
            if ack_noise > 1e-6:
                ack_snr_linear = ack_peak_val / ack_noise
                ack_snr_db = 20 * np.log10(ack_snr_linear + 1e-12)
            else:
                ack_snr_db = 60.0 if ack_peak_val > 0.1 else 0.0
            
            # ACK detection threshold
            if ack_peak_val > 0.3 and ack_snr_db > 15.0:
                ack_detected = True
        
        # Debounce detections
        if now - self.last_detection_ts < 0.3:
            return
        
        # === 3. Process detections ===
        detection_made = False
        
        if ack_detected and (now - self.last_ack_rx_ts) > 0.5:
            self.last_ack_rx_ts = now
            self.last_activity_ts = now
            self._no_pong_streak = 0  # Reset streak on pong
            detection_made = True
            logger.info(f"ACK_RX (PONG) detected | SNR={ack_snr_db:.1f}dB")
            
            # If we're waiting for pong, transition to CONNECTED
            if self.state in [ProtocolState.DISCOVERY_TX, ProtocolState.DISCOVERY_RX]:
                old = self.state
                self.state = ProtocolState.CONNECTED
                self.peer_linked = True
                self._ale_role = "initiator"  # We received ACK to our ping, so we're initiator
                self._ale_cq_sent = False
                self._ale_exchange_complete = False
                logger.info(f"STATE: {old.name} -> CONNECTED reason=ack_rx (role=initiator)")
        
        if best_ping is not None:
            self.last_detection_ts = now
            detection_made = True
            
            # Determine if self or peer
            latency_sec = self.config.remote_latency_ms / 1000.0
            adjusted_time = now - latency_sec
            cycle_dur = self.config.slot_duration * self.config.slots_per_cycle
            time_in_cycle = (adjusted_time - self.cycle_start_time) % cycle_dur
            arrival_slot = int(time_in_cycle / self.config.slot_duration)
            is_self = (arrival_slot == 0)
            
            if is_self:
                self.self_snr_history.append(best_ping['snr_db'])
                if len(self.self_snr_history) > 20:
                    self.self_snr_history.pop(0)
                if not self.config.monitor_self:
                    return
            
            tag = "SELF" if is_self else "PING_RX"
            logger.info(f"{tag} [{best_ping['spin_name'].upper():^6}] Slot {arrival_slot} | "
                       f"SNR={best_ping['snr_db']:.1f}dB @ {best_ping['freq']:.0f}Hz")
            
            if not is_self:
                self._update_peer(arrival_slot, best_ping['snr_db'], None, best_ping['freq'])
                self.last_activity_ts = now
                self._no_pong_streak = 0  # Reset on any peer activity
                
                # State transitions
                if self.state in [ProtocolState.IDLE, ProtocolState.LISTENING]:
                    old = self.state
                    self.state = ProtocolState.DISCOVERY_RX
                    logger.info(f"STATE: {old.name} -> DISCOVERY_RX reason=ping_rx snr={best_ping['snr_db']:.1f}dB")
                elif self.state == ProtocolState.DISCOVERY_TX:
                    # We heard a peer while transmitting -> they responded
                    old = self.state
                    self.state = ProtocolState.DISCOVERY_RX
                    logger.info(f"STATE: {old.name} -> DISCOVERY_RX reason=peer_response")
        
        # Update detection timestamp even if no detection (for debounce)
        if not detection_made:
            self.last_detection_ts = now

    def _analyze_spin(self, buffer, peak_idx):
        """Fine-grained spin analysis across all pings in sequence."""
        tpl_len = len(self.templates['center'])
        seq_start = peak_idx - tpl_len // 2
        
        chunks = []
        ping_len = int(0.2 * self.config.sample_rate)
        
        for pt in self.deep_times:
            p_start = seq_start + int(pt * self.config.sample_rate)
            p_end = p_start + ping_len
            if p_start >= 0 and p_end <= len(buffer):
                chunk = buffer[p_start:p_end]
                iq = iq_downconvert(chunk, self.config.base_freq, self.config.sample_rate)
                chunks.append(iq)
                
        if chunks:
            return detect_spin_robust(chunks, self.config.sample_rate)
        return None
    
    def _update_peer(self, slot_id, snr_db, spin, freq):
        """Track peer statistics for adaptive protocol."""
        if slot_id not in self.peers:
            self.peers[slot_id] = {
                'snr': [],
                'spin': [],
                'freq_offset': [],
                'last_seen': 0,
                'confidence': 0.0,
                'rx_count': 0
            }
        
        peer = self.peers[slot_id]
        peer['snr'].append(snr_db)
        peer['rx_count'] += 1
        peer['last_seen'] = time.time()
        
        # Keep rolling window of 10 measurements
        if len(peer['snr']) > 10:
            peer['snr'].pop(0)
        
        # Track frequency offset if spin is valid
        if spin and spin[0] in ['left', 'right', 'ambiguous']:
            offset = spin[1] if hasattr(spin[1], 'item') else float(spin[1])
            peer['freq_offset'].append(offset)
            peer['spin'].append(spin[0])
            
            if len(peer['freq_offset']) > 10:
                peer['freq_offset'].pop(0)
                peer['spin'].pop(0)
        
        # Calculate confidence (based on SNR consistency and rx count)
        if len(peer['snr']) >= 3:
            avg_snr = np.mean(peer['snr'])
            snr_std = np.std(peer['snr'])
            peer['confidence'] = min(1.0, (avg_snr - 15.0) / 30.0) * (1.0 - min(1.0, snr_std / 10.0))
        
        # Log peer summary periodically
        if peer['rx_count'] % 5 == 0:
            avg_snr = np.mean(peer['snr']) if peer['snr'] else 0
            avg_offset = np.mean(peer['freq_offset']) if peer['freq_offset'] else 0
            logger.info(f"PEER[Slot {slot_id}]: RX×{peer['rx_count']} | "
                       f"SNR={avg_snr:.1f}±{np.std(peer['snr']):.1f}dB | "
                       f"Δf={avg_offset:.1f}Hz | Conf={peer['confidence']:.2f}")

    def tick(self):
        """
        Called periodically to handle timing and TX
        """
        now = time.time()
        
        # Automation
        if self.config.auto_mode:
            self._update_auto_mode()
        
        # Slot Management
        cycle_dur = self.config.slot_duration * self.config.slots_per_cycle
        time_in_cycle = (now - self.cycle_start_time) % cycle_dur
        new_slot = int(time_in_cycle / self.config.slot_duration)
        
        if new_slot != self.current_slot:
            self.current_slot = new_slot
            self._on_slot_start(new_slot)

        # ALE-lite state machine with fallback sweep
        if self.state == ProtocolState.IDLE:
            old = self.state
            self.state = ProtocolState.LISTENING
            logger.info(f"STATE: {old.name} -> {self.state.name} reason=startup")
            
        elif self.state == ProtocolState.LISTENING:
            # Randomized jitter on silence before attempting discovery
            if now - self.last_activity_ts > (self._silence_timeout_s + self._listen_jitter_s):
                if now < self._backoff_until:
                    pass  # Stay in listening during backoff window
                else:
                    old = self.state
                    self.state = ProtocolState.DISCOVERY_TX
                    self._discovery_window_end = now + 1.0
                    logger.info(f"STATE: {old.name} -> {self.state.name} reason=silence>{self._silence_timeout_s:.1f}s+jitter")
                    
        elif self.state == ProtocolState.DISCOVERY_TX:
            # Keep TX within a short window, then backoff to avoid collisions
            if now > self._discovery_window_end:
                backoff = random.uniform(0.8, 2.2)
                self._backoff_until = now + backoff
                old = self.state
                self.state = ProtocolState.LISTENING
                
                # Track no ping-pong streak
                self._no_pong_streak = min(self._no_pong_streak + 1, 15)
                
                # === Fallback deep sweep logic ===
                if self._no_pong_streak >= 3 and not self._fallback_deep_sweep:
                    if self.mode in [ProtocolMode.TURBO, ProtocolMode.FAST, ProtocolMode.NORMAL]:
                        # Initiate temporary DEEP sweep
                        self._fallback_deep_sweep = True
                        self._fallback_return_mode = self.mode
                        self.mode = ProtocolMode.DEEP
                        sdur = MODE_PARAMS[self.mode]['slot_dur']
                        self.config.slot_duration = sdur
                        self.cycle_start_time = time.time()
                        logger.info(f"FALLBACK: No pong x{self._no_pong_streak}, temporary DEEP sweep (will return to {self._fallback_return_mode.name})")
                
                # Progressive mode downgrade on persistent silence
                if self.config.auto_mode and not self._fallback_deep_sweep:
                    if self._no_pong_streak >= 4 and self.mode in [ProtocolMode.TURBO, ProtocolMode.FAST]:
                        self.mode = ProtocolMode.NORMAL
                        self.mode_lock_until = now + 3.0
                        sdur = MODE_PARAMS[self.mode]['slot_dur']
                        self.config.slot_duration = sdur
                        self.cycle_start_time = time.time()
                        logger.info("Auto-Mode: No ping-pong x4, downgrade to NORMAL")
                    elif self._no_pong_streak >= 6 and self.mode == ProtocolMode.NORMAL:
                        self.mode = ProtocolMode.DEEP
                        self.mode_lock_until = now + 5.0
                        sdur = MODE_PARAMS[self.mode]['slot_dur']
                        self.config.slot_duration = sdur
                        self.cycle_start_time = time.time()
                        logger.info("Auto-Mode: Persistent silence x6, downgrade to DEEP")
                    elif self._no_pong_streak >= 8 and self.mode == ProtocolMode.DEEP:
                        self.mode = ProtocolMode.RESCUE
                        self.mode_lock_until = now + 8.0
                        sdur = MODE_PARAMS[self.mode]['slot_dur']
                        self.config.slot_duration = sdur
                        self.cycle_start_time = time.time()
                        logger.info("Auto-Mode: Extreme silence x8, downgrade to RESCUE")
                
                logger.info(f"STATE: {old.name} -> {self.state.name} reason=tx_window_end backoff={backoff:.2f}s streak={self._no_pong_streak}")
                
        elif self.state == ProtocolState.DISCOVERY_RX:
            # If we hear activity recently, try to connect
            if now - self.last_activity_ts < 1.5:
                old = self.state
                self.state = ProtocolState.CONNECTED
                self.peer_linked = True
                self._no_pong_streak = 0
                self._ale_role = "responder"  # We heard peer and responded, so we're responder
                self._ale_de_sent = False
                self._ale_exchange_complete = False
                # End fallback sweep if active
                if self._fallback_deep_sweep and self._fallback_return_mode:
                    self.mode = self._fallback_return_mode
                    sdur = MODE_PARAMS[self.mode]['slot_dur']
                    self.config.slot_duration = sdur
                    self.cycle_start_time = time.time()
                    logger.info(f"FALLBACK END: Connected, returning to {self.mode.name}")
                    self._fallback_deep_sweep = False
                    self._fallback_return_mode = None
                logger.info(f"STATE: {old.name} -> CONNECTED reason=activity<1.5s (role=responder)")
            elif now - self.last_activity_ts > 5.0:
                # Timeout waiting for confirmation
                old = self.state
                self.state = ProtocolState.LISTENING
                logger.info(f"STATE: {old.name} -> LISTENING reason=rx_timeout>5s")
                
        elif self.state == ProtocolState.CONNECTED:
            # Maintain link; downgrade if no activity
            if now - self.last_activity_ts > 30.0:
                old = self.state
                self.state = ProtocolState.LISTENING
                self.peer_linked = False
                # Reset ALE state on disconnect
                self._ale_role = "none"
                self._ale_cq_sent = False
                self._ale_de_sent = False
                self._ale_exchange_complete = False
                logger.info(f"STATE: {old.name} -> {self.state.name} reason=idle>30s")
            else:
                # ALE-style CQ/DE exchange (once per connection)
                self._do_ale_exchange()
                
                # Autonomous timestamp broadcast - party protocol: all4one, one4all
                if now >= self._next_heartbeat_ts:
                    self._broadcast_party_timestamp()
                    self._next_heartbeat_ts = now + random.uniform(3.0, 8.0)  # More frequent for party

    def _update_auto_mode(self):
        """
        Automatically throttle speed based on SNR.
        Uses self-echo SNR (if available) and peer SNR for decisions.
        """
        now = time.time()
        
        # Respect mode lock (prevent oscillation)
        if now < self.mode_lock_until:
            return
            
        # Determine best available SNR estimate
        snr_estimate = None
        
        # Prefer self-echo SNR (most reliable for our TX power)
        if self.self_snr_history:
            snr_estimate = np.mean(self.self_snr_history[-5:])  # Last 5 measurements
        elif self.peers:
            # Fall back to peer SNR average
            all_snrs = []
            for p in self.peers.values():
                all_snrs.extend(p['snr'])
            if all_snrs:
                snr_estimate = np.mean(all_snrs)
        
        self.current_snr_estimate = snr_estimate

        if snr_estimate is None:
            # No data -> stay in DEEP (search mode)
            if self.mode != ProtocolMode.DEEP:
                self.mode = ProtocolMode.DEEP
                self.mode_lock_until = now + 5.0
                logger.info("Auto-Mode: No SNR data, switching to DEEP (search)")
                # Apply timing for new mode
                sdur = MODE_PARAMS[self.mode]['slot_dur']
                if abs(self.config.slot_duration - sdur) > 1e-6:
                    self.config.slot_duration = sdur
                    self.cycle_start_time = time.time()
            return
        
        # Find optimal mode based on SNR
        # We want the FASTEST mode where snr_estimate > snr_min
        target = ProtocolMode.RESCUE  # Worst case
        for mode in [ProtocolMode.TURBO, ProtocolMode.FAST, ProtocolMode.NORMAL, ProtocolMode.DEEP, ProtocolMode.RESCUE]:
            if snr_estimate >= MODE_PARAMS[mode]['snr_min']:
                target = mode
                break
        
        # Apply hysteresis (only switch if target is different and stable)
        if target != self.mode:
            # Switching UP (faster) requires higher confidence
            if target.value < self.mode.value:  # Lower enum = faster
                # Need SNR to be well above threshold
                if snr_estimate >= MODE_PARAMS[target]['snr_min'] + 5.0:
                    self.mode = target
                    self.mode_lock_until = now + 3.0
                    logger.info(f"Auto-Mode: SNR={snr_estimate:.1f}dB -> Upgrade to {target.name}")
                    # Apply timing for new mode
                    sdur = MODE_PARAMS[self.mode]['slot_dur']
                    if abs(self.config.slot_duration - sdur) > 1e-6:
                        self.config.slot_duration = sdur
                        self.cycle_start_time = time.time()
            else:
                # Switching DOWN (slower) - be more aggressive
                self.mode = target
                self.mode_lock_until = now + 2.0
                logger.info(f"Auto-Mode: SNR={snr_estimate:.1f}dB -> Downgrade to {target.name}")
                # Apply timing for new mode
                sdur = MODE_PARAMS[self.mode]['slot_dur']
                if abs(self.config.slot_duration - sdur) > 1e-6:
                    self.config.slot_duration = sdur
                    self.cycle_start_time = time.time()
            
    def _on_slot_start(self, slot_idx):
        """Slot transition handler with adaptive TX."""
        self.slot_history.append({'slot': slot_idx, 'time': time.time()})
        if len(self.slot_history) > 20:
            self.slot_history.pop(0)
        
        # Initiator Logic - TX on slot 0
        if slot_idx == 0:
            self._cycle_count += 1
            if self.state in [ProtocolState.DISCOVERY_TX, ProtocolState.IDLE, ProtocolState.LISTENING]:
                # Use gapped pings for DEEP/RESCUE to allow mid-train pong
                if self.mode in [ProtocolMode.DEEP, ProtocolMode.RESCUE]:
                    self._send_ping_with_gap()
                else:
                    self._send_cq()
            elif self.state == ProtocolState.DISCOVERY_RX:
                # Send explicit ACK pattern (pong) to complete handshake
                self._send_ack()
            elif self.state == ProtocolState.CONNECTED:
                # Send periodic ping to maintain link + any chat
                self._send_cq()
                self._send_chat_burst_if_any()
        
        # Responder Logic - TX on slot 1 (reply slot)
        elif slot_idx == 1:
            if self.state == ProtocolState.DISCOVERY_RX:
                # Send ACK (pong) in response slot
                self._send_ack()
        
        # Cleanup stale peers (not seen in 30s)
        self._cleanup_peers()
            
    def _send_cq(self):
        """
        Adaptive CQ transmission based on current mode and peer state.
        Generates appropriate ping sequence for current speed mode.
        """
        # === Lighthouse Beacon Logic ===
        # If in fast mode, periodically send a DEEP sequence to allow weak stations to detect us.
        # Every 10th cycle (approx every 3-5 seconds in Turbo mode)
        is_lighthouse = False
        if self.mode in [ProtocolMode.TURBO, ProtocolMode.FAST] and (self._cycle_count % 10 == 0):
            params = MODE_PARAMS[ProtocolMode.DEEP]
            is_lighthouse = True
            logger.info("LIGHTHOUSE BEACON: Forcing DEEP sequence for discovery")
        else:
            params = MODE_PARAMS[self.mode]
        
        # Choose frequency slot based on peer feedback or cycle through
        # For echo test: use center. For diversity: rotate.
        spin_choice = 'center'
        if self.forced_spin in ['center', 'left', 'right']:
            spin_choice = self.forced_spin
        else:
            # Only apply peer-based adaptation if not forced
            if self.peers:
                offsets = []
                for p in self.peers.values():
                    offsets.extend(p.get('freq_offset', []))
                if offsets:
                    avg_offset = np.mean(offsets)
                    if avg_offset < -50:
                        spin_choice = 'left'
                    elif avg_offset > 50:
                        spin_choice = 'right'
        if self.peers:
            # If we're hearing peers on a specific offset, match it
            offsets = []
            for p in self.peers.values():
                offsets.extend(p.get('freq_offset', []))
            if offsets:
                avg_offset = np.mean(offsets)
                if avg_offset < -50:
                    spin_choice = 'left'
                elif avg_offset > 50:
                    spin_choice = 'right'
        
        # Generate signal based on mode
        freq = self.config.base_freq + self.freq_offsets[spin_choice]
        
        if params['pings'] == 1:
            # Single ping (TURBO/FAST)
            sig = self.generator.generate_ping(frequency=freq, wavelet_name='morl')
        else:
            # Sequence (NORMAL/DEEP/RESCUE)
            sig, _, _ = self.generator.generate_ping_sequence(
                num_pings=params['pings'],
                ping_interval=params['interval'],
                frequency=freq,
                padding=0.0
            )
        
        sig = (sig / np.max(np.abs(sig)) * 0.8).astype(np.float32)
        
        # Calculate data rate estimate
        seq_duration = len(sig) / self.config.sample_rate
        effective_rate = 1.0 / (self.config.slot_duration * self.config.slots_per_cycle)  # cycles/sec
        
        snr_txt = f"SNR={self.current_snr_estimate:.1f}dB" if self.current_snr_estimate is not None else "SNR=?"
        logger.info(f"TX_CQ {self.mode.name} [{spin_choice.upper()}] @ {freq:.0f}Hz | "
                   f"Pings={params['pings']} | Dur={seq_duration:.2f}s | Peers={len(self.peers)} | {snr_txt}")
        
        self.tx_queue.put(sig)
        # Remember template and time for latency calibration
        self.last_tx_template = sig.copy()
        self.last_tx_time = time.time()

    def _do_ale_exchange(self):
        """ALE-style CQ/DE exchange after handshake completes.
        
        Initiator sends: CQ CQ CQ DE <callsign>
        Responder replies: DE <callsign> <callsign>
        
        This mimics amateur radio ALE calling protocol.
        """
        if self._ale_exchange_complete:
            return
        
        if self._ale_role == "initiator" and not self._ale_cq_sent:
            # Initiator sends CQ call
            cq_msg = f"CQ CQ CQ DE {self.config.callsign}"
            self.tx_chat_queue.put(cq_msg)
            self._ale_cq_sent = True
            logger.info(f"ALE: CQ transmitted ({self.config.callsign} initiator)")
        
        elif self._ale_role == "responder" and not self._ale_de_sent:
            # Responder sends DE response
            de_msg = f"DE {self.config.callsign} {self.config.callsign}"
            self.tx_chat_queue.put(de_msg)
            self._ale_de_sent = True
            self._ale_exchange_complete = True  # Responder is done after DE
            logger.info(f"ALE: DE response transmitted ({self.config.callsign} responder)")
        
        elif self._ale_role == "initiator" and self._ale_cq_sent:
            # Initiator considers exchange complete after CQ (responder's DE will come)
            self._ale_exchange_complete = True
            logger.info(f"ALE: Exchange complete ({self.config.callsign} initiator)")

    def _broadcast_party_timestamp(self):
        """Broadcast timestamp to all peers - party protocol style.
        
        Format: [callsign] @HH:MM:SS.mmm UTC -> ALL
        
        This creates a mesh of synchronized time awareness.
        All stations hear all timestamps - true party protocol: all4one, one4all.
        """
        from datetime import datetime
        
        # Generate UTC timestamp with milliseconds
        utc_now = datetime.utcnow()
        ts_str = utc_now.strftime("%H:%M:%S.") + f"{utc_now.microsecond // 1000:03d}"
        
        # Party broadcast message
        party_msg = f"[{self.config.callsign}] @{ts_str} UTC -> ALL"
        
        try:
            self.tx_chat_queue.put_nowait(party_msg)
            logger.info(f"PARTY TX: {party_msg}")
            self._send_chat_burst_if_any()
        except Exception:
            pass

    def _send_chat_burst_if_any(self):
        """Placeholder: when connected, send a short audible burst to mark chat data.
        Actual text modulation TBD. This ensures TX activity for link maintenance.
        """
        try:
            text = self.tx_chat_queue.get_nowait()
        except queue.Empty:
            return
        # Simple 1200 Hz tone burst of 200ms to indicate chat send
        duration = 0.2
        t = np.linspace(0, duration, int(self.config.sample_rate * duration), endpoint=False)
        burst = (0.5 * np.sin(2*np.pi*1200*t)).astype(np.float32)
        self.tx_queue.put(burst)
        logger.info(f"CHAT TX: {text}")
        # Locally echo to RX queue for demo if no demod yet
        self.rx_chat_queue.put(f"{self.config.callsign}: {text}")

    def _send_ack(self):
        """Send an explicit short ACK pattern (pong) to signal responder presence."""
        fs = self.config.sample_rate
        # Pattern: 3x 50ms 1000Hz tone with 50ms gaps
        tone_dur = 0.05
        gap_dur = 0.05
        t = np.linspace(0, tone_dur, int(fs * tone_dur), endpoint=False)
        tone = (0.5 * np.sin(2*np.pi*1000*t)).astype(np.float32)
        gap = np.zeros(int(fs * gap_dur), dtype=np.float32)
        ack = np.concatenate([tone, gap, tone, gap, tone])
        self.tx_queue.put(ack)
        self.last_ack_ts = time.time()
        logger.info("ACK_TX (PONG): 3x1000Hz 50ms")

    def _send_ping_with_gap(self):
        """Send ping train with listening gaps for deep SNR scenarios.
        Structure: ping-gap-ping-gap-... allows responder to inject pong mid-train.
        """
        params = MODE_PARAMS[self.mode]
        num_pings = params['pings']
        
        # Choose frequency
        spin_choice = 'center'
        if self.forced_spin in ['center', 'left', 'right']:
            spin_choice = self.forced_spin
        freq = self.config.base_freq + self.freq_offsets[spin_choice]
        
        fs = self.config.sample_rate
        ping = self.generator.generate_ping(frequency=freq, wavelet_name='morl')
        ping = (ping / np.max(np.abs(ping)) * 0.8).astype(np.float32)
        
        # Gap duration: enough for ACK (~250ms) + margin
        gap_dur = 0.35  # 350ms gap
        gap_samples = int(gap_dur * fs)
        gap = np.zeros(gap_samples, dtype=np.float32)
        
        # Build sequence with gaps: P-G-P-G-P-G-P (for 4 pings)
        segments = []
        for i in range(num_pings):
            segments.append(ping)
            if i < num_pings - 1:  # No gap after last ping
                segments.append(gap)
        
        sig = np.concatenate(segments)
        
        snr_txt = f"SNR={self.current_snr_estimate:.1f}dB" if self.current_snr_estimate is not None else "SNR=?"
        logger.info(f"TX_CQ_GAP {self.mode.name} [{spin_choice.upper()}] @ {freq:.0f}Hz | "
                   f"Pings={num_pings} with gaps | {snr_txt}")
        
        self.tx_queue.put(sig)
        self.last_tx_template = ping.copy()  # Store single ping for latency cal
        self.last_tx_time = time.time()

    def _try_auto_latency_calibration(self, now_ts: float):
        """Estimate effective remote latency by correlating the last TX template against RX buffer.
        Includes all pipeline delays; smooths updates with outlier rejection.
        """
        if self.last_tx_template is None or self.last_tx_time == 0.0:
            return
        # Guard: only attempt within a reasonable window after TX (e.g., 2s)
        if (now_ts - self.last_tx_time) > 2.5:
            self._last_latency_try_ts = now_ts
            return
        rx = self.rx_buffer
        tpl = self.last_tx_template
        if len(rx) < len(tpl) + 100:
            self._last_latency_try_ts = now_ts
            return
        try:
            # Cross-correlation via FFT for speed
            N = len(rx)
            M = len(tpl)
            corr = np.correlate(rx, tpl, mode='valid') if (N * M) < 2000000 else None
            if corr is None:
                # FFT method for large arrays
                nfft = 1
                while nfft < N + M:
                    nfft <<= 1
                RX = np.fft.rfft(rx, nfft)
                TP = np.fft.rfft(tpl[::-1], nfft)
                c = np.fft.irfft(RX * TP, nfft)
                start = M - 1
                end = start + (N - M + 1)
                corr = c[start:end]
            i_max = int(np.argmax(corr))
            fs = self.config.sample_rate
            tpl_start_time = now_ts - (N - (i_max + M)) / fs
            measured_delay = tpl_start_time - self.last_tx_time
            
            # Sanity bounds: accept 10ms..2000ms
            if measured_delay < 0.010 or measured_delay > 2.0:
                self._last_latency_try_ts = now_ts
                return
            
            measured_ms = measured_delay * 1000.0
            old_latency_ms = self.config.remote_latency_ms
            
            # === Outlier rejection ===
            # Reject if jump is > 1.5x previous or > 200ms delta (unless first measurement)
            if old_latency_ms > 10.0:
                ratio = measured_ms / old_latency_ms if old_latency_ms > 0 else 999
                delta = abs(measured_ms - old_latency_ms)
                if ratio > 1.5 or ratio < 0.67 or delta > 200.0:
                    # Outlier - ignore this measurement
                    self._last_latency_try_ts = now_ts
                    return
            
            # Add to history for median filtering
            self._latency_history.append(measured_ms)
            if len(self._latency_history) > 5:
                self._latency_history.pop(0)
            
            # Use median of recent history for stability
            if len(self._latency_history) >= 3:
                median_latency = float(np.median(self._latency_history))
            else:
                median_latency = measured_ms
            
            # EMA smoothing on top of median
            alpha = 0.3
            if old_latency_ms > 0:
                new_ms = (1 - alpha) * old_latency_ms + alpha * median_latency
            else:
                new_ms = median_latency
            
            # Only update if change is significant (>5ms)
            if abs(new_ms - self.config.remote_latency_ms) > 5.0:
                self.config.remote_latency_ms = new_ms
                self._last_latency_update_ts = now_ts
                logger.info(f"Auto-Latency: {new_ms:.0f}ms (measured {measured_ms:.0f}ms, median {median_latency:.0f}ms)")
                if hasattr(self, 'agg_log_writer'):
                    try:
                        self.agg_log_writer({'event': 'latency_update', 'lat_ms': float(new_ms)})
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self._last_latency_try_ts = now_ts
    
    def _cleanup_peers(self):
        """Remove stale peer entries."""
        now = time.time()
        stale = [sid for sid, p in self.peers.items() if now - p['last_seen'] > 30.0]
        for sid in stale:
            logger.info(f"PEER[Slot {sid}]: Timeout (last seen {now - self.peers[sid]['last_seen']:.0f}s ago)")
            del self.peers[sid]
    
    def get_peer_summary(self):
        """Get current peer statistics for UI/logging."""
        summary = []
        for slot_id, peer in self.peers.items():
            avg_snr = np.mean(peer['snr']) if peer['snr'] else 0
            avg_offset = np.mean(peer['freq_offset']) if peer['freq_offset'] else 0
            summary.append({
                'slot': slot_id,
                'rx_count': peer['rx_count'],
                'avg_snr': avg_snr,
                'avg_freq_offset': avg_offset,
                'confidence': peer['confidence']
            })
        return summary

    def get_current_rms(self):
        """Get RMS amplitude of the latest audio buffer for VU meter."""
        if len(self.rx_buffer) == 0:
            return 0.0
        # Take last 100ms
        chunk_len = int(self.config.sample_rate * 0.1)
        chunk = self.rx_buffer[-chunk_len:]
        return np.sqrt(np.mean(chunk**2))

