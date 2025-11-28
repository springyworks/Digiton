"""
DIGITON AUTO-SPEED PROTOCOL
Automatically adapts transmission speed based on measured channel SNR.

Speed Modes:
- TURBO: 5ms pulses (200 baud) @ +10dB SNR - Single pulse detection
- FAST: 20ms pulses (50 baud) @ 0dB SNR - Single pulse detection  
- NORMAL: 50ms pulses (20 baud) @ -10dB SNR - 4x coherent integration
- SLOW: 80ms pulses (12 baud) @ -30dB SNR - 64x coherent integration
- DEEP: 80ms pulses @ -50 to -60dB SNR - 256-1024x coherent integration

The protocol:
1. Master sends CQ at NORMAL speed (always detectable)
2. Station responds at NORMAL speed
3. Both measure SNR of received ping
4. Master sends SPEED_GRANT with recommended mode
5. Both switch to new speed for data transfer
6. If errors occur, fallback to slower mode automatically
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
from hf_channel_simulator import HFChannelSimulator
import digiton.wavelets
import digiton.sdr

os.makedirs('data', exist_ok=True)

SAMPLE_RATE = 8000
CENTER_FREQ = 1500
SPIN_OFFSET = 200

# Speed Mode Definitions
SPEED_MODES = {
    'TURBO': {
        'sigma': 0.001,      # 1ms pulse width
        'interval': 0.005,   # 5ms per bit
        'integration': 1,     # No integration needed
        'min_snr': 10,       # +10dB required
        'baud': 200
    },
    'FAST': {
        'sigma': 0.004,      # 4ms pulse width  
        'interval': 0.020,   # 20ms per bit
        'integration': 1,     # No integration
        'min_snr': 0,        # 0dB required
        'baud': 50
    },
    'NORMAL': {
        'sigma': 0.010,      # 10ms pulse width
        'interval': 0.050,   # 50ms per bit
        'integration': 4,     # 4x for margin
        'min_snr': -10,      # -10dB required
        'baud': 20
    },
    'SLOW': {
        'sigma': 0.015,      # 15ms pulse width
        'interval': 0.080,   # 80ms per bit
        'integration': 64,   # 64x stacking
        'min_snr': -30,      # -30dB required
        'baud': 12
    },
    'DEEP': {
        'sigma': 0.015,      # 15ms pulse width
        'interval': 0.100,   # 100ms per pulse
        'integration': 512,  # 512x stacking for -50 to -60dB
        'min_snr': -50,      # -50 to -60dB
        'baud': 1  # Effective baud after integration
    }
}

class AutoSpeedModem:
    def __init__(self, fs=SAMPLE_RATE):
        self.fs = fs
        self.current_mode = 'NORMAL'  # Start at NORMAL
        self.measured_snr = None
        
    def generate_pulse(self, spin='right', mode='NORMAL'):
        """Generate pulse for given speed mode"""
        params = SPEED_MODES[mode]
        return digiton.wavelets.morlet_pulse(
            fc_hz=CENTER_FREQ,
            spin_offset_hz=SPIN_OFFSET,
            sigma=params['sigma'],
            fs=self.fs,
            spin=spin,
            trunc_sigmas=4.0
        )
    
    def generate_train(self, spin='right', mode='NORMAL'):
        """Generate coherent pulse train for detection"""
        params = SPEED_MODES[mode]
        repeats = params['integration']
        interval = params['interval']
        
        pulse = self.generate_pulse(spin, mode)
        pulse_len = len(pulse)
        
        interval_samples = int(interval * self.fs)
        total_samples = interval_samples * repeats
        
        train = np.zeros(total_samples)
        for i in range(repeats):
            start = i * interval_samples
            end = start + pulse_len
            if end <= total_samples:
                train[start:end] += pulse
                
        return train
    
    def sdr_downconvert(self, real_signal):
        """Downconvert to baseband I/Q"""
        return digiton.sdr.iq_downconvert(real_signal, CENTER_FREQ, self.fs, lpf_hz=500)
    
    def coherent_integrate(self, rx_signal, mode='NORMAL'):
        """Perform coherent integration for detection"""
        params = SPEED_MODES[mode]
        repeats = params['integration']
        interval = params['interval']
        
        if repeats == 1:
            return rx_signal[:int(interval * self.fs)]
            
        # Use library function
        # Note: digiton.sdr.coherent_integrate expects concatenated pulses
        # We need to ensure we pass exactly repeats * interval_samples
        interval_samples = int(interval * self.fs)
        total_len = interval_samples * repeats
        
        if len(rx_signal) < total_len:
            # Pad if short
            padded = np.zeros(total_len)
            padded[:len(rx_signal)] = rx_signal
            signal_to_integrate = padded
        else:
            signal_to_integrate = rx_signal[:total_len]
            
        integrated = digiton.sdr.coherent_integrate(signal_to_integrate, repeats)
        return integrated / repeats

    
    def measure_snr(self, rx_signal, mode='NORMAL'):
        """
        Measure SNR of received pulse AFTER integration.
        Returns estimated SNR in dB.
        """
        # Integrate
        integrated = self.coherent_integrate(rx_signal, mode)
        
        # Downconvert
        iq = self.sdr_downconvert(integrated)
        
        # Find signal region (center of interval)
        center = len(iq) // 2
        pulse_width = int(SPEED_MODES[mode]['sigma'] * 8 * self.fs)
        
        # Signal power (center region where pulse is)
        signal_start = max(0, center - pulse_width)
        signal_end = min(len(iq), center + pulse_width)
        signal_region = iq[signal_start:signal_end]
        signal_power = np.mean(np.abs(signal_region)**2)
        
        # Noise power (edges - before and after pulse)
        noise_width = pulse_width * 2
        noise_before = iq[:min(noise_width, signal_start)]
        noise_after = iq[max(signal_end, len(iq) - noise_width):]
        
        if len(noise_before) > 0 and len(noise_after) > 0:
            noise_region = np.concatenate([noise_before, noise_after])
        elif len(noise_before) > 0:
            noise_region = noise_before
        elif len(noise_after) > 0:
            noise_region = noise_after
        else:
            noise_region = iq  # Fallback
            
        noise_power = np.mean(np.abs(noise_region)**2)
        
        if noise_power == 0:
            return 100  # Perfect signal
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -100
        
        # Adjust for integration gain (we already applied it, so subtract it for "raw" SNR)
        integration_gain_db = 10 * np.log10(SPEED_MODES[mode]['integration'])
        raw_snr_db = snr_db - integration_gain_db
        
        return raw_snr_db
    
    def select_speed_mode(self, measured_snr_db):
        """
        Select optimal speed mode based on measured SNR.
        Returns mode name and why.
        """
        # Add 6dB margin for safety
        safe_snr = measured_snr_db - 6
        
        # Find best mode
        if safe_snr >= SPEED_MODES['TURBO']['min_snr']:
            return 'TURBO', f"Excellent SNR ({measured_snr_db:.1f}dB) - TURBO mode"
        elif safe_snr >= SPEED_MODES['FAST']['min_snr']:
            return 'FAST', f"Good SNR ({measured_snr_db:.1f}dB) - FAST mode"
        elif safe_snr >= SPEED_MODES['NORMAL']['min_snr']:
            return 'NORMAL', f"Fair SNR ({measured_snr_db:.1f}dB) - NORMAL mode"
        elif safe_snr >= SPEED_MODES['SLOW']['min_snr']:
            return 'SLOW', f"Weak SNR ({measured_snr_db:.1f}dB) - SLOW mode"
        else:
            return 'DEEP', f"Very weak SNR ({measured_snr_db:.1f}dB) - DEEP mode"
    
    def encode_bits(self, bits, mode='NORMAL'):
        """Encode bit sequence as spin pulses"""
        params = SPEED_MODES[mode]
        interval = params['interval']
        repeats = params['integration']
        
        total_samples = int(len(bits) * interval * repeats * self.fs)
        encoded = np.zeros(total_samples)
        
        for i, bit in enumerate(bits):
            spin = 'right' if bit == 1 else 'left'
            train = self.generate_train(spin, mode)
            
            start = int(i * interval * repeats * self.fs)
            end = start + len(train)
            if end <= total_samples:
                encoded[start:end] = train
        
        return encoded

def simulate_auto_speed_protocol():
    print("=" * 70)
    print("DIGITON AUTOMATIC SPEED ADAPTATION PROTOCOL")
    print("=" * 70)
    
    modem = AutoSpeedModem()
    
    # Test Scenario: Watterson Channel at -50dB (Deep Mode needed)
    snr_db = -50
    print(f"\nTest Scenario: Watterson Fading Channel @ {snr_db}dB SNR")
    print(f"Expected Mode: DEEP (256x coherent integration)")
    print("-" * 70)
    
    # === PHASE 1: DISCOVERY (DEEP MODE for very weak signals) ===
    # In a real system, we'd try NORMAL first, then fall back to DEEP if no response
    # Here we start with DEEP since we know the channel is -50dB
    discovery_mode = 'DEEP'
    print(f"\n[PHASE 1] DISCOVERY - Using {discovery_mode} mode for initial contact")
    print(f"  (In real system: try NORMAL first, escalate to DEEP if needed)")
    
    # Master sends CQ at DEEP
    cq_train = modem.generate_train('right', discovery_mode)
    print(f"  Master TX: CQ (Right Spin, {discovery_mode} mode, {SPEED_MODES[discovery_mode]['integration']}x integration)")
    
    # Apply channel
    simulator = HFChannelSimulator(sample_rate=SAMPLE_RATE, snr_db=snr_db, 
                                   doppler_spread_hz=0.5, multipath_delay_ms=2.0)
    rx_cq = simulator.simulate_hf_channel(cq_train, include_fading=True, freq_offset=True)
    rx_cq_audio = np.real(rx_cq)
    
    # Station measures SNR
    measured_snr = modem.measure_snr(rx_cq_audio, discovery_mode)
    print(f"  Station RX: Measured SNR = {measured_snr:.1f}dB (raw, before integration)")
    
    # === PHASE 2: SPEED NEGOTIATION ===
    print("\n[PHASE 2] SPEED NEGOTIATION")
    
    # Station selects mode based on measurement
    new_mode, reason = modem.select_speed_mode(measured_snr)
    print(f"  Station Decision: {reason}")
    print(f"  Switching to: {new_mode} mode")
    print(f"    - Pulse width: {SPEED_MODES[new_mode]['sigma']*1000:.1f}ms")
    print(f"    - Integration: {SPEED_MODES[new_mode]['integration']}x")
    print(f"    - Data rate: ~{SPEED_MODES[new_mode]['baud']} symbols/sec")
    
    # Station responds at same mode (to ensure delivery)
    response_train = modem.generate_train('left', discovery_mode)
    print(f"  Station TX: ACK (Left Spin, {discovery_mode} mode)")
    
    rx_response = simulator.simulate_hf_channel(response_train, include_fading=True, freq_offset=True)
    
    # Master also measures
    master_measured_snr = modem.measure_snr(np.real(rx_response), discovery_mode)
    master_mode, master_reason = modem.select_speed_mode(master_measured_snr)
    print(f"  Master RX: Measured SNR = {master_measured_snr:.1f}dB")
    print(f"  Master confirms: {master_mode} mode")
    
    # === PHASE 3: DATA TRANSFER (NEW MODE) ===
    print(f"\n[PHASE 3] DATA TRANSFER - Using {new_mode} mode")
    
    # Station sends message "HI" (2 chars = 16 bits)
    message_bits = [0,1,0,0,1,0,0,0, 0,1,0,0,1,0,0,1]  # "HI" in ASCII
    data_signal = modem.encode_bits(message_bits, new_mode)
    
    transfer_time = len(data_signal) / SAMPLE_RATE
    print(f"  Station TX: 'HI' (16 bits)")
    print(f"  Transfer time: {transfer_time:.2f} seconds")
    print(f"  Effective rate: {16/transfer_time:.2f} bits/sec")
    
    # Apply channel
    rx_data = simulator.simulate_hf_channel(data_signal, include_fading=True, freq_offset=True)
    rx_data_audio = np.real(rx_data)
    
    # Master receives (simulate detection of first bit)
    first_bit_len = int(SPEED_MODES[new_mode]['interval'] * SPEED_MODES[new_mode]['integration'] * SAMPLE_RATE)
    first_bit_rx = rx_data_audio[:first_bit_len]
    
    integrated = modem.coherent_integrate(first_bit_rx, new_mode)
    iq = modem.sdr_downconvert(integrated)
    
    # Detect spin
    center = len(iq) // 2
    pulse_width = int(SPEED_MODES[new_mode]['sigma'] * 8 * SAMPLE_RATE)
    iq_pulse = iq[center - pulse_width : center + pulse_width]
    
    phase = np.unwrap(np.angle(iq_pulse))
    inst_freq = np.diff(phase) / (2 * np.pi / SAMPLE_RATE)
    avg_freq = np.mean(inst_freq)
    
    detected_spin = 'RIGHT' if avg_freq > 0 else 'LEFT'
    expected_spin = 'LEFT'  # First bit is 0
    
    print(f"  Master RX: First bit detected as {detected_spin} (Expected: {expected_spin})")
    
    # === VISUALIZATION ===
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Speed Mode Table
    ax1 = plt.subplot(3, 2, 1)
    ax1.axis('off')
    mode_data = []
    for mode_name, params in SPEED_MODES.items():
        marker = '→' if mode_name == new_mode else ' '
        mode_data.append([
            f"{marker} {mode_name}",
            f"{params['min_snr']:+.0f}dB",
            f"{params['integration']}x",
            f"{params['baud']} baud"
        ])
    
    table = ax1.table(cellText=mode_data, 
                     colLabels=['Mode', 'Min SNR', 'Integration', 'Rate'],
                     cellLoc='left', loc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax1.set_title(f"Speed Modes (Selected: {new_mode})", fontweight='bold', pad=20)
    
    # 2. SNR Measurement
    ax2 = plt.subplot(3, 2, 2)
    modes_list = list(SPEED_MODES.keys())
    min_snrs = [SPEED_MODES[m]['min_snr'] for m in modes_list]
    ax2.barh(modes_list, min_snrs, color=['red' if m != new_mode else 'green' for m in modes_list])
    ax2.axvline(measured_snr, color='blue', linestyle='--', linewidth=2, label=f'Measured: {measured_snr:.1f}dB')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_title('SNR vs Speed Mode Thresholds')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. CQ Signal (Clean vs Noisy)
    ax3 = plt.subplot(3, 2, 3)
    t_cq = np.arange(len(cq_train)) / SAMPLE_RATE
    ax3.plot(t_cq, cq_train, 'b', linewidth=1, label='Clean TX')
    ax3.set_title('Phase 1: Discovery CQ (NORMAL mode)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    ax4 = plt.subplot(3, 2, 4)
    t_rx_cq = np.arange(len(rx_cq_audio)) / SAMPLE_RATE
    ax4.plot(t_rx_cq[::5], rx_cq_audio[::5], 'r', linewidth=0.3, label=f'RX @ {snr_db}dB')
    ax4.set_title(f'Phase 1: Received CQ (SNR: {measured_snr:.1f}dB)')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Data Transfer (First few bits)
    ax5 = plt.subplot(3, 1, 3)
    t_data = np.arange(len(rx_data_audio)) / SAMPLE_RATE
    ax5.plot(t_data[::20], rx_data_audio[::20], 'k', linewidth=0.2)
    ax5.set_title(f"Phase 3: Data Transfer '{chr(72)}{chr(73)}' ({new_mode} mode, {transfer_time:.1f}s)")
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Amplitude')
    ax5.grid(alpha=0.3)
    
    # Annotate bit boundaries
    for i in range(min(4, len(message_bits))):
        bit_time = i * SPEED_MODES[new_mode]['interval'] * SPEED_MODES[new_mode]['integration']
        bit_val = message_bits[i]
        ax5.axvline(bit_time, color='g', linestyle=':', alpha=0.5)
        ax5.text(bit_time + 0.5, ax5.get_ylim()[1]*0.8, f"bit{i}={bit_val}", rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('data/11_auto_speed_protocol.png', dpi=150)
    print(f"\n[SAVED] Visualization: data/11_auto_speed_protocol.png")
    
    # Save audio
    wavfile.write('data/11_auto_speed_data_clean.wav', SAMPLE_RATE, 
                  (data_signal / np.max(np.abs(data_signal)) * 0.5 * 32767).astype(np.int16))
    wavfile.write('data/11_auto_speed_data_noisy.wav', SAMPLE_RATE,
                  (rx_data_audio / np.max(np.abs(rx_data_audio)) * 0.5 * 32767).astype(np.int16))
    
    print(f"[SAVED] Audio: data/11_auto_speed_data_clean.wav")
    print(f"[SAVED] Audio: data/11_auto_speed_data_noisy.wav")
    
    print("\n" + "=" * 70)
    print("PROTOCOL SUMMARY")
    print("=" * 70)
    print(f"Channel SNR:        {snr_db}dB")
    print(f"Measured SNR:       {measured_snr:.1f}dB")
    print(f"Selected Mode:      {new_mode}")
    print(f"Integration Factor: {SPEED_MODES[new_mode]['integration']}x")
    print(f"Data Rate:          {16/transfer_time:.2f} bits/sec")
    print(f"Transfer Time:      {transfer_time:.2f}s for 2 characters")
    print("=" * 70)

if __name__ == "__main__":
    simulate_auto_speed_protocol()
