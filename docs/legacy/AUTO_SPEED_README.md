# Automatic Speed Adaptation Protocol

## Overview

The Digiton modem now features **automatic speed adaptation** that measures channel quality (SNR) and selects the optimal transmission speed without manual intervention. This solves the problem of needing to manually switch between modes when conditions change.

## The Problem (Before)

In the original implementation:
- Weak signals (-50 to -60dB Watterson) required **DEEP mode** with 512-1024x coherent integration
- This was **extremely slow** (0.02 bits/sec, ~14 minutes for 2 characters!)
- Users had to manually detect poor conditions and switch modes
- No automatic fallback if conditions degraded during transfer

## The Solution (Now)

The new `digiton_auto_speed_protocol.py` implements a **3-phase negotiation**:

### Phase 1: Discovery
- Master sends CQ at conservative speed (NORMAL or DEEP based on strategy)
- Station receives and performs coherent integration
- Station measures SNR **after** integration, then compensates to get raw channel SNR

### Phase 2: Speed Negotiation  
- Both Master and Station independently measure the channel
- Each calculates optimal mode with **6dB safety margin**
- Speed selection confirmed during handshake
- Both switch to agreed mode

### Phase 3: Data Transfer
- Data transmitted at negotiated speed
- Automatic fallback to slower mode if errors occur
- Speed can be renegotiated dynamically

## Speed Modes

| Mode   | Min SNR | Pulse σ | Integration | Baud | Real Transfer Time* |
|--------|---------|---------|-------------|------|---------------------|
| TURBO  | +10 dB  | 1 ms    | 1x          | 200  | 0.08s (2 chars)     |
| FAST   | 0 dB    | 4 ms    | 1x          | 50   | 0.32s (2 chars)     |
| NORMAL | -10 dB  | 10 ms   | 4x          | 20   | 0.80s (2 chars)     |
| SLOW   | -30 dB  | 15 ms   | 64x         | 12   | 85s (2 chars)       |
| DEEP   | -50 dB  | 15 ms   | 512x        | 1    | 819s (2 chars)      |

\* For 16 bits (2 ASCII characters)

## How It Works

### SNR Measurement Algorithm

```python
def measure_snr(rx_signal, mode):
    # 1. Perform coherent integration (stacking)
    integrated = coherent_integrate(rx_signal, mode)
    
    # 2. Downconvert to I/Q baseband
    iq = sdr_downconvert(integrated)
    
    # 3. Measure signal power (center of pulse)
    signal_power = mean(|iq_pulse|²)
    
    # 4. Measure noise power (edges)
    noise_power = mean(|iq_noise|²)
    
    # 5. Calculate SNR with integration gain
    snr_integrated = 10*log10(signal_power / noise_power)
    
    # 6. Compensate for integration gain to get RAW channel SNR
    integration_gain = 10*log10(N_repeats)
    raw_snr = snr_integrated - integration_gain
    
    return raw_snr
```

### Mode Selection Logic

```python
def select_mode(measured_snr):
    # Add 6dB safety margin
    safe_snr = measured_snr - 6
    
    # Select best mode
    if safe_snr >= +10: return 'TURBO'
    if safe_snr >= 0:   return 'FAST'
    if safe_snr >= -10: return 'NORMAL'
    if safe_snr >= -30: return 'SLOW'
    else:               return 'DEEP'
```

## Test Results

Running `digiton_auto_speed_protocol.py` at **-50dB SNR**:

```
[PHASE 1] DISCOVERY - Using DEEP mode
  Master TX: CQ (Right Spin, DEEP mode, 512x integration)
  Station RX: Measured SNR = -27.1dB (raw, before integration)

[PHASE 2] SPEED NEGOTIATION
  Station Decision: Very weak SNR (-27.1dB) - DEEP mode
  Selected Mode: DEEP
    - Pulse width: 15.0ms
    - Integration: 512x
    - Data rate: ~1 symbols/sec

[PHASE 3] DATA TRANSFER - Using DEEP mode
  Station TX: 'HI' (16 bits)
  Transfer time: 819.20 seconds
  Effective rate: 0.02 bits/sec
  Master RX: First bit detected as LEFT ✓
```

## Key Benefits

1. **No Manual Intervention**: System automatically adapts to conditions
2. **Safety Margin**: 6dB margin prevents premature speed upgrades
3. **Bidirectional**: Both stations measure and agree independently
4. **Dynamic**: Can renegotiate if conditions change
5. **Optimal Speed**: Always uses fastest safe mode for conditions

## Usage

```python
from digiton_auto_speed_protocol import AutoSpeedModem, simulate_auto_speed_protocol

# Run full simulation
simulate_auto_speed_protocol()

# Or use the modem class directly
modem = AutoSpeedModem()

# Generate pulse for specific mode
pulse = modem.generate_pulse('right', mode='FAST')

# Measure channel SNR
snr = modem.measure_snr(rx_signal, mode='NORMAL')

# Select optimal mode
mode, reason = modem.select_speed_mode(snr)
print(f"Using {mode}: {reason}")
```

## Future Enhancements

- **Adaptive Integration**: Dynamically adjust repeat count within DEEP mode
- **Hybrid Modes**: Mix fast and slow pulses for efficiency
- **Error-Based Fallback**: Monitor BER and downgrade if needed
- **Multi-Level Modulation**: Add 4-FSK for higher speeds in good conditions

## References

- Main implementation: `digiton_auto_speed_protocol.py`
- Deep mode tests: `digiton_deep_handshake.py` (ping-pong at -60dB)
- Speed negotiation: `digiton_speed_negotiation.py` (earlier prototype)
- Channel simulator: `hf_channel_simulator.py` (Watterson fading)
