# Digiton System Capabilities & Testing Guide

This document outlines the core capabilities of the Spin Digiton Modem and the test suite required to verify its functionality and prevent regressions.

## 1. System Capabilities

The modem must maintain the following operational standards:

### Core Modem Performance

- **Signal Detection**: Reliable detection of "Spin" states (frequency offsets) using Complex Morlet Wavelets.
- **SNR Sensitivity**:
  - **Standard Operation**: > -10 dB SNR.
  - **Deep Mode**: Down to **-60 dB SNR** using Coherent Integration (up to 256x repeats).
- **Speed Adaptation**: Support for multiple modes:
  - `TURBO` (200 baud)
  - `FAST` (50 baud)
  - `NORMAL` (20 baud)
  - `SLOW` (12 baud)
  - `DEEP` (1 baud)

### Protocol (Wavelet Party Protocol)

- **Roles**:
  - **Initiator**: Sets the tempo by transmitting the `WWBEAT` signal.
  - **Responder**: Syncs to `WWBEAT` and transmits in assigned slots.
- **Handshake**: Robust link establishment even in deep noise.
- **Collision Avoidance**: Slotted Aloha mechanism.

### Channel Resilience

- **Watterson Model**: Resilience to multipath fading and Doppler spread typical of HF channels.

---

## 2. Required Test Suite

To ensure the system functions correctly, the following tests must be executed.

### A. Regression Tests (Automated)

*Run these after every code change to ensure basic functionality.*

| Test Script | Description | Expected Outcome |
| :--- | :--- | :--- |
| `tests/regression/test_modem.py` | Verifies core modulation/demodulation, symbol encoding, and basic signal processing. | **PASS**: All assertions hold. |
| `tests/regression/test_chat.py` | Verifies the chat interface logic and message handling. | **PASS**: Message queues function correctly. |

**Command:**

```bash
python3 run_tests.py
```

### B. Experimental & Stress Tests (Manual/Periodic)

*Run these to verify performance limits and physics simulations.*

| Test Script | Description | Critical Check |
| :--- | :--- | :--- |
| `tests/experiments/test_wpp.py` | **Protocol Stress Test**. Simulates a full handshake at **-60 dB SNR**. | Must detect `WWBEAT` and establish link. *(Note: Requires real-time sync)* |
| `tests/experiments/test_physics.py` | **Physics Verification**. Checks Heisenberg uncertainty limits and wavelet properties. | Must validate time-frequency constraints. |
| `tests/experiments/test_hf_channel.py` | **Channel Simulator**. Verifies the Watterson channel model implementation. | Must generate valid fading profiles. |

**Command:**

```bash
python3 tests/experiments/test_wpp.py
# etc...
```

## 3. Speed Adaptation & Discovery Strategy

### The "Ultra-Mode" Dilemma
When the modem operates in `TURBO` or `FAST` mode (high SNR, short pings), the signal energy is low. A distant, weak station (which requires `DEEP` mode integration) would normally fail to detect these fast pings, making the Initiator "invisible" to new, weak peers.

### Solution: Lighthouse Beacons
To solve this, the protocol implements a **Lighthouse Strategy**:
- Even when in `TURBO` or `FAST` mode, the Initiator periodically (e.g., every 10th cycle) transmits a **DEEP Mode Sequence**.
- This high-energy, repeated sequence acts as a beacon, allowing weak stations to detect the grid and request a speed downgrade or announce their presence.
- **Status**: Implemented in `digiton/protocol/wpp.py`.

## 4. Release Checklist

Before marking a version as stable:

1. [ ] All Regression Tests pass (`run_tests.py`).
2. [ ] `test_wpp.py` passes at -60dB (verifies Deep Mode).
3. [ ] No "Master" or "CQ" terminology remains in the codebase (use "Initiator" / "WWBEAT").
