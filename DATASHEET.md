# Spin Digiton Modem - Technical Datasheet

**Version:** 1.0  
**Date:** November 28, 2025  
**Status:** Experimental  

---

## 1. OVERVIEW

The Spin Digiton Modem is an acoustic frequency-shift communication system utilizing Gaussian-windowed carrier pulses with frequency offset modulation to encode binary data as "spin" states detectable by I/Q downconversion techniques.

### 1.1 Key Features
- Binary FSK modulation using ±200 Hz frequency offset
- Gaussian pulse shaping for spectral efficiency
- I/Q demodulation for spin detection
- Adaptive data rate (50-200 baud)
- Multi-access protocol support (WPP - Wavelet Party Protocol)

### 1.2 Target Applications
- Acoustic data transmission
- Low-power wireless sensor networks
- Educational demonstration of SDR concepts
- Near-field acoustic communication

---

## 2. ELECTRICAL CHARACTERISTICS

### 2.1 Operating Parameters

| Parameter | Symbol | Min | Typ | Max | Unit | Notes |
|-----------|--------|-----|-----|-----|------|-------|
| Sample Rate | fs | 8000 | 8000 | 8000 | Hz | Fixed |
| Center Frequency | fc | 1300 | 1500 | 1700 | Hz | Audio band |
| Spin Offset | Δf | -200 | ±200 | +200 | Hz | Binary states |
| Pulse Width (σ) | σ | 10 | 20 | 50 | ms | Gaussian sigma |
| Symbol Rate (Slow) | Rs_slow | 10 | 20 | 30 | baud | Standard ping |
| Symbol Rate (Fast) | Rs_fast | 100 | 200 | 250 | baud | Burst mode |

### 2.2 Signal Specifications

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Right Spin Frequency | 1700 | Hz | Logic "1" |
| Left Spin Frequency | 1300 | Hz | Logic "0" |
| Pulse Energy | 99.7% | % | Within ±3σ |
| Envelope Function | Gaussian | - | exp(-t²/2σ²) |
| Phase Continuity | 0 | rad | Single pulse |

---

## 3. FUNCTIONAL DESCRIPTION

### 3.1 Modulation
1. **Pulse Generation**
   - Gaussian envelope: A(t) = exp(-t²/2σ²)
   - Carrier modulation: s(t) = A(t) × cos(2πf·t)
   - Right Spin: f = fc + Δf = 1700 Hz
   - Left Spin: f = fc - Δf = 1300 Hz

2. **I/Q Downconversion**
   - Local Oscillator: LO(t) = exp(-j·2π·fc·t)
   - Mixed signal: IQ(t) = s(t) × LO(t)
   - Low-pass filter: 4th order Butterworth, 500 Hz cutoff
   - Result: Complex baseband signal with spin as frequency offset

3. **Spin Detection**
   - Instantaneous frequency: f_inst = (1/2π) × dφ/dt
   - Weighted average: f_avg = Σ(f_inst × |IQ|²) / Σ|IQ|²
   - Decision threshold: ±50 Hz

### 3.2 Protocol Layers

**Physical Layer (PHY)**
- Pulse generation and detection
- Spin encoding/decoding
- Noise handling

**MAC Layer (WPP)**
- Time-slotted access (8 slots per 3.2s cycle)
- CQ/ACK handshake protocol
- Adaptive speed negotiation

**Application Layer**
- Chat messaging
- Burst data transfer
- Multi-user coordination

---

## 4. PERFORMANCE SPECIFICATIONS

### 4.1 Detection Performance

| SNR (dB) | Bit Error Rate | Recommended Mode |
|----------|----------------|------------------|
| < -10 | > 10⁻² | Not recommended |
| -10 to 0 | 10⁻²-10⁻³ | Slow mode (σ=20ms) |
| 0 to 10 | 10⁻³-10⁻⁴ | Standard (σ=10-20ms) |
| > 10 | < 10⁻⁴ | Fast mode (σ=5ms) |

### 4.2 Throughput

| Mode | Pulse Width | Bits/Slot | Data Rate | Latency |
|------|-------------|-----------|-----------|---------|
| Slow Ping | 40-100ms | 1 | ~10 bps | 400ms |
| Standard | 20-40ms | 2-5 | ~20 bps | 400ms |
| Fast Burst | 10ms | 10-40 | 50-200 bps | 400ms |

---

## 5. TEST SPECIFICATIONS

### 5.1 Mandatory Compliance Tests

#### 5.1.1 Pulse Generation Tests
1. **Gaussian Envelope Verification**
   - **Purpose:** Verify pulse shape matches ideal Gaussian
   - **Method:** Generate pulse with σ=20ms, measure at ±1σ, ±2σ, ±3σ
   - **Pass Criteria:** 
     - Amplitude at ±1σ: 0.606 ± 0.01
     - Amplitude at ±2σ: 0.135 ± 0.01
     - Amplitude at ±3σ: 0.011 ± 0.005

2. **Frequency Accuracy Test**
   - **Purpose:** Verify carrier frequency matches specification
   - **Method:** FFT analysis of Right/Left spin pulses
   - **Pass Criteria:**
     - Right Spin peak: 1700 Hz ± 5 Hz
     - Left Spin peak: 1300 Hz ± 5 Hz

#### 5.1.2 I/Q Downconversion Tests
3. **LO Frequency Test**
   - **Purpose:** Verify local oscillator at 1500 Hz
   - **Method:** Inject 1500 Hz tone, measure DC offset in I/Q
   - **Pass Criteria:** I/Q output < 10 Hz offset from DC

4. **Spin Detection Accuracy**
   - **Purpose:** Verify correct spin identification at various SNR
   - **Method:** Generate 100 Right + 100 Left pulses, add AWGN
   - **Pass Criteria:**
     - SNR = 10 dB: > 99% correct detection
     - SNR = 0 dB: > 95% correct detection
     - SNR = -10 dB: > 80% correct detection

#### 5.1.3 Protocol Tests
5. **WPP Handshake Test**
   - **Purpose:** Verify Master-Station handshake sequence
   - **Method:** Simulate CQ → REQ → GRANT sequence
   - **Pass Criteria:**
     - All 3 messages detected
     - Slot timing within ±10ms
     - No false positives

6. **Multi-User Collision Test**
   - **Purpose:** Verify slot separation prevents collisions
   - **Method:** Simultaneous transmissions in adjacent slots
   - **Pass Criteria:** Both messages decoded correctly

### 5.2 Regression Test Suite

#### 5.2.1 Core Functionality
7. **Modem Reference Test** (`spin_digiton_modem.py`)
   - Generate 6-pulse sequence (3 Right, 3 Left)
   - Detect all 6 pulses with correct spin
   - Measured frequency offset within ±10 Hz

8. **Chat Protocol Test** (`digiton_chat.py`)
   - Execute 4-cycle WPP session
   - Verify slot alignment
   - Generate valid spectrogram

#### 5.2.2 Advanced Features
9. **Adaptive Speed Test** (`digiton_adaptive_speed.py`)
   - Fast burst: Transmit "HELLO" in 40ms
   - Decode all 5 characters
   - Verify burst fits within slot (< 400ms)

10. **Party Mix Test** (`digiton_party_mix.py`)
    - Weak station (-10 dB) uses slow mode
    - Strong station (+20 dB) uses fast mode
    - Both decoded without interference

#### 5.2.3 Visualization
11. **3D Corkscrew Test** (`digiton_3d_analyzer.py`)
    - Generate I/Q 3D plot
    - Verify right-hand spiral for Right Spin
    - Verify left-hand spiral for Left Spin
    - HTML playback synchronized with audio

### 5.3 Acceptance Test Procedure

**Test Sequence:**
1. Run all Python test scripts in order (01-04)
2. Verify PNG outputs match expected format
3. Check WAV files audible and 4 seconds duration
4. Open 05_3d_corkscrew.html in browser
5. Verify interactive controls functional
6. Deploy to GitHub Pages
7. Verify landing page loads all assets

**Automated Validation:**
```bash
# Run full test suite
./run_all_tests.sh

# Expected output:
# ✓ 01_spin_digiton_modem.py - 6/6 pulses detected
# ✓ 02_digiton_chat.py - 4 cycles completed
# ✓ 03_digiton_adaptive_speed.py - 'HELLO' decoded
# ✓ 04_digiton_party_mix.py - Alice & Bob separated
# ✓ 05_digiton_3d_analyzer.py - HTML generated
```

---

## 6. DESIGN CONSTRAINTS & LIMITATIONS

### 6.1 Known Limitations
- Mono audio only (frequency offset ≠ true I/Q in air)
- Fixed sample rate (8 kHz)
- No error correction coding
- Vulnerable to multipath fading
- Requires quiet acoustic environment

### 6.2 Future Enhancements
- Forward Error Correction (FEC)
- Adaptive filtering for noise
- Frequency hopping spread spectrum
- Real I/Q transmission via stereo audio
- Mobile app implementation

---

## 7. ALTERNATIVE PROJECT NAMES

In honor of Jean Morlet and the wavelet transform heritage:

### 7.1 Recommended Names
1. **MorletModem** - Direct tribute to Jean Morlet
2. **GaborPhone** - After Dennis Gabor (Gaussian × carrier = Gabor atom)
3. **WaveletWire** - Emphasizes wavelet foundation
4. **SpinWave** - Captures the spin modulation concept
5. **HelixLink** - References the 3D corkscrew topology
6. **ChirpComm** - Alludes to chirplet relationships

### 7.2 Technical Alternatives
- **Analytiq** - I/Q analytic signal focus
- **GaussNet** - Gaussian pulse networking
- **OffsetOmni** - Frequency offset omnibus
- **PhaseFlow** - Emphasizes phase evolution

### 7.3 Marketing-Friendly
- **AcoustiQ** - Acoustic + I/Q
- **SonicSpin** - Audio spin modulation
- **WaveForm** - Generic but clear
- **PulsePath** - Pulse-based communication

---

## 8. COMPLIANCE & STANDARDS

**Applicable Standards:**
- None (experimental/educational)

**Acoustic Safety:**
- All signals < 2000 Hz (non-ultrasonic)
- Recommended playback volume: < 60 dB SPL
- HTML player volume: 0.18 (-15 dB digital)

**Open Source:**
- MIT License (recommended)
- GitHub: springyworks/Digiton

---

## 9. REVISION HISTORY

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-28 | Initial datasheet | AI/User |

---

## 10. CONTACT & SUPPORT

**Repository:** https://github.com/springyworks/Digiton  
**Documentation:** https://springyworks.github.io/Digiton/  
**Issue Tracker:** GitHub Issues  

---

**END OF DATASHEET**
