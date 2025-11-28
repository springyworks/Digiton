# Spin Digiton Modem - Technical Datasheet

**Version:** 2.0  
**Date:** November 28, 2025  
**Status:** Experimental  
**Documentation:** MANUAL.pdf (Technical Manual & User Guide)

---

## 1. OVERVIEW

The Spin Digiton Modem is an acoustic frequency-shift communication system utilizing Gaussian-windowed carrier pulses (Complex Morlet Wavelets) with frequency offset modulation to encode binary data as "spin" states detectable by I/Q downconversion techniques.

### 1.1 Key Features
- **Automatic Speed Adaptation:** 5 modes from TURBO (200 baud) to DEEP (1 baud)
- **-60dB SNR Operation:** Coherent integration (up to 1024×) for extreme weak signals
- **Binary FSK modulation** using ±200 Hz frequency offset
- **Complex Morlet Wavelets** with Gaussian pulse shaping
- **I/Q demodulation** (Complex Baseband) for robust spin detection
- **Watterson Fading Resilience** via frequency diversity
- **Multi-access protocol** support (WPP - Wavelet Party Protocol)
- **Real-time SNR measurement** with auto-fallback

### 1.2 Target Applications
- Acoustic data transmission in harsh environments
- HF radio modem (via SSB transceiver audio interface)
- Low-power wireless sensor networks
- Educational demonstration of SDR concepts
- Near-field acoustic communication
- **Emergency/EMCOMM:** Reliable weak-signal communication

---

## 2. ELECTRICAL CHARACTERISTICS

### 2.1 Operating Parameters

| Parameter | Symbol | Min | Typ | Max | Unit | Notes |
|-----------|--------|-----|-----|-----|------|-------|
| Sample Rate | fs | 8000 | 8000 | 8000 | Hz | Fixed |
| Center Frequency | fc | 1300 | 1500 | 1700 | Hz | Audio band |
| Spin Offset | Δf | -200 | ±200 | +200 | Hz | Binary states |
| Pulse Width (σ) | σ | 1 | 10 | 20 | ms | Mode-dependent |
| Symbol Rate (TURBO) | Rs_turbo | - | 200 | - | baud | +10dB SNR |
| Symbol Rate (FAST) | Rs_fast | - | 50 | - | baud | 0dB SNR |
| Symbol Rate (NORMAL) | Rs_norm | - | 20 | - | baud | -10dB SNR |
| Symbol Rate (SLOW) | Rs_slow | - | 12 | - | baud | -30dB SNR |
| Symbol Rate (DEEP) | Rs_deep | - | 1 | - | baud | -50 to -60dB |
| Integration Factor | N | 1 | 4-512 | 1024 | - | Mode-dependent |

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
   - Result: Complex baseband signal ($I + jQ$) with spin as frequency offset
   - **Note:** This representation is mathematically identical to complex-valued signal processing used in SDRs.

3. **Spin Detection**
   - Instantaneous frequency: f_inst = (1/2π) × dφ/dt
   - Weighted average: f_avg = Σ(f_inst × |IQ|²) / Σ|IQ|²
   - Decision threshold: ±50 Hz

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
   - Result: Complex baseband signal ($I + jQ$) with spin as frequency offset
   - **Note:** This representation is mathematically identical to complex-valued signal processing used in SDRs.

3. **Spin Detection**
   - Instantaneous frequency: f_inst = (1/2π) × dφ/dt
   - Weighted average: f_avg = Σ(f_inst × |IQ|²) / Σ|IQ|²
   - Decision threshold: ±50 Hz

### 3.2 Automatic Speed Adaptation

**Speed Modes:**

| Mode   | Min SNR | Pulse σ | Integration | Baud | Use Case |
|--------|---------|---------|-------------|------|----------|
| TURBO  | +10 dB  | 1 ms    | 1x          | 200  | Local    |
| FAST   | 0 dB    | 4 ms    | 1x          | 50   | Good     |
| NORMAL | -10 dB  | 10 ms   | 4x          | 20   | Average  |
| SLOW   | -30 dB  | 15 ms   | 64x         | 12   | Weak     |
| DEEP   | -50 dB  | 15 ms   | 512x        | 1    | Extreme  |

**Protocol:**
1. Master sends CQ (NORMAL or DEEP mode based on strategy)
2. Station receives and measures SNR after coherent integration
3. SNR adjusted for integration gain to get raw channel SNR
4. Both select mode with 6dB safety margin
5. Speed negotiated during handshake
6. Data transfer at agreed speed
7. Auto-fallback if errors detected

### 3.3 Protocol Layers

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
| < -60 | > 10⁻¹ | Signal lost |
| -60 to -10 | < 10⁻³ | **Deep Search Mode** (Coherent Integration) |
| -10 to 0 | 10⁻²-10⁻³ | Slow mode (σ=20ms) |
| 0 to 10 | 10⁻³-10⁻⁴ | Standard (σ=10-20ms) |
| > 10 | < 10⁻⁴ | Fast mode (σ=5ms) |

**Note on Deep Search Mode:**
By utilizing coherent integration of repetitive pulse trains (similar to WSPR/JT65), the modem can achieve detection at extremely low SNR (-60dB). This "astonishing feature" allows for "ping & pong" exchanges well below the noise floor, suitable for long-range or high-interference environments.

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

7. **Auto-Speed Protocol Test** (`digiton_auto_speed_protocol.py`)
   - **Purpose:** Verify automatic speed negotiation at extreme SNR
   - **Method:**
     - Simulate -50dB Watterson channel
     - Master sends CQ in DEEP mode (512x coherent integration)
     - Station measures raw channel SNR (compensating for integration gain)
     - Both select appropriate mode with 6dB safety margin
     - Confirm speed selection matches channel conditions
     - Transfer test data at agreed speed
   - **Pass Criteria:**
     - SNR measurement within ±3dB of actual
     - Correct mode selected (DEEP for -50dB)
     - Successful bit detection after integration
     - No false mode upgrades (safety margin enforced)

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

#### 5.2.4 Deep Search Mode
12. **Low SNR Ping & Pong Test** (`digiton_deep_handshake.py`)
    - **Purpose:** Verify coherent integration at -60dB SNR
    - **Method:**
      - Master sends "Ping" (Right Spin Train)
      - Slave responds "Pong" (Left Spin Train)
      - Add -60dB AWGN noise
    - **Pass Criteria:**
      - Both Ping and Pong sequences detected
      - Correct spin identified for each train

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

## 9. DOCUMENTATION

### 9.1 Available Documents

| Document | File | Description |
|----------|------|-------------|
| **Technical Manual** | MANUAL.pdf | Complete user guide with theory, API, and examples (20 pages, 5MB) |
| **Datasheet** | DATASHEET.md | Technical specifications (this document) |
| **Speed Adaptation Guide** | AUTO_SPEED_README.md | Auto-speed protocol details |
| **Test Results** | TEST_RESULTS.md | Validation and performance data |
| **Protocol Spec** | WAVELET_PARTY_PROTOCOL.md | WPP MAC layer specification |
| **Chat History** | chat-continuation.md | Development notes and decisions |

### 9.2 Visualization Files

All test visualizations are in `data/` directory:
- `01_heisenberg_digiton.png` - Concept diagram
- `01_spin_digiton_modem.png` - Basic modem test
- `02_digiton_chat_spin.png` - Chat protocol demo
- `05_3d_corkscrew.png` - 3D I/Q trajectory
- `10_deep_ping_pong_test.png` - Deep mode at -60dB
- `11_auto_speed_protocol.png` - Speed adaptation demo
- `15_digiton_sdr_spin.png` - I/Q processing
- `16_digiton_spin_watterson.png` - Watterson fading test
- `17_digiton_deep_spin.png` - Deep search mode

### 9.3 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic modem test
python3 spin_digiton_modem.py

# Test auto-speed adaptation
python3 digiton_auto_speed_protocol.py

# Deep mode ping-pong at -60dB
python3 digiton_deep_handshake.py

# Generate manual PDF
python3 generate_manual_pdf.py
```

---

## 10. REVISION HISTORY

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-28 | Initial datasheet | Team |
| 2.0 | 2025-11-28 | Added auto-speed protocol, MANUAL.pdf, deep mode validation | Team |

---

## 11. CONTACT & SUPPORT

**Repository:** https://github.com/springyworks/Digiton  
**Documentation:** https://springyworks.github.io/Digiton/  
**Issue Tracker:** GitHub Issues  

---

**END OF DATASHEET**
