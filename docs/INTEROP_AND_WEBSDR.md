# Digiton (Python) <-> MorletModem (Rust) Interoperability & WebSDR Integration

## Overview

This document outlines the architectural strategy for integrating the Python research prototype (`Digiton`) with the high-performance Rust implementation (`MorletModemRust`), specifically in the context of HF Radio operations and WebSDR (Software Defined Radio) networks.

## 1. The "Hybrid" Architecture

The strengths of each language are leveraged as follows:

* **Python (`Digiton`)**:
  * **Orchestration**: Handles complex, low-speed logic like negotiating frequencies, parsing WebSDR HTML interfaces, and managing the "Deep Handshake" state machine.
  * **I/O Bridge**: Connects to WebSDR audio streams (via PulseAudio/ALSA or direct HTTP stream decoding) and pipes raw samples to the Rust backend.
  * **Visualization**: Plots the "Spin" (chirplet) diagrams and constellation maps in real-time using Matplotlib/VisPy.

* **Rust (`MorletModemRust`)**:
  * **DSP Core**: Performs the heavy lifting: Continuous Wavelet Transform (CWT), matched filtering, and symbol recovery.
  * **Low Latency**: Processes incoming audio chunks faster than real-time, allowing the Python layer to "catch up" or handle jitter.
  * **Headless Operation**: Can run as a background service or a subprocess spawned by the Python controller.

## 2. WebSDR Integration (Maasbree & Others)

### The Loop

1. **Transmission**: The local station transmits the `Digiton` signal via an HF transceiver (e.g., SSB mode).
2. **Propagation**: The signal travels via the ionosphere (Skywave) to a remote receiver.
3. **Reception**: A WebSDR station (e.g., Maasbree, Twente) receives the signal.
4. **Return Path**: The WebSDR streams the audio over the internet back to the `Digiton` control station.

### The "Routing Delay" Challenge

WebSDR streams introduce significant latency (2s to 10s) due to buffering and encoding.

* **Impact on Handshake**: Standard TCP-like handshakes will time out. The `Digiton` protocol must be "Delay Tolerant".
* **Solution: "Deep Handshake" with Long Horizons**:
  * The protocol uses "Chirplets" (Spin) which are time-invariant to a large degree.
  * The handshake timeout windows must be dynamically adjusted based on the measured Round Trip Time (RTT).
  * **Ping-Pong Estimation**: Before data transfer, `Digiton` sends a specific "Ping" chirplet. The Rust decoder looks for the "Pong" from the WebSDR stream. The time difference establishes the `System_Latency`.

## 3. HF-Watterson Simulation Space

To test without polluting the airwaves, we use the Rust `hf-channel` crate.

* **Scenario**:
  ```text
  [Python Generator] -> [Rust HF Channel Sim] -> [Rust Decoder]
  ```
* **Implementation**:
  * Python generates a clean `.wav` file.
  * Rust `hf-channel` applies:
    * **Multipath**: Discrete echoes (e.g., 2ms delay, -3dB).
    * **Doppler Spread**: Ionospheric movement simulation (Watterson model).
    * **Noise**: AWGN (Atmospheric noise).
  * Rust `decode_wav` attempts to recover the data.

## 4. Proposed Workflow

1. **Python Script (`modem_listen.py`)**:
   * Opens an audio stream (e.g., from a virtual cable connected to a browser playing WebSDR).
   * Spawns `apps/bin/decode_wav` (or a new streaming binary) as a subprocess.
   * Writes raw audio to `stdin` of the Rust process.
   * Reads JSON-formatted symbol data from `stdout` of the Rust process.

2. **Rust Streaming Binary (`stream_decode`)**:
   * *To be implemented in `apps/`*.
   * Reads `f32` or `i16` samples from `stdin`.
   * Maintains a rolling buffer (circular buffer).
   * Runs the Morlet Wavelet Transform on the buffer.
   * Outputs detected symbols immediately to `stdout`.

## 5. Next Steps

1. **Create `stream_decode` in Rust**: Adapt `decode_wav.rs` to read from a pipe instead of a file.
2. **Update Python Wrapper**: Modify `digiton_sdr_spin.py` to use the Rust binary for detection instead of the slow Python implementation.
3. **Latency Test**: Measure the exact delay of the Maasbree WebSDR to calibrate the handshake timers.
