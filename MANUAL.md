---
title: "DIGITON MODEM - Technical Manual & User Guide"
author: "Digiton Project Team"
date: "November 29, 2025"
geometry: margin=1in
fontsize: 11pt
documentclass: article
toc: true
toc-depth: 3
colorlinks: true
linkcolor: blue
urlcolor: blue
---

# DIGITON MODEM
## Technical Manual & User Guide

**Version:** 2.1  
**Date:** November 29, 2025  
**Author:** Digiton Project Team  
**License:** MIT

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theory of Operation](#2-theory-of-operation)
3. [Use Cases & Operational Scenarios](#3-use-cases--operational-scenarios)
4. [Speed Adaptation System](#4-speed-adaptation-system)
5. [Installation & Setup](#5-installation--setup)
6. [Basic Operation](#6-basic-operation)
7. [Advanced Features](#7-advanced-features)
8. [Test Results & Validation](#8-test-results--validation)
9. [Troubleshooting](#9-troubleshooting)
10. [API Reference](#10-api-reference)
11. [Appendix](#11-appendix)

---

## 1. Introduction

### 1.1 What is Digiton?

**Digiton** is an experimental acoustic modem that uses **Complex Morlet Wavelets** (Gaussian-enveloped sinusoids) with **Spin Digiton modulation** for robust data transmission over acoustic channels. The system is designed to operate in extremely challenging conditions, including:

- **Watterson fading channels** (HF propagation simulation)
- **Very low SNR** (-60dB operation with coherent integration)
- **Multipath interference** and Doppler spread
- **Automatic speed adaptation** based on channel quality

### 1.2 Key Features

✅ **5 Automatic Speed Modes** (TURBO to DEEP)  
✅ **-60dB SNR Operation** with coherent integration  
✅ **Watterson Channel Resilience** via frequency diversity  
✅ **Complex I/Q Processing** for robust spin detection  
✅ **Time-Slotted Protocol** (Wavelet Party Protocol)  
✅ **3D Visualization** of signal trajectories  
✅ **Real-time Speed Negotiation** between stations

### 1.3 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              DIGITON MODEM ARCHITECTURE                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Application Layer                                      │
│  ├─ Chat Protocol                                       │
│  ├─ File Transfer                                       │
│  └─ Burst Data                                          │
│                                                         │
│  MAC Layer (WPP)                                        │
│  ├─ Time Slots (8 × 0.4s)                               │
│  ├─ WWBEAT (Initiator Heartbeat)                        │
│  └─ Speed Negotiation                                   │
│                                                         │
│  Physical Layer                                         │
│  ├─ Spin Digiton Modulation (±200 Hz)                   │
│  ├─ Complex Morlet Wavelets                             │
│  ├─ I/Q Downconversion                                  │
│  ├─ Coherent Integration (1x - 1024x)                   │
│  └─ Matched Filter Detection                            │
│                                                         │
│  Channel Simulator                                      │
│  ├─ Watterson Fading Model                              │
│  ├─ AWGN Noise (-60dB to +20dB)                         │
│  ├─ Multipath (2-20ms delay)                            │
│  └─ Doppler Spread (0.1-2 Hz)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Theory of Operation

### 2.1 Spin Digiton Modulation

The **Spin Digiton** uses frequency offset to encode binary data:
- **Right Spin (1)**: +200 Hz offset from carrier ($f_c + \Delta f$)
- **Left Spin (0)**: -200 Hz offset from carrier ($f_c - \Delta f$)

This is essentially **Binary Frequency Shift Keying (BFSK)**, but implemented using **Gabor Atoms** (Gaussian-windowed sinusoids) which achieve the theoretical minimum time-frequency uncertainty (Heisenberg limit).

### 2.2 Coherent Integration (Deep Mode)

To operate at **-60dB SNR**, the modem employs **Coherent Integration**. Instead of transmitting a single pulse, the same symbol is repeated $N$ times (e.g., $N=256$). The receiver aligns and sums these repeats *before* detection.

- **Signal Power** increases by factor $N$.
- **Noise Power** increases by factor $\sqrt{N}$ (assuming uncorrelated noise).
- **SNR Gain**: $10 \log_{10}(N)$.

For $N=256$, the gain is $\approx 24$ dB, allowing a -60dB signal to be detected as if it were -36dB (which is then further enhanced by the matched filter processing gain).

---

## 3. Use Cases & Operational Scenarios

This section outlines typical operational scenarios for the Digiton system, focusing on the interaction between the human operator, the protocol layers, and the physical medium.

### 3.1 Use Case 1: "The Lonely Initiator" (Discovery)

**Scenario:**
A HAM radio operator ("Alice") turns on her transceiver and computer. She wants to make contact with anyone, anywhere. She starts the Digiton software in **Initiator Mode**.

**Actors:**
* **Alice (User)**: The human operator.
* **Digiton Stack**: The software modem (Layers 1-7).
* **Ether**: The HF radio channel.

**Sequence Diagram (SYSML-style):**

```text
+-------------------+       +-------------------+       +-------------------+
|   Alice (User)    |       |   Digiton Stack   |       |       Ether       |
|     (Layer 8)     |       |   (Layers 1-7)    |       |      (Medium)     |
+---------+---------+       +---------+---------+       +---------+---------+
          |                           |                           |
          | 1. "Start Initiator"      |                           |
          +-------------------------->|                           |
          |                           |                           |
          |                           | 2. Enter WPP Loop         |
          |                           |                           |
          |                           | 3. [L2] Wait for Slot 0   |
          |                           |                           |
          |                           | 4. [L1] TX WWBEAT (Ping)  |
          |                           +-------------------------->| >>> (Ping) >>>
          |                           |                           |
          |                           | 5. [L2] Listen Slots 1-7  |
          |                           |    (Rx Mode)              |
          |                           |                           |
          |                           |      (Silence...)         |
          |                           |                           |
          |                           | 6. [L2] Loop -> Slot 0    |
          |                           |                           |
          |                           | 7. [L1] TX WWBEAT         |
          |                           +-------------------------->| >>> (Ping) >>>
          |                           |                           |
          | <--- "Broadcasting..." ---+                           |
          |                           |                           |
```

### 3.2 Use Case 2: "The Deep Contact" (Handshake)

**Scenario:**
"Bob" is listening on a WebSDR in Maasbree (Netherlands). He is far away and the signal is weak (-40dB). He hears Alice's `WWBEAT` (via Coherent Integration) and decides to respond.

**Conditions:**
* **Alice**: Initiator (Local).
* **Bob**: Responder (Remote, -40dB path).
* **Latency**: 0.4s slot duration accommodates WebSDR lag.

**Sequence Diagram:**

```text
+-------------------+       +-------------------+       +-------------------+       +-------------------+
|   Alice (User)    |       |   Alice Stack     |       |       Ether       |       |    Bob Stack      |
|     (Layer 8)     |       |   (Layers 1-7)    |       |      (Medium)     |       |   (Responder)     |
+---------+---------+       +---------+---------+       +---------+---------+       +---------+---------+
          |                           |                           |                           |
          |                           | 1. TX WWBEAT (Repeats)    |                           |
          |                           +-------------------------->| >>> (Weak Sig) >>>        |
          |                           |                           |                           |
          |                           |                           |                           | 2. [L1] Detect (Integrate)
          |                           |                           |                           |
          |                           |                           |                           | 3. [L2] Sync Clock
          |                           |                           |                           |
          |                           |                           |                           | 4. [L2] Pick Slot 3
          |                           |                           |                           |
          |                           |      (Slots 1, 2...)      |                           |
          |                           |                           |                           |
          |                           |                           | 5. [L1] TX PONG (Slot 3)  |
          |                           |                           |<--------------------------+
          |                           |                           |                           |
          |                           | 6. [L1] Detect PONG       |                           |
          |                           |    (Correlation)          |                           |
          |                           |                           |                           |
          | 7. "New Friend!"          | 7. [L2] Register Bob      |                           |
          |<--------------------------+    (Slot 3 assigned)      |                           |
          |                           |                           |                           |
```

### 3.3 Use Case 3: "The Party Mix" (Multi-User / Auto-Speed)

**Scenario:**
Alice has established contact with **Bob** (Weak, -40dB) and **Charlie** (Strong, +10dB, Local). She wants to chat with both.
* **Challenge**: Bob needs slow, repeated signals. Charlie can talk fast.
* **Solution**: The protocol adapts.

**Sequence Diagram:**

```text
+-------+   +-------+       +-------+       +-------+       +-------+
| Alice |   | Stack |       | Ether |       |  Bob  |       |Charlie|
| (L8)  |   | (L1-7)|       |       |       |(-40dB)|       |(+10dB)|
+---+---+   +---+---+       +---+---+       +---+---+       +---+---+
    |           |               |               |               |
    | 1. "Hi"   |               |               |               |
    +---------->|               |               |               |
    |           | 2. [L3] Route |               |               |
    |           |               |               |               |
    |           | 3. [L2] Sched |               |               |
    |           |    (Hybrid)   |               |               |
    |           |               |               |               |
    |           | 4. TX (DEEP)  |               |               |
    |           +------------------------------>|               |
    |           | "H...i..."    |               |               |
    |           | (Repeats)     |               |               |
    |           |               |               |               |
    |           | 5. TX (TURBO) |               |               |
    |           +---------------------------------------------->|
    |           | "Hi" (Fast)   |               |               |
    |           |               |               |               |
    |           |               | 6. RX "Hi"    |               |
    |           |               |<--------------+               |
    |           |               |               |               |
    |           |               |               | 7. RX "Hi"    |
    |           |               |<------------------------------+
    |           |               |               |               |
```

### 3.4 OSI Layer Mapping

| Layer | Name | Digiton Implementation |
| :--- | :--- | :--- |
| **8** | **User** | The HAM Operator (Alice/Bob). |
| **7** | **Application** | Chat Window / Visualizer (`apps/listener.py`). |
| **6** | **Presentation** | ASCII / UTF-8 Encoding. |
| **5** | **Session** | "Party" Management (Who is in which slot?). |
| **4** | **Transport** | Reliable Delivery (ACKs - *Planned*). |
| **3** | **Network** | Routing (Unicast to Slot ID vs Broadcast). |
| **2** | **Data Link** | **WPP (Wavelet Party Protocol)**. Slotted Aloha, Sync, Collision Avoidance. |
| **1** | **Physical** | **Spin Modem**. Morlet Wavelets, Coherent Integration (Repeats). |

---

## 4. Speed Adaptation System

The modem automatically selects the best speed mode based on the measured SNR of the received signal.

| Mode | Baud Rate | SNR Req. | Integration | Use Case |
|------|-----------|----------|-------------|----------|
| **TURBO** | 200 | +10 dB | 1× | Clean channels, high speed |
| **FAST** | 50 | 0 dB | 4× | Normal operation |
| **NORMAL** | 20 | -10 dB | 16× | Moderate fading |
| **SLOW** | 12 | -30 dB | 128× | Severe fading |
| **DEEP** | 1 (base) | -50 dB | 1024× | Extreme weak signal |

---

## 5. Installation & Setup

### 5.1 Prerequisites

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`, `PyWavelets`

### 5.2 Installation

```bash
git clone https://github.com/springyworks/Digiton.git
cd Digiton
pip install -r requirements.txt
```

---

## 6. Basic Operation

### 6.1 Running the Listener

To start the modem in listening mode (Responder):

```bash
python3 apps/listener.py
```

### 6.2 Running the Initiator

To start the modem as the Initiator (broadcasting WWBEAT):

```bash
python3 apps/initiator.py
```

*(Note: Ensure your audio device is configured correctly in the script settings)*

---

## 7. Advanced Features

### 7.1 3D Visualization

The `digiton_visualizer.py` tool provides a 3D view of the signal in the Time-Frequency-Amplitude domain. This is useful for understanding the "Corkscrew" nature of the Spin Digiton signal.

### 7.2 Channel Simulation

The built-in `HFChannelSimulator` allows testing without a radio. It simulates:
- Multipath fading (Watterson model)
- Doppler spread
- Additive White Gaussian Noise (AWGN)

---

## 8. Test Results & Validation

### 8.1 Deep SNR Stress Test (Nov 2025)

| SNR (dB) | Repeats | Result | Notes |
| :--- | :--- | :--- | :--- |
| **-20 dB** | 1 | **SUCCESS** | Link established immediately. |
| **-40 dB** | 16 | **SUCCESS** | Deep noise. 16x repeats required. |
| **-60 dB** | 256 | **SUCCESS** | **EXTREME LIMIT.** Signal 60dB below noise floor. |

---

## 9. Troubleshooting

- **No Signal Detected**: Check audio input levels. Ensure `WWBEAT` is being transmitted.
- **High Error Rate**: Try reducing the speed mode (e.g., force `SLOW` mode).
- **Synchronization Loss**: Ensure the Initiator is transmitting a stable `WWBEAT`.

---

## 10. API Reference

See the inline documentation in `digiton/` for detailed API usage.

---

## 11. Appendix

### 11.1 Glossary

- **WWBEAT**: Worldwide Beat. The heartbeat signal sent by the Initiator.
- **Initiator**: The station that sets the timing grid.
- **Responder**: A station that syncs to the Initiator.
- **Spin**: The frequency offset (±200Hz) used to encode data.
