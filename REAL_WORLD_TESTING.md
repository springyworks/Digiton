# Real-World Echo Testing Guide

This guide explains how to perform a real-world "Echo Test" using the Digiton modem, a physical radio transmitter, and a remote WebSDR receiver.

## Objective
Measure the **Round Trip Time (RTT)** and **Signal Quality (SNR)** of a ping signal sent from your location, propagated through the ionosphere, received by a remote WebSDR, and routed back to your computer via the internet.

## Prerequisites

1.  **Hardware**:
    *   HF Transceiver (e.g., Icom IC-705) connected to your computer via USB Audio.
    *   Antenna tuned to the target frequency (e.g., 14.070 MHz or similar data segment).
2.  **Software**:
    *   **PipeWire** (Ubuntu 24.04 default) or PulseAudio.
    *   **Pavucontrol** (PulseAudio Volume Control) for routing.
    *   **Web Browser** open to a WebSDR (e.g., [Maasbree WebSDR](http://sdr.websdrmaasbree.nl:8901/)).
3.  **Python Environment**:
    *   `sounddevice` installed (`pip install sounddevice`).

## Setup Instructions

### 1. Configure WebSDR
1.  Open the WebSDR in your browser.
2.  Tune to your transmit frequency.
3.  Set the mode to **USB** (Upper Sideband).
4.  Adjust the filter bandwidth to approx 3kHz (standard voice/data).
5.  Ensure you can hear the static/audio from the browser.

### 2. Configure Audio Routing (PipeWire/Pavucontrol)
1.  Open `pavucontrol`.
2.  **Input Routing**:
    *   The Python script will look for the "Default Input" device.
    *   In `pavucontrol` -> **Recording** tab, find the Python application (once running).
    *   Change its source to **"Monitor of [Your Audio Output]"** (to capture the WebSDR audio from the browser).
    *   *Alternatively*, route the browser audio to a virtual sink and record from that.
3.  **Output Routing**:
    *   In `pavucontrol` -> **Playback** tab, find the Python application.
    *   Change its output to your **USB Audio Codec** (the radio interface).

### 3. Run the Test
Execute the test script:

```bash
python3 apps/real_echo_test.py
```

### 4. Operation
1.  The script will list available devices and start listening.
2.  Press **ENTER** to transmit a Ping.
3.  **Watch the Terminal**:
    *   You will see "Transmitting Ping...".
    *   The radio should key up (VOX or CAT control required, or manual PTT).
    *   After a few seconds (propagation + internet lag), you should see:
        `[RX EVENT] SNR: 15.2 dB | Peak: 45.2`
        `>>> ROUND TRIP TIME: 1450.2 ms`

## Troubleshooting
*   **Feedback Loop**: If you hear a loud squeal, you are recording your own output. Ensure you are recording the *WebSDR* (Browser) audio, not the *Radio* audio.
*   **No Detection**:
    *   Check volume levels. The WebSDR audio should be loud enough but not clipping.
    *   Check frequency alignment. If the WebSDR is off by >50Hz, detection might fail (though Morlet is robust).
