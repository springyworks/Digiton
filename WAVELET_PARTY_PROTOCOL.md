 what should the protocol do? ;;0. make noticible we are there (CQ) on frequency (HAM HF bands), anyone may respond 1. detect the faintest of signals ;2. if signal allows then trhottle up 3. if there are more than one responder, very low priority- try to create a party-group-chat 4. participants/responders can be from all over the world 5. all participant have excellent stable  clocks and 


# Wavelet Party Protocol (WPP) & The Physics of the Ping

## Part 1: Why Wavelets? (The Physics)

You asked why a **Wavelet Ping** (specifically a Morlet wavelet) is more effective than a simple **Decaying Sinusoid** (Damped Sine Wave) for detecting signals in deep noise (-60dB).

The answer lies in **Signal Processing Physics** and the **Heisenberg Uncertainty Principle**.

### 1. The Gabor Limit (Time-Frequency Optimality)
*   **The Problem:** To detect a signal in noise, you want to filter out everything that *isn't* the signal.
    *   If you filter tightly in **Frequency** (pure sine), you lose time resolution (the signal "rings" forever).
    *   If you filter tightly in **Time** (short click), you accept too much broadband noise.
*   **The Solution:** The **Gaussian Envelope**.
    *   A Morlet wavelet is a Sine wave multiplied by a Gaussian window ($e^{-t^2}$).
    *   Mathematically, this shape minimizes the product of time-spread and frequency-spread. It is the **most compact signal possible** in the time-frequency domain.
    *   **Result:** It captures the maximum amount of signal energy in the smallest possible "box" of time and frequency, excluding the maximum amount of noise.

### 2. Matched Filter Sharpness (Autocorrelation)
*   **Damped Sine:** Starts instantly and decays slowly. Its autocorrelation (the "detection spike") is asymmetric and wide. It smears out the detection peak.
*   **Wavelet:** Is symmetric (grows, peaks, decays). Its autocorrelation is a perfect, sharp spike.
    *   **Benefit:** In the presence of **Multipath** (HF radio echoes), a sharp correlation peak allows you to distinguish the "Direct Path" from the "Reflected Path". A smeared peak merges them into a mess.

### 3. Robustness to "Static Crashes"
*   Lightning and electrical sparks look like **Damped Sinusoids** (broadband impulses that ring in tuned circuits).
*   If you use a Damped Sine as your ping, your detector will trigger on every lightning crash.
*   A **Wavelet** has a specific "smooth start, smooth end" shape that does not occur naturally in spark noise. The matched filter rejects the sharp edges of lightning static much better.

---

## Part 2: The Wavelet Party Protocol (WPP)

The goal is to automate the discovery of stations on a frequency without using precise wall-clock time (GPS) or complex modems.

### The "Rhythm" Concept
Instead of digital packets, we use a **Musical Rhythm** established by the Master (Net Control).

### 1. The CQ-Ping (The Metronome)

*   **Master Station** transmits a Wavelet Ping every **T** seconds (e.g., 3.2s).
*   This establishes the **Grid**.
*   **Listeners** hear the Ping. They don't need to know the time. They just know "The next Ping is in 3.2 seconds". They synchronize their internal stopwatch to the Ping arrival.

### 2. Slotted Aloha (The Response)

*   The time between CQ-Pings is divided into **8 Slots**.
*   **Slot 0:** Master CQ-Ping.
*   **Slots 1-7:** Available for Responders.
*   A new station (Responder) wants to join. It picks a **Random Slot** (e.g., Slot 4).
*   It waits for the CQ-Ping, counts to Slot 4, and transmits a **PONG**.

### 3. Collision & Backoff
*   **Scenario:** Station B and Station C both pick Slot 4.
*   **Result:** They transmit together. The Master hears a "Garbled" or "Loud" signal but might not decode a clean ID (if we were sending IDs).
*   **Resolution:**
    *   If the Master does not ACK them, they assume failure.
    *   In the next cycle, they pick **New Random Slots**.
    *   Probability says they will eventually pick different slots (e.g., B picks 2, C picks 6).

### 4. The "Party" (Connected State)
*   Once the Master identifies a station in a specific slot (e.g., "I hear you in Slot 2"), that slot is **Assigned**.
*   The Master can now use that slot to send data specifically to that station, or request data from them.
*   The "Party" grows as more slots are filled.

### Summary
This protocol mimics a group of people clapping in rhythm. Even if you join late, you can hear the beat, wait for a pause, and clap your hands to say "I'm here!".
