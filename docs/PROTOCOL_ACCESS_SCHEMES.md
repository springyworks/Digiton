# Digiton Multiple Access Schemes: TDMA & FDMA

**Version:** 1.0  
**Date:** November 29, 2025  
**Context:** Wavelet Party Protocol (WPP) & Spin Digiton Modem

---

## 1. Overview

The Digiton system employs a **Hybrid Multiple Access** strategy to allow multiple users to coexist and communicate on the same HF radio channel. This is critical for the "Party" aspect of the protocol, where an Initiator (Host) manages a grid of Responders (Guests).

The two primary methods used are:
1.  **TDMA (Time-Division Multiple Access):** Separating users by *Time Slots*.
2.  **FDMA (Frequency-Division Multiple Access):** Separating users by *Frequency Offsets* (Spin).

---

## 2. TDMA: Time-Division Multiple Access

TDMA is the backbone of the **Wavelet Party Protocol (WPP)**. It ensures that users do not talk over each other by assigning specific time windows for transmission.

### 2.1 The Time Grid
The protocol divides time into repeating **Cycles**. Each cycle is divided into **8 Slots**.

*   **Cycle Duration:** Variable, depending on the Speed Mode (e.g., 3.2 seconds in Normal Mode).
*   **Slot Duration:** Cycle Duration / 8 (e.g., 0.4 seconds).

### 2.2 Slot Assignment
*   **Slot 0 (The Beat):** Reserved for the **Initiator**.
    *   The Initiator transmits the `WWBEAT` (Worldwide Beat) in Slot 0.
    *   This signal provides the **Timing Reference** for all other users.
    *   It contains the "Grid Info" (Speed, Mode, etc.).
*   **Slots 1-7 (The Party):** Reserved for **Responders**.
    *   When a new user hears the `WWBEAT`, they synchronize their internal clock to it.
    *   They pick a random empty slot (e.g., Slot 3) to transmit their `PONG` (Handshake).
    *   Once registered, they "own" that slot for the duration of the session.

### 2.3 Half-Full Break-in
Because the Initiator only transmits in Slot 0, it spends Slots 1-7 **listening**. This allows for a "Half-Full Break-in" capability:
*   The Initiator can hear interruptions or new users joining even while maintaining the grid.
*   This is superior to continuous transmission modes (like RTTY or standard FT8) where the transmitter is blind while sending.

---

## 3. FDMA: Frequency-Division Multiple Access

FDMA provides an additional layer of separation using the frequency domain. This is inherent to the **Spin Digiton** modulation scheme.

### 3.1 Spin Modulation as FDMA
The core modulation uses two distinct frequencies relative to the carrier ($f_c$):
*   **Left Spin (0):** $f_c - 200 \text{ Hz}$
*   **Right Spin (1):** $f_c + 200 \text{ Hz}$

While this is primarily for modulation (BFSK), it effectively creates two frequency channels. A user transmitting a "0" is spectrally distinct from a user transmitting a "1".

### 3.2 Frequency Banks (The "Orchestra")
The Digiton receiver utilizes a **Frequency Bank** (or Filter Bank) approach.
*   Instead of listening to just one center frequency (e.g., 1500 Hz), the modem can monitor multiple "Lanes".
*   **Lane A:** 1500 Hz (Default)
*   **Lane B:** 1000 Hz
*   **Lane C:** 2000 Hz

This allows:
1.  **Multi-Channel Operation:** Multiple independent "Parties" can exist in the same SSB passband (3kHz bandwidth).
2.  **Mistuning Tolerance:** If a user is off-frequency by 50Hz, the wide bandwidth of the Morlet Wavelet (and the filter bank) can still capture the energy, unlike narrow-band modes that require precise tuning.

### 3.3 The "Spin" Advantage
Because the signal is a "Spinning" vector in the complex I/Q domain, the receiver can mathematically separate signals that are overlapping in time but distinct in frequency.
*   **Orthogonality:** If User A is at 1500Hz and User B is at 1800Hz, their wavelets are largely orthogonal.
*   The receiver can decode User A while ignoring User B, effectively implementing FDMA without explicit channel switching.

---

## 4. Summary: The Hybrid Approach

Digiton combines these methods to maximize robustness:

| Feature | Method | Benefit |
| :--- | :--- | :--- |
| **Collision Avoidance** | **TDMA** | Prevents users from transmitting at the exact same time. |
| **Grid Synchronization** | **TDMA** | Keeps everyone dancing to the same beat (Slot 0). |
| **Data Modulation** | **FDMA** | Uses frequency shift (Spin) to encode bits robustly. |
| **Multi-User Capacity** | **FDMA** | Allows multiple parties in one SSB channel via Frequency Banks. |

This hybrid **TDMA/FDMA** architecture is what makes the "Party Protocol" possible, turning a chaotic HF channel into a structured communication environment.
