import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def generate_digiton(t, t_center, f_hz, width_s):
    """
    Generates a 'Digiton' - A Gaussian-enveloped sinusoid.
    This is the Gabor Atom, the most efficient signal in nature.
    """
    # Sigma is derived from width. Let's say width is 6 sigma (99.7% energy)
    sigma = width_s / 6.0 
    
    # Gaussian Envelope
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    
    # Carrier Wave
    carrier = np.cos(2 * np.pi * f_hz * t)
    
    return envelope * carrier

def simulate_chat_sequence():
    print("--- DIGITON CHAT SIMULATION ---")
    print("Protocol: Wavelet Party Protocol (WPP)")
    print("Signal:   Gaussian Digiton (Gabor Atom)")
    print("-" * 60)

    # Parameters
    fs = 48000
    cycle_duration = 1.0 # Speed up for demo (usually 3.2s)
    num_slots = 8
    slot_duration = cycle_duration / num_slots
    total_cycles = 4
    total_time = total_cycles * cycle_duration
    
    t = np.linspace(0, total_time, int(total_time * fs))
    signal = np.zeros_like(t)
    
    # Frequencies (Different stations might use slightly different tones, 
    # or same tone in different slots. Let's use same tone for WPP standard)
    freq = 800 
    digiton_width = 0.05 # 50ms pulses
    
    # --- CYCLE 1: Master Calls CQ ---
    print("[0.0s] Cycle 1: MASTER sends CQ-Ping (Slot 0)")
    signal += generate_digiton(t, 0.0 + (slot_duration * 0.5), freq, digiton_width)
    
    # --- CYCLE 2: Alice Joins ---
    print(f"[{cycle_duration:.1f}s] Cycle 2: MASTER sends CQ")
    signal += generate_digiton(t, cycle_duration + (slot_duration * 0.5), freq, digiton_width)
    
    print(f"[{cycle_duration + 2*slot_duration:.1f}s] Cycle 2: ALICE responds in Slot 2")
    # Alice is in Slot 2 (Indices 0..7, so 3rd slot)
    t_alice = cycle_duration + (2 * slot_duration) + (slot_duration * 0.5)
    signal += generate_digiton(t, t_alice, freq, digiton_width) * 0.8 # Slightly weaker
    
    # --- CYCLE 3: Bob Joins, Alice Continues ---
    print(f"[{2*cycle_duration:.1f}s] Cycle 3: MASTER sends CQ")
    signal += generate_digiton(t, 2*cycle_duration + (slot_duration * 0.5), freq, digiton_width)
    
    print(f"[{2*cycle_duration + 2*slot_duration:.1f}s] Cycle 3: ALICE maintains Slot 2")
    t_alice_3 = 2*cycle_duration + (2 * slot_duration) + (slot_duration * 0.5)
    signal += generate_digiton(t, t_alice_3, freq, digiton_width) * 0.8
    
    print(f"[{2*cycle_duration + 5*slot_duration:.1f}s] Cycle 3: BOB joins in Slot 5")
    t_bob = 2*cycle_duration + (5 * slot_duration) + (slot_duration * 0.5)
    signal += generate_digiton(t, t_bob, freq, digiton_width) * 0.6 # Weaker station
    
    # --- CYCLE 4: Full Party (Data Exchange) ---
    print(f"[{3*cycle_duration:.1f}s] Cycle 4: MASTER sends CQ")
    signal += generate_digiton(t, 3*cycle_duration + (slot_duration * 0.5), freq, digiton_width)
    
    # Alice sends "Data" (Double Digiton)
    print(f"[{3*cycle_duration + 2*slot_duration:.1f}s] Cycle 4: ALICE sends DATA (Double Pulse)")
    t_alice_4a = 3*cycle_duration + (2 * slot_duration) + (slot_duration * 0.3)
    t_alice_4b = 3*cycle_duration + (2 * slot_duration) + (slot_duration * 0.7)
    signal += generate_digiton(t, t_alice_4a, freq, digiton_width/2) * 0.8
    signal += generate_digiton(t, t_alice_4b, freq, digiton_width/2) * 0.8
    
    # Bob maintains
    t_bob_4 = 3*cycle_duration + (5 * slot_duration) + (slot_duration * 0.5)
    signal += generate_digiton(t, t_bob_4, freq, digiton_width) * 0.6

    # --- VISUALIZATION ---
    plt.figure(figsize=(14, 6))
    plt.plot(t, signal, 'k', linewidth=0.8)
    
    # Annotate
    plt.title("Wavelet Party Protocol: Digiton Chat Sequence")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Draw Slot Boundaries
    for i in range(int(total_cycles * num_slots)):
        plt.axvline(i * slot_duration, color='g', linestyle=':', alpha=0.3)
        
    # Label Cycles
    for i in range(total_cycles):
        plt.text(i * cycle_duration + 0.05, 1.1, f"CYCLE {i+1}", fontsize=12, fontweight='bold', color='blue')
        plt.text(i * cycle_duration + 0.05, 0.9, "MASTER (CQ)", fontsize=8, color='blue')

    # Label Stations
    # Alice Cycle 2
    plt.text(t_alice, 0.9, "ALICE (Join)", ha='center', color='green')
    # Bob Cycle 3
    plt.text(t_bob, 0.7, "BOB (Join)", ha='center', color='red')
    # Alice Data
    plt.text(t_alice_4a, 0.9, "ALICE (Data)", ha='center', color='green')

    plt.ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig('data/06_digiton_chat_sequence.png')
    print("-" * 60)
    print("Visualization saved to 'digiton_chat_sequence.png'")

    # Save Audio
    # Normalize to avoid clipping
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal_norm = signal / max_val * 0.9 # Leave some headroom
    else:
        signal_norm = signal
    
    # Convert to 16-bit PCM
    signal_int16 = (signal_norm * 32767).astype(np.int16)
    wavfile.write('digiton_chat_sequence.wav', fs, signal_int16)
    print("Audio saved to 'digiton_chat_sequence.wav'")

if __name__ == "__main__":
    simulate_chat_sequence()
