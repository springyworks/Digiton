import numpy as np
from scipy.io import wavfile
import digiton.wavelets
import os

# Configuration
FS = 8000
FC = 1500
OFFSET = 200
SIGMA = 0.010 # 10ms
OUTPUT_FILE = "cross_check.wav"

def generate_sequence():
    print(f"Generating {OUTPUT_FILE}...")
    
    # Sequence: Right, Silence, Left, Silence, Right
    sequence = ['right', 'silence', 'left', 'silence', 'right']
    
    audio = np.array([], dtype=np.float32)
    
    silence_duration = 0.1 # 100ms silence
    silence_samples = int(silence_duration * FS)
    silence_chunk = np.zeros(silence_samples, dtype=np.float32)
    
    for item in sequence:
        if item == 'silence':
            audio = np.concatenate([audio, silence_chunk])
        else:
            # Generate pulse
            pulse = digiton.wavelets.morlet_pulse(
                fc_hz=FC,
                spin_offset_hz=OFFSET,
                sigma=SIGMA,
                fs=FS,
                spin=item,
                trunc_sigmas=4.0
            )
            # Normalize pulse to avoid clipping when saving, though float32 is fine
            # Let's keep it at natural amplitude (max ~1.0)
            audio = np.concatenate([audio, pulse])
            
    # Normalize to +/- 0.9 for WAV
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
        
    # Save as 16-bit PCM (more standard for hound/wavfile compat)
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(OUTPUT_FILE, FS, audio_int16)
    print(f"Saved {len(audio)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_sequence()
