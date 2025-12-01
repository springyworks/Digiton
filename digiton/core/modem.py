import numpy as np
from digiton.wavelets import morlet_pulse
from digiton.sdr import iq_downconvert, detect_spin

SAMPLE_RATE = 8000
CENTER_FREQ = 1500
SPIN_OFFSET = 200 # +/- 200Hz shift

class SpinDigitonModem:
    def __init__(self, fs=SAMPLE_RATE):
        self.fs = fs
        
    def generate_pulse(self, spin='right', sigma=0.01):
        """
        Generate a single Gaussian pulse with Frequency Offset (SDR Spin).
        Right Spin (+200Hz) -> 1700Hz
        Left Spin (-200Hz) -> 1300Hz
        """
        return morlet_pulse(CENTER_FREQ, SPIN_OFFSET, sigma, self.fs, spin=spin, trunc_sigmas=3.0)

    def sdr_downconvert(self, real_signal):
        """Downconvert to Baseband I/Q at CENTER_FREQ"""
        return iq_downconvert(np.asarray(real_signal), CENTER_FREQ, self.fs, lpf_hz=500.0)
        
    def detect_spin(self, iq_chunk):
        """Analyze I/Q chunk to determine spin direction."""
        return detect_spin(np.asarray(iq_chunk), self.fs)
