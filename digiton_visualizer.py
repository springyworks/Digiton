import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from hf_channel_simulator import WaveletPingGenerator, SAMPLE_RATE

def gaussian_pulse(t, t0, sigma, f0):
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * f0 * t)
    return envelope * carrier

def plot_spectrogram(ax, x, fs, title):
    f, t, Sxx = signal.spectrogram(x, fs, nperseg=256, noverlap=200)
    ax.pcolormesh(t * 1000, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylim(0, 4000)

# Setup
fs = SAMPLE_RATE
duration = 0.2
t = np.linspace(0, duration, int(fs * duration))

# 1. True Gabor Digiton (Gaussian)
sigma = 0.02 # 20ms width
gabor_ping = gaussian_pulse(t, 0.1, sigma, 1500)

# 2. Project Ping (Exponential Decay)
gen = WaveletPingGenerator(fs)
project_ping = gen.generate_ping(wavelet_name='morl', duration=duration)
# Normalize length
if len(project_ping) > len(t):
    project_ping = project_ping[:len(t)]
else:
    project_ping = np.pad(project_ping, (0, len(t) - len(project_ping)))

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Heisenberg Digiton (Gaussian) vs. Project Ping (Exponential)", fontsize=16)

# Time Domain
axes[0, 0].plot(t * 1000, gabor_ping)
axes[0, 0].set_title("True Digiton (Gaussian Envelope)")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t * 1000, project_ping, color='orange')
axes[0, 1].set_title("Project Ping (Exponential Decay)")
axes[0, 1].set_xlabel("Time (ms)")
axes[0, 1].grid(True, alpha=0.3)

# Frequency Domain (Spectrogram)
plot_spectrogram(axes[1, 0], gabor_ping, fs, "Digiton Spectrogram (Compact)")
plot_spectrogram(axes[1, 1], project_ping, fs, "Project Ping Spectrogram (Wideband Tail)")

plt.tight_layout()
plt.savefig('data/11_digiton_comparison.png')
print("Comparison plot saved to data/11_digiton_comparison.png")
