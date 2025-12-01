import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import sounddevice as sd
import queue
import sys
import os
import threading
import time
from scipy import signal

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from digiton.sdr import iq_downconvert

# Configuration
SAMPLE_RATE = 48000
DEFAULT_CENTER_FREQ = 1500
MAX_WINDOW_SIZE = 48000  # 1 second buffer

# Use Dark Background for better visibility (TensorBoard/Cyberpunk style)
plt.style.use('dark_background')

class RealtimeCorkscrew:
    def __init__(self):
        self.q = queue.Queue()
        self.buffer = np.zeros(MAX_WINDOW_SIZE, dtype=np.float32)
        self.fft_history = [] # List of (freqs, amps)
        self.max_fft_lines = 60 # Increased for longer trail
        
        # Setup Plot
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.subplots_adjust(bottom=0.25) # Make room for sliders
        
        # --- Plot 1: Corkscrew (Time Pipe) ---
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.line1, = self.ax1.plot([], [], [], lw=2.0, alpha=0.9, color='#00ff00') # Bright Green
        self.ax1.set_title("1. I/Q Corkscrew (Time Flow)")
        self.ax1.set_xlabel("Real (I)")
        self.ax1.set_ylabel("Imag (Q)")
        self.ax1.set_zlabel("Time (s)")
        self.ax1.view_init(elev=20, azim=45)
        
        # --- Plot 2: 3D FFT Waterfall ---
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.lines2 = []
        
        # Create a colormap for the waterfall (Plasma is good for intensity)
        self.cmap = plt.get_cmap('plasma')
        
        # Pre-allocate lines for waterfall
        for i in range(self.max_fft_lines):
            # Color will be set dynamically in update
            l, = self.ax2.plot([], [], [], alpha=0.8, lw=1.5)
            self.lines2.append(l)
            
        self.ax2.set_title("2. 3D FFT Waterfall")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.set_zlabel("Time (s)")
        self.ax2.view_init(elev=20, azim=45) # Match initial view

        # View Sync State
        self.last_elev = 20
        self.last_azim = 45
        self.last_dist = 10 # Default dist is usually around 10

        # --- Controls (Sliders) ---
        # 1. Time Scale (Window Size)
        ax_time = self.fig.add_axes([0.1, 0.1, 0.3, 0.03])
        self.s_time = Slider(ax_time, 'Time Window', 0.05, 1.0, valinit=0.2)
        
        # 2. Zoom (Amplitude)
        ax_zoom = self.fig.add_axes([0.1, 0.05, 0.3, 0.03])
        self.s_zoom = Slider(ax_zoom, 'Zoom (Amp)', 0.1, 5.0, valinit=1.0)
        
        # 3. Speed (Decimation/Update)
        ax_speed = self.fig.add_axes([0.5, 0.1, 0.3, 0.03])
        self.s_speed = Slider(ax_speed, 'Sim Speed', 1, 10, valinit=4, valstep=1)
        
        # 4. Threshold (Cut-through)
        ax_thresh = self.fig.add_axes([0.5, 0.05, 0.3, 0.03])
        self.s_thresh = Slider(ax_thresh, 'Threshold', 0.0, 0.5, valinit=0.0)
        
        # 5. Center Frequency
        ax_freq = self.fig.add_axes([0.1, 0.15, 0.7, 0.03])
        self.s_freq = Slider(ax_freq, 'Center Freq', 500, 2500, valinit=DEFAULT_CENTER_FREQ)

        # Test Signal Button
        self.btn_ax = self.fig.add_axes([0.85, 0.05, 0.1, 0.075])
        self.btn = Button(self.btn_ax, 'Inject Test')
        self.btn.on_clicked(self.inject_test_signal)

    def inject_test_signal(self, event):
        print("Injecting Test Signal...")
        t = threading.Thread(target=self._feed_test_signal)
        t.daemon = True
        t.start()

    def _feed_test_signal(self):
        duration = 0.5
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        sigma = 0.05
        envelope = np.exp(-(t - duration/2)**2 / (2 * sigma**2))
        
        # Use current slider freq + 50Hz
        f_center = self.s_freq.val
        sig = 0.5 * envelope * np.cos(2 * np.pi * (f_center + 50) * t)
        sig = sig.astype(np.float32)
        
        chunk_size = 1024
        delay = chunk_size / SAMPLE_RATE
        for i in range(0, len(sig), chunk_size):
            chunk = sig[i:i+chunk_size]
            self.q.put(chunk.reshape(-1, 1))
            time.sleep(delay)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def update(self, frame):
        # 1. Consume Data
        try:
            while True:
                data = self.q.get_nowait()
                data = data.flatten()
                shift = len(data)
                if shift >= MAX_WINDOW_SIZE:
                    self.buffer[:] = data[-MAX_WINDOW_SIZE:]
                else:
                    self.buffer = np.roll(self.buffer, -shift)
                    self.buffer[-shift:] = data
        except queue.Empty:
            pass
        
        # 2. Read Sliders
        time_window_s = self.s_time.val
        zoom_level = self.s_zoom.val
        down_rate = int(self.s_speed.val)
        threshold = self.s_thresh.val
        center_freq = self.s_freq.val
        
        # 3. Prepare Data View
        window_samples = int(time_window_s * SAMPLE_RATE)
        view_data = self.buffer[-window_samples:]
        
        # --- Update Plot 1: Corkscrew ---
        iq = iq_downconvert(view_data, center_freq, SAMPLE_RATE)
        iq_plot = iq[::down_rate]
        
        # Apply Threshold (Cut-through)
        # Simple gate: if mag < threshold, set to 0 (or NaN to break line)
        mag = np.abs(iq_plot)
        mask = mag > threshold
        
        # To make lines "break" at low amp, we can insert NaNs, 
        # but for speed let's just plot everything and rely on alpha/zoom
        # Or just zero it out
        iq_plot[~mask] = 0 
        
        t_axis = np.linspace(0, time_window_s, len(iq_plot))
        
        self.line1.set_data(np.real(iq_plot) * zoom_level, np.imag(iq_plot) * zoom_level)
        self.line1.set_3d_properties(t_axis)
        
        # Update Limits 1
        limit = 0.5 / zoom_level if zoom_level > 0 else 0.5
        self.ax1.set_xlim(-limit, limit)
        self.ax1.set_ylim(-limit, limit)
        self.ax1.set_zlim(0, time_window_s)
        
        # --- Update Plot 2: Waterfall ---
        # Compute FFT of the *latest* chunk of view_data (e.g. last 1024 samples)
        fft_size = 512
        if len(view_data) >= fft_size:
            latest = view_data[-fft_size:] * np.hanning(fft_size)
            fft = np.fft.rfft(latest)
            freqs = np.fft.rfftfreq(fft_size, 1/SAMPLE_RATE)
            amps = np.abs(fft) / fft_size
            
            # Filter to interesting band (Center +/- 500Hz)
            mask = (freqs > center_freq - 500) & (freqs < center_freq + 500)
            f_sub = freqs[mask]
            a_sub = amps[mask] * zoom_level * 5 # Boost for visibility
            
            self.fft_history.append((f_sub, a_sub))
            if len(self.fft_history) > self.max_fft_lines:
                self.fft_history.pop(0)
        
        # Draw Waterfall Lines
        for i, line in enumerate(self.lines2):
            if i < len(self.fft_history):
                f, a = self.fft_history[i]
                # Z position is relative index
                z = np.full_like(f, i * (time_window_s / self.max_fft_lines))
                
                line.set_data(f, a)
                line.set_3d_properties(z)
                
                # Dynamic Color based on age (index)
                # Newest (end of list) = Bright Yellow/Orange
                # Oldest (start of list) = Dark Purple/Blue
                # Normalize index 0..1
                norm_idx = i / self.max_fft_lines
                color = self.cmap(norm_idx)
                line.set_color(color)
                
            else:
                line.set_data([], [])
                line.set_3d_properties([])
                
        self.ax2.set_xlim(center_freq - 500, center_freq + 500)
        self.ax2.set_ylim(0, 1.0) # Amplitude
        self.ax2.set_zlim(0, time_window_s)

        # --- Sync Views ---
        # Check if ax1 moved
        if (self.ax1.elev != self.last_elev) or (self.ax1.azim != self.last_azim):
            self.ax2.view_init(elev=self.ax1.elev, azim=self.ax1.azim)
            self.last_elev = self.ax1.elev
            self.last_azim = self.ax1.azim
        # Check if ax2 moved
        elif (self.ax2.elev != self.last_elev) or (self.ax2.azim != self.last_azim):
            self.ax1.view_init(elev=self.ax2.elev, azim=self.ax2.azim)
            self.last_elev = self.ax2.elev
            self.last_azim = self.ax2.azim

        return [self.line1] + self.lines2

    def start(self):
        print("Starting Real-time 3D Listener...")
        print(f"Listening on default device (PipeWire).")
        
        stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=self.audio_callback,
            blocksize=1024
        )
        
        with stream:
            ani = FuncAnimation(self.fig, self.update, interval=50, blit=False)
            plt.show()

if __name__ == "__main__":
    try:
        viz = RealtimeCorkscrew()
        viz.start()
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
