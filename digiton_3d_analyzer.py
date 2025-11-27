import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import plotly.graph_objects as go
import sys
import os

def analyze_wav_3d(wav_path, center_freq=1500, output_html='data/05_3d_corkscrew.html', output_png='data/05_3d_corkscrew.png'):
    print(f"--- 3D DIGITON ANALYZER ---")
    print(f"Loading: {wav_path}")
    
    # 1. Load WAV
    try:
        fs, data = wavfile.read(wav_path)
    except FileNotFoundError:
        print(f"Error: File {wav_path} not found.")
        return

    # Normalize
    if data.dtype == np.int16:
        data = data / 32768.0
    
    # If stereo, take one channel
    if len(data.shape) > 1:
        data = data[:, 0]
        
    duration = len(data) / fs
    print(f"Duration: {duration:.2f}s | Sample Rate: {fs}Hz")
    
    # 2. Downconvert to I/Q (The "Trick")
    print(f"Downconverting with Center Freq: {center_freq}Hz...")
    t = np.arange(len(data)) / fs
    lo = np.exp(-1j * 2 * np.pi * center_freq * t)
    mixed = data * lo
    
    # Low Pass Filter to remove 2*fc component
    # Cutoff at 500Hz (since spin is +/- 200Hz)
    sos = signal.butter(4, 500, 'low', fs=fs, output='sos')
    iq_signal = signal.sosfilt(sos, mixed)
    
    # 3. Downsample for Plotting
    # We want to see the spirals. Spin is ~200Hz.
    # We need ~20 points per rotation to look good -> 4000 Hz plotting rate.
    target_plot_fs = 4000
    decimate_factor = int(fs / target_plot_fs)
    if decimate_factor < 1: decimate_factor = 1
    
    iq_plot = iq_signal[::decimate_factor]
    t_plot = t[::decimate_factor]
    
    print(f"Plotting {len(t_plot)} points...")

    # 4. Interactive 3D Plot with Plotly
    I = np.real(iq_plot)
    Q = np.imag(iq_plot)
    amp = np.abs(iq_plot)
    
    # Mask noise for cleaner plot
    mask = amp > 0.05
    
    # Create Plotly 3D scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=t_plot[mask],
        y=I[mask],
        z=Q[mask],
        mode='markers',
        marker=dict(
            size=2,
            color=t_plot[mask],
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="Time (s)")
        ),
        text=[f"t={t:.3f}s<br>I={i:.3f}<br>Q={q:.3f}" for t, i, q in zip(t_plot[mask], I[mask], Q[mask])],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=f"3D Digiton Corkscrews (I/Q vs Time)<br>Source: {os.path.basename(wav_path)}",
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='In-Phase (I)',
            zaxis_title='Quadrature (Q)',
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=800
    )
    
    print(f"Saving interactive HTML to {output_html}")
    fig.write_html(output_html)
    print(f"✓ Open {output_html} in your browser to fly around!")

    # 5. Static Matplotlib version for reference
    fig_static = plt.figure(figsize=(14, 10))
    ax = fig_static.add_subplot(111, projection='3d')
    
    ax.scatter(t_plot[mask], I[mask], Q[mask], c=t_plot[mask], cmap='turbo', s=1, alpha=0.6)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("In-Phase (I)")
    ax.set_zlabel("Quadrature (Q)")
    ax.set_title(f"3D Digiton Corkscrews (I/Q vs Time)\nSource: {os.path.basename(wav_path)}")
    ax.view_init(elev=20, azim=-45)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    print(f"Saving static view to {output_png}")
    plt.savefig(output_png, dpi=150)
    plt.close()
    
    print("\n✓ All done! No downloads needed - just open the HTML file!")

if __name__ == "__main__":
    target_file = 'data/02_digiton_chat_spin.wav'
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    analyze_wav_3d(target_file)
