import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import plotly.graph_objects as go
import sys
import os

import base64

def analyze_wav_3d(wav_path, center_freq=1500, output_html='data/05_3d_corkscrew.html', output_png='data/05_3d_corkscrew.png'):
    print(f"--- 3D DIGITON ANALYZER ---")
    print(f"Loading: {wav_path}")
    
    # 1. Load WAV
    try:
        fs, data = wavfile.read(wav_path)
        # Read raw bytes for embedding
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
            wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
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
        mode='lines',
        line=dict(
            width=3,
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
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='In-Phase (I)',
            zaxis_title='Quadrature (Q)',
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            camera=dict(
                eye=dict(x=1.8, y=0, z=1.2) # Start position
            )
        )
    )
    
    print(f"Generating HTML with custom navigation and rotation...")
    
    # Get the plot div (without full HTML wrapper)
    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Custom HTML Template
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Digiton 3D Visualization</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        .nav {{ background: #f8f9fa; padding: 15px 20px; border-bottom: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .nav a {{ text-decoration: none; color: #3498db; font-weight: bold; }}
        .nav a:hover {{ text-decoration: underline; }}
        .nav span {{ color: #6c757d; margin: 0 10px; }}
        .nav .current {{ color: #2c3e50; font-weight: bold; }}
        .controls {{ position: absolute; top: 80px; right: 20px; z-index: 100; background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); min-width: 200px; }}
        .audio-player {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; }}
        button {{ background: #3498db; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; width: 100%; margin-bottom: 5px; }}
        button:hover {{ background: #2980b9; }}
        .time-display {{ font-size: 12px; color: #666; margin-top: 5px; font-family: monospace; }}
        .progress-bar {{ width: 100%; height: 4px; background: #ddd; border-radius: 2px; margin-top: 5px; position: relative; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #3498db; width: 0%; transition: width 0.1s linear; }}
        .hint {{ font-size: 11px; color: #999; margin-top: 5px; font-style: italic; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="index.html">🌀 Digiton Home</a> <span>&rsaquo;</span> <span class="current">3D Corkscrew Visualization</span>
    </div>
    
    <div class="controls">
        <div><small><strong>Interactive Mode:</strong></small></div>
        <div class="audio-player">
            <button id="playBtn" onclick="toggleAudio()">▶️ Play Audio</button>
            <button id="seekBtn" onclick="toggleSeekMode()" style="background: #27ae60;">🎯 Click to Seek</button>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar"></div>
            </div>
            <div class="time-display" id="timeDisplay">0.00s / {duration:.2f}s</div>
            <div class="hint">💡 Click points to jump to that time</div>
            <audio id="audioPlayer" style="display:none;">
                <source src="data:audio/wav;base64,{wav_b64}" type="audio/wav">
            </audio>
        </div>
    </div>

    {plot_div}

    <script>
        var isRotating = false;
        var animationId = null;
        var seekMode = true; // Start with seek mode enabled
        var audioDuration = {duration};

        // Audio control
        function toggleAudio() {{
            var audio = document.getElementById('audioPlayer');
            var btn = document.getElementById('playBtn');
            if (audio.paused) {{
                audio.volume = 0.18; // -15dB
                audio.play();
                btn.textContent = '⏸️ Pause Audio';
                startProgressUpdate();
            }} else {{
                audio.pause();
                btn.textContent = '▶️ Play Audio';
            }}
        }}

        function toggleSeekMode() {{
            seekMode = !seekMode;
            var btn = document.getElementById('seekBtn');
            if (seekMode) {{
                btn.textContent = '🎯 Click to Seek';
                btn.style.background = '#27ae60';
            }} else {{
                btn.textContent = '🔒 Seek Disabled';
                btn.style.background = '#95a5a6';
            }}
        }}

        function startProgressUpdate() {{
            var audio = document.getElementById('audioPlayer');
            var progressBar = document.getElementById('progressBar');
            var timeDisplay = document.getElementById('timeDisplay');
            
            function updateProgress() {{
                if (!audio.paused) {{
                    var percent = (audio.currentTime / audio.duration) * 100;
                    progressBar.style.width = percent + '%';
                    timeDisplay.textContent = audio.currentTime.toFixed(2) + 's / ' + audio.duration.toFixed(2) + 's';
                    
                    // Update playback line on plot
                    updatePlaybackLine(audio.currentTime);
                    
                    requestAnimationFrame(updateProgress);
                }}
            }}
            requestAnimationFrame(updateProgress);
        }}

        function updatePlaybackLine(currentTime) {{
            var gd = document.getElementsByClassName('plotly-graph-div')[0];
            if (!gd) return;
            
            // Create a vertical plane (curtain) at the current playback time
            var update = {{
                x: [[currentTime, currentTime, currentTime, currentTime]],
                y: [[-1, -1, 1, 1]],
                z: [[-1, 1, -1, 1]],
                i: [[0, 0]],
                j: [[1, 2]],
                k: [[2, 3]],
            }};
            
            // Check if playback plane already exists
            var planeExists = false;
            if (gd.data) {{
                for (var i = 0; i < gd.data.length; i++) {{
                    if (gd.data[i].name === 'Playback Position') {{
                        planeExists = true;
                        // Update existing plane
                        Plotly.restyle(gd, update, i);
                        break;
                    }}
                }}
            }}
            
            // Add plane if it doesn't exist
            if (!planeExists) {{
                var playbackPlane = {{
                    type: 'mesh3d',
                    x: [currentTime, currentTime, currentTime, currentTime],
                    y: [-1, -1, 1, 1],
                    z: [-1, 1, -1, 1],
                    i: [0, 0],
                    j: [1, 2],
                    k: [2, 3],
                    opacity: 0.5,
                    color: '#e74c3c',
                    name: 'Playback Position',
                    hoverinfo: 'skip',
                    showlegend: false
                }};
                Plotly.addTraces(gd, playbackPlane);
            }}
        }}

        // Click handler for seeking
        var checkPlot = setInterval(function() {{
            var gd = document.getElementsByClassName('plotly-graph-div')[0];
            if (gd && gd._fullLayout) {{
                clearInterval(checkPlot);
                startRotation(gd);
                setupInteractionDetection(gd);
                setupClickSeek(gd);
            }}
        }}, 100);

        var lastSeekTime = 0;
        var seekDebounceMs = 500; // Minimum 500ms between seeks
        
        function setupClickSeek(gd) {{
            gd.on('plotly_click', function(data) {{
                if (!seekMode) return;
                
                // Debounce: prevent rapid-fire seeks during click-hold
                var now = Date.now();
                if (now - lastSeekTime < seekDebounceMs) {{
                    return; // Ignore this click
                }}
                lastSeekTime = now;
                
                if (data.points && data.points.length > 0) {{
                    var point = data.points[0];
                    var clickedTime = point.x; // X-axis is time
                    
                    if (clickedTime >= 0 && clickedTime <= audioDuration) {{
                        var audio = document.getElementById('audioPlayer');
                        audio.volume = 0.18; // -15dB
                        audio.currentTime = clickedTime;
                        
                        // Auto-play if not already playing
                        if (audio.paused) {{
                            audio.play();
                            document.getElementById('playBtn').textContent = '⏸️ Pause Audio';
                            startProgressUpdate();
                        }}
                        
                        // Visual feedback
                        console.log('Seeking to: ' + clickedTime.toFixed(2) + 's');
                    }}
                }}
            }});
            
            // Highlight on hover
            gd.on('plotly_hover', function(data) {{
                if (!seekMode) return;
                if (data.points && data.points.length > 0) {{
                    var point = data.points[0];
                    var timeDisplay = document.getElementById('timeDisplay');
                    timeDisplay.textContent = '➤ ' + point.x.toFixed(2) + 's (click to seek)';
                }}
            }});
            
            gd.on('plotly_unhover', function() {{
                var audio = document.getElementById('audioPlayer');
                var timeDisplay = document.getElementById('timeDisplay');
                timeDisplay.textContent = audio.currentTime.toFixed(2) + 's / ' + audio.duration.toFixed(2) + 's';
            }});
        }}

        function setupInteractionDetection(gd) {{
            // Stop rotation when user interacts with camera controls
            gd.on('plotly_relayout', function(eventData) {{
                // Only stop if it's a camera change (not our seek operation)
                if (eventData['scene.camera'] && isRotating) {{
                    stopRotation();
                }}
            }});
            
            // Stop on drag
            var isDragging = false;
            gd.addEventListener('mousedown', function(e) {{
                // Check if clicking on the 3D plot area (not the controls)
                if (e.target.closest('.plotly-graph-div')) {{
                    isDragging = true;
                }}
            }});
            
            gd.addEventListener('mousemove', function(e) {{
                if (isDragging && isRotating) {{
                    stopRotation();
                }}
            }});
            
            gd.addEventListener('mouseup', function() {{
                isDragging = false;
            }});
        }}

        function stopRotation() {{
            isRotating = false;
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
        }}

        function startRotation(gd) {{
            isRotating = true;
            var t = 0;
            var radius = 1.8;
            var speed = 0.01;
            var duration = 6.28;
            var startTime = null;
            
            function animate(timestamp) {{
                if (!isRotating) return;
                
                if (!startTime) startTime = timestamp;
                
                t += speed;
                var x = radius * Math.cos(t);
                var y = radius * Math.sin(t);
                
                Plotly.relayout(gd, 'scene.camera.eye', {{x: x, y: y, z: 1.2}});
                
                if (t < duration && isRotating) {{
                    animationId = requestAnimationFrame(animate);
                }} else {{
                    isRotating = false;
                }}
            }}
            
            setTimeout(function() {{
                if (isRotating) {{
                    animationId = requestAnimationFrame(animate);
                }}
            }}, 1000);
        }}
    </script>
</body>
</html>
    """
    
    with open(output_html, 'w') as f:
        f.write(html_content)
        
    print(f"✓ Saved interactive HTML to {output_html}")

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
