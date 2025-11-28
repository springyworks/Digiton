#!/usr/bin/env python3
"""
Generate clean 3D reference plots of Complex Morlet Wavelets
showing canonical right-turn and left-turn corkscrew trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

os.makedirs('data', exist_ok=True)

def generate_complex_morlet(t, f_offset, sigma=0.01):
    """
    Generate a Complex Morlet Wavelet
    s(t) = A(t) * exp(j*2*pi*f*t)
    where A(t) = exp(-t^2 / (2*sigma^2))
    
    For real-valued audio representation:
    s_real(t) = A(t) * cos(2*pi*f*t)
    
    For I/Q representation after downconversion:
    s_IQ(t) = A(t) * exp(j*2*pi*f_offset*t)
    """
    envelope = np.exp(-t**2 / (2 * sigma**2))
    # This is the complex baseband representation (after downconversion)
    iq_signal = envelope * np.exp(1j * 2 * np.pi * f_offset * t)
    return iq_signal

def plot_reference_wavelets():
    """Create clean reference 3D plots for MANUAL.md"""
    
    # Time vector - centered around zero for clean visualization
    fs = 8000
    sigma = 0.015  # 15ms pulse width
    t = np.linspace(-4*sigma, 4*sigma, int(8*sigma*fs))
    
    # Generate Right and Left spin wavelets (baseband I/Q)
    f_right = +200  # Right spin: +200 Hz offset
    f_left = -200   # Left spin: -200 Hz offset
    
    iq_right = generate_complex_morlet(t, f_right, sigma)
    iq_left = generate_complex_morlet(t, f_left, sigma)
    
    # Extract I, Q components
    I_right = np.real(iq_right)
    Q_right = np.imag(iq_right)
    
    I_left = np.real(iq_left)
    Q_left = np.imag(iq_left)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 7))
    
    # === RIGHT SPIN (Right-Hand Helix) ===
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Color by time (progression)
    colors = cm.viridis(np.linspace(0, 1, len(t)))
    
    # Plot as connected line with color gradient
    for i in range(len(t)-1):
        ax1.plot(I_right[i:i+2], Q_right[i:i+2], t[i:i+2]*1000, 
                color=colors[i], linewidth=2, alpha=0.8)
    
    # Add arrow to show direction
    arrow_idx = len(t) // 4
    ax1.quiver(I_right[arrow_idx], Q_right[arrow_idx], t[arrow_idx]*1000,
              I_right[arrow_idx+10] - I_right[arrow_idx],
              Q_right[arrow_idx+10] - Q_right[arrow_idx],
              (t[arrow_idx+10] - t[arrow_idx])*1000,
              color='red', arrow_length_ratio=0.3, linewidth=2)
    
    ax1.set_xlabel('I (In-Phase)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Q (Quadrature)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('RIGHT SPIN (+200 Hz)\nRight-Hand Helix', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # Add reference plane at t=0
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    ax1.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
    
    # === LEFT SPIN (Left-Hand Helix) ===
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot as connected line with color gradient
    for i in range(len(t)-1):
        ax2.plot(I_left[i:i+2], Q_left[i:i+2], t[i:i+2]*1000, 
                color=colors[i], linewidth=2, alpha=0.8)
    
    # Add arrow to show direction
    ax2.quiver(I_left[arrow_idx], Q_left[arrow_idx], t[arrow_idx]*1000,
              I_left[arrow_idx+10] - I_left[arrow_idx],
              Q_left[arrow_idx+10] - Q_left[arrow_idx],
              (t[arrow_idx+10] - t[arrow_idx])*1000,
              color='red', arrow_length_ratio=0.3, linewidth=2)
    
    ax2.set_xlabel('I (In-Phase)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Q (Quadrature)', fontsize=12, fontweight='bold')
    ax2.set_zlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('LEFT SPIN (-200 Hz)\nLeft-Hand Helix', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.3)
    
    # Add reference plane at t=0
    ax2.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save high-resolution figure
    plt.savefig('data/reference_morlet_wavelets_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data/reference_morlet_wavelets_3d.png")
    
    # === Create a second figure showing the mathematical representation ===
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Right Spin - I component
    axes[0, 0].plot(t*1000, I_right, 'b', linewidth=2)
    axes[0, 0].set_title('RIGHT SPIN: I (In-Phase)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linewidth=0.5)
    
    # Right Spin - Q component
    axes[0, 1].plot(t*1000, Q_right, 'r', linewidth=2)
    axes[0, 1].set_title('RIGHT SPIN: Q (Quadrature)', fontweight='bold')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='k', linewidth=0.5)
    
    # Left Spin - I component
    axes[1, 0].plot(t*1000, I_left, 'b', linewidth=2)
    axes[1, 0].set_title('LEFT SPIN: I (In-Phase)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='k', linewidth=0.5)
    
    # Left Spin - Q component
    axes[1, 1].plot(t*1000, Q_left, 'r', linewidth=2)
    axes[1, 1].set_title('LEFT SPIN: Q (Quadrature)', fontweight='bold')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('data/reference_morlet_iq_components.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data/reference_morlet_iq_components.png")
    
    # === Create envelope and spectrum plot ===
    fig3, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gaussian envelope
    envelope = np.exp(-t**2 / (2 * sigma**2))
    axes[0].plot(t*1000, envelope, 'k', linewidth=3, label='Gaussian Envelope')
    axes[0].fill_between(t*1000, 0, envelope, alpha=0.3)
    axes[0].set_title('Gaussian Envelope: A(t) = exp(-t²/2σ²)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (ms)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # Add sigma markers
    axes[0].axvline(-3*sigma*1000, color='r', linestyle='--', alpha=0.5, label='±3σ (99.7%)')
    axes[0].axvline(+3*sigma*1000, color='r', linestyle='--', alpha=0.5)
    axes[0].text(0, 0.5, f'σ = {sigma*1000:.1f} ms', fontsize=12, 
                ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Frequency spectrum (FFT)
    from scipy import signal
    f_right_sig = envelope * np.cos(2 * np.pi * (1500 + f_right) * t)
    f_left_sig = envelope * np.cos(2 * np.pi * (1500 + f_left) * t)
    
    f, Pxx_right = signal.welch(f_right_sig, fs, nperseg=1024)
    f, Pxx_left = signal.welch(f_left_sig, fs, nperseg=1024)
    
    axes[1].semilogy(f, Pxx_right, 'b', linewidth=2, label='Right Spin (1700 Hz)', alpha=0.7)
    axes[1].semilogy(f, Pxx_left, 'r', linewidth=2, label='Left Spin (1300 Hz)', alpha=0.7)
    axes[1].set_title('Frequency Spectrum', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylabel('Power Spectral Density', fontsize=12)
    axes[1].set_xlim(1000, 2000)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].axvline(1500, color='k', linestyle=':', alpha=0.5, label='Center (1500 Hz)')
    
    plt.tight_layout()
    plt.savefig('data/reference_morlet_envelope_spectrum.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data/reference_morlet_envelope_spectrum.png")
    
    plt.show()

if __name__ == "__main__":
    print("Generating reference Complex Morlet Wavelet visualizations...")
    plot_reference_wavelets()
    print("\n✓ All reference plots generated successfully!")
