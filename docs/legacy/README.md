# Digiton Modem

**Advanced Acoustic Modem with Automatic Speed Adaptation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

---

## 🚀 Features

✅ **5 Automatic Speed Modes** - TURBO (200 baud) to DEEP (1 baud)  
✅ **-60dB SNR Operation** - Coherent integration for extreme weak signals  
✅ **Watterson Fading Resilience** - Frequency diversity survives multipath  
✅ **Complex I/Q Processing** - SDR-style signal processing  
✅ **Real-time Speed Negotiation** - Automatic channel-aware adaptation  
✅ **3D Visualization** - Interactive signal trajectory plots  

---

## 📖 Documentation

- **[MANUAL.pdf](MANUAL.pdf)** - Complete Technical Manual (20 pages, 5MB)
- **[DATASHEET.md](DATASHEET.md)** - Technical Specifications
- **[AUTO_SPEED_README.md](AUTO_SPEED_README.md)** - Speed Adaptation Guide
- **[Online Docs](https://springyworks.github.io/Digiton/)** - GitHub Pages

---

## 🛠️ Quick Start

```bash
# Clone repository
git clone https://github.com/springyworks/Digiton.git
cd Digiton

# Install dependencies
pip install -r requirements.txt

# Run basic modem test
python3 spin_digiton_modem.py

# Test auto-speed adaptation at -50dB
python3 digiton_auto_speed_protocol.py

# Deep mode ping-pong at -60dB
python3 digiton_deep_handshake.py
```

---

## 🎯 Speed Modes

| Mode   | Min SNR | Integration | Baud | Transfer Time* |
|--------|---------|-------------|------|----------------|
| TURBO  | +10 dB  | 1×          | 200  | 0.08s          |
| FAST   | 0 dB    | 1×          | 50   | 0.32s          |
| NORMAL | -10 dB  | 4×          | 20   | 0.80s          |
| SLOW   | -30 dB  | 64×         | 12   | 85s            |
| DEEP   | -50 dB  | 512×        | 1    | 819s           |

\* Time to transfer 2 ASCII characters

---

## 📊 Test Results

**Auto-Speed Protocol at -50dB:**
```
Channel SNR:       -50.0 dB
Measured SNR:      -27.1 dB (compensated)
Selected Mode:     DEEP
Integration:       512×
Detection:         ✓ Success
```

**Deep Mode Ping-Pong at -60dB:**
- 1024× coherent integration
- Matched filter detection
- Right/Left spin discrimination
- **Result:** Signal recovered from noise floor!

---

## 🔬 How It Works

### Spin Digiton Modulation
- **Right Spin** (1700 Hz) = Binary `1`
- **Left Spin** (1300 Hz) = Binary `0`
- **Gaussian Envelope** = Complex Morlet Wavelet

### I/Q Downconversion
```
LO(t) = exp(-j·2π·1500·t)
IQ(t) = signal(t) × LO(t)
```

### Coherent Integration
```
Integration Gain = 10·log₁₀(N) dB
Example: 512× = 27 dB gain
```

---

## 📁 Project Structure

```
Digiton/
├── MANUAL.pdf                      # Technical manual (generated)
├── DATASHEET.md                    # Specifications
├── AUTO_SPEED_README.md           # Speed adaptation guide
├── spin_digiton_modem.py          # Core modem
├── digiton_auto_speed_protocol.py # Auto-speed system
├── digiton_deep_handshake.py      # Deep mode test
├── hf_channel_simulator.py        # Watterson channel
├── data/                          # Generated plots & audio
└── docs/                          # GitHub Pages
```

---

## 🖼️ Visualizations

![Auto-Speed Protocol](data/11_auto_speed_protocol.png)
*Automatic speed adaptation selecting DEEP mode at -50dB SNR*

![Deep Mode Recovery](data/10_deep_ping_pong_test.png)
*Coherent integration recovering ping-pong signals at -60dB*

![3D Corkscrew](data/05_3d_corkscrew.png)
*I/Q signal trajectory showing right and left spin helices*

---

## 🧪 Running Tests

```bash
# Basic modem
python3 spin_digiton_modem.py

# Chat protocol
python3 digiton_chat.py

# Speed adaptation
python3 digiton_auto_speed_protocol.py

# Watterson fading
python3 digiton_spin_watterson.py

# Deep mode
python3 digiton_deep_handshake.py

# 3D visualization
python3 digiton_3d_analyzer.py

# Generate PDF manual
python3 generate_manual_pdf.py
```

---

## 🤝 Contributing

Contributions welcome! This is an educational/experimental project.

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🔗 Links

- **GitHub:** https://github.com/springyworks/Digiton
- **Docs:** https://springyworks.github.io/Digiton/
- **Issues:** https://github.com/springyworks/Digiton/issues

---

**Made with ❤️ by the Digiton Team**
