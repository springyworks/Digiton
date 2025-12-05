# Digiton

Spin Digiton Modem & Wavelet Party Protocol

Digiton is a Python-based software modem and protocol suite designed for robust communication over HF channels using Morlet wavelets and "Spin" modulation.

## Documentation

Project documentation is available on GitHub Pages:
[https://springyworks.github.io/Digiton/](https://springyworks.github.io/Digiton/)

*   [**Manual**](MANUAL.md): User guide and technical details.
*   [**Real World Testing**](REAL_WORLD_TESTING.md): Guide for WebSDR echo tests.
*   [**Protocol Access Schemes**](docs/PROTOCOL_ACCESS_SCHEMES.md): Detailed explanation of TDMA (Slots) and FDMA (Spin/Banks).
*   [**Architecture Decision**](docs/WHY_MORLET_VS_CHIRP.md): Why we use Morlet Wavelets instead of Chirps (Break-in, Doppler, etc.).

## Project Structure

- `digiton/`: Core library containing the modem, protocol, and channel simulation logic.
- `apps/`: Executable applications (e.g., `listener.py`, `visualizer.py`).
- `tests/`: Comprehensive test suite (regression and experiments).
- `docs/`: Source files for the documentation site.
- `legacy/`: Archived scripts and data from previous versions.

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
python3 run_tests.py
```
