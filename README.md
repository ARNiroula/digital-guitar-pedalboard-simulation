# digital-guitar-pedalboard-simulation
Digital Guitar Pedalboard Simulation.

A barebone simulation of Digital Pedal Board. The raw input comes from either:
- Guitar/Audio input through an audio interface
- Synthetic string input using karplus-strong string synthesis

## Goals
The input will then be sent through a chain of different types of popular audio effects such as delay, flanger, overdrive, etc. 
Visualization of input/output and their spectrum graph will also be available.


## Milestones
- [x] Real time audio input using raw audio and synthetic audio using karplus-strong algorithm 
    - [x] The audio should run in different thread so that GUI won't be affected
- [x] Basic Effects (distortion, compressor)
- [x] Effect chain (combine multiple effects)
- [x] Basic Input/Output audio graph
- [x] Basic Spectrum graph
    - [x] Show frequency spectrum for input and output signal
- [] Flexible Spectrum graph
    - [] Option to modify showing log/plain graph
    - [] Option to show either decibel or simple magnitude calculated through FFT
- [x] Keyboard as Keyboard shortcuts to use karplus-strong synth for string sound
- [] Effects
    - [x] Compressor
    - [x] EQ
    - [x] Overdrive
    - [x] Distortion
    - [x] Chorus
    - [x] Flanger
    - [x] Delay
    - [x] Reverb
    - [] Wah
    - [] Phaser
- [x] Settings to control the Input Source, Audio Devices, Audio Settings (sampling rate, buffer size)
... and other Milestones that'll be added later!

## To run this project
### Prerequisites
- Python 3.12+
- PyAudio Prerequisites: See [PyAudio installation notes](https://pypi.org/project/PyAudio/#:~:text=PyPi-,Installation,-See%20the%20INSTALLATION)

### Running the project
- Clone the repository
- Create a virtual environment
```bash
# Create a virtual environment in the directory
python -m venv ./venv
# For linux/macOS
source ./venv/bin/activate

# For windows
source ./venv/Scripts/activate
```
- Install the required packages
```bash
pip install -r ./requirements.txt
```
- Run the `main.py`
```bash
python main.py
```
