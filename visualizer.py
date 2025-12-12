import numpy as np


class SpectrumAnalyzer:
    """Compute FFT spectrum from audio data"""

    def __init__(self, sample_rate: int, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        # Pre-compute frequency bins (only positive frequencies)
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        # Window function to reduce spectral leakage
        self.window = np.hanning(fft_size)

    def compute(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (frequencies, magnitudes_db)"""
        # Pad or truncate to fft_size
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        else:
            audio = audio[: self.fft_size]

        # Apply window and compute FFT
        windowed = audio * self.window
        fft = np.fft.rfft(windowed)

        # Convert to magnitude in dB
        magnitude = np.abs(fft)
        # Avoid log(0) by adding small epsilon
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        return self.freqs, magnitude_db
