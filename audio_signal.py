import numpy as np
import pyaudio
from PyQt6.QtCore import QObject, QRunnable, pyqtSlot, pyqtSignal


# Signals need to be in a QObject
class AudioSignals(QObject):
    """Signals for thread-safe communication from audio worker"""

    audio_data = pyqtSignal(np.ndarray, np.ndarray)  # (input, output)
    error = pyqtSignal(str)


class AudioIO(QRunnable):
    def __init__(
        self,
        block_len: int = 1024,
        channels: int = 1,
        rate: int = 44100,
        pa_format: int = pyaudio.paInt16,
    ):
        super().__init__()
        self.block_len = block_len
        self.channels = channels
        self.rate = rate
        self.pa_format = pa_format
        self.is_running = True
        self.signals = AudioSignals()

        # Effects chain (list of callables)
        self.effects = []

        self._stream = None
        self._p = None

    @pyqtSlot()
    def run(self):
        try:
            self._p = pyaudio.PyAudio()
            self._stream = self._p.open(
                format=self.pa_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                output=True,
                frames_per_buffer=self.block_len,
            )

            while self.is_running:
                input_bytes = self._stream.read(
                    self.block_len, exception_on_overflow=False
                )
                audio_in = np.frombuffer(input_bytes, dtype=np.int16).astype(np.float32)

                # Normalize to -1.0 to 1.0
                audio_in = audio_in / 32768.0

                # Store input for visualization
                input_copy = audio_in.copy()

                # Apply effects chain
                audio_out = audio_in.copy()
                for effect in self.effects:
                    audio_out = effect(audio_out)

                # Emit both input and output for visualization
                self.signals.audio_data.emit(input_copy, audio_out.copy())

                # Convert back to int16 and output
                output = np.clip(audio_out * 32768.0, -32768, 32767).astype(np.int16)
                self._stream.write(output.tobytes())

        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self._cleanup()

    def _cleanup(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._p:
            self._p.terminate()

    @pyqtSlot()
    def stop(self):
        self.is_running = False
