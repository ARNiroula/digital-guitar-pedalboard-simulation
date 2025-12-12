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
        input_device_index: int = None,
        output_device_index: int = None,
    ):
        super().__init__()
        self.block_len = block_len
        self.channels = channels
        self.rate = rate
        self.pa_format = pa_format
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.is_running = True
        self.signals = AudioSignals()
        self.effects = []
        self._stream_in = None
        self._stream_out = None
        self._p = None

    @pyqtSlot()
    def run(self):
        try:
            self._p = pyaudio.PyAudio()

            # Open separate input and output streams for different devices
            self._stream_in = self._p.open(
                format=self.pa_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                output=False,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.block_len,
            )

            self._stream_out = self._p.open(
                format=self.pa_format,
                channels=self.channels,
                rate=self.rate,
                input=False,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.block_len,
            )

            while self.is_running:
                # Read from input device
                input_bytes = self._stream_in.read(
                    self.block_len, exception_on_overflow=False
                )
                audio_in = np.frombuffer(input_bytes, dtype=np.int16).astype(np.float32)
                audio_in = audio_in / 32768.0

                input_copy = audio_in.copy()
                audio_out = audio_in.copy()

                # Process through effects chain
                for effect in self.effects:
                    if callable(effect):
                        audio_out = effect(audio_out)
                    elif hasattr(effect, "process"):
                        audio_out = effect.process(audio_out)

                self.signals.audio_data.emit(input_copy, audio_out.copy())

                # Write to output device
                output = np.clip(audio_out * 32768.0, -32768, 32767).astype(np.int16)
                self._stream_out.write(output.tobytes())

        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self._cleanup()

    def _cleanup(self):
        if self._stream_in:
            self._stream_in.stop_stream()
            self._stream_in.close()
        if self._stream_out:
            self._stream_out.stop_stream()
            self._stream_out.close()
        if self._p:
            self._p.terminate()

    @pyqtSlot()
    def stop(self):
        self.is_running = False
