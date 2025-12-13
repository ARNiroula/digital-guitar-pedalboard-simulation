import traceback

import numpy as np
import pyaudio
from PyQt6.QtCore import QObject, QRunnable, pyqtSlot, pyqtSignal

from io_manager.synthetic_input import KarplusStrongSynth


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
        use_synth: bool = False,
        synth: "KarplusStrongSynth" = None,
        enable_output: bool = True,
    ):
        super().__init__()
        self.block_len = block_len
        self.channels = channels
        self.rate = rate
        self.pa_format = pa_format
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.use_synth = use_synth
        self.synth = synth
        self.enable_output = enable_output
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

            # Only open input stream if not using synth
            if not self.use_synth:
                self._stream_in = self._p.open(
                    format=self.pa_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    output=False,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.block_len,
                )

            # Only open output stream if output is enabled
            if self.enable_output:
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
                try:
                    # Get input audio
                    if self.use_synth and self.synth:
                        # Generate synthetic audio
                        audio_in = self.synth.generate(self.block_len)
                    else:
                        # Read from microphone
                        input_bytes = self._stream_in.read(
                            self.block_len, exception_on_overflow=False
                        )
                        audio_in = np.frombuffer(input_bytes, dtype=np.int16).astype(
                            np.float32
                        )
                        audio_in = audio_in / 32768.0

                    # Ensure correct size
                    audio_in = self._ensure_block_size(audio_in)

                    input_copy = audio_in.copy()
                    audio_out = audio_in.copy()

                    # Apply effects with size checking
                    for effect in self.effects:
                        try:
                            if callable(effect):
                                audio_out = effect(audio_out)
                            elif hasattr(effect, "process"):
                                audio_out = effect.process(audio_out)

                            # Ensure size remains consistent after each effect
                            audio_out = self._ensure_block_size(audio_out)
                        except Exception as e:
                            # If an effect fails, pass through unchanged
                            print(f"Effect error: {e}")
                            continue

                    # Emit for visualization
                    self.signals.audio_data.emit(input_copy, audio_out.copy())

                    # Output audio (if enabled)
                    if self.enable_output and self._stream_out:
                        output = np.clip(audio_out * 32768.0, -32768, 32767).astype(
                            np.int16
                        )
                        self._stream_out.write(output.tobytes())

                except Exception as e:
                    # Don't crash on individual frame errors
                    print(f"Frame error: {e}")
                    print(traceback.format_exc())
                    break
                    continue

        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self._cleanup()

    def _ensure_block_size(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio array is exactly block_len samples"""
        if audio is None:
            return np.zeros(self.block_len, dtype=np.float32)

        if len(audio) == self.block_len:
            return audio

        # Create correctly sized array
        result = np.zeros(self.block_len, dtype=np.float32)

        if len(audio) > self.block_len:
            # Truncate
            result[:] = audio[: self.block_len]
        else:
            # Pad with zeros
            result[: len(audio)] = audio

        return result

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
