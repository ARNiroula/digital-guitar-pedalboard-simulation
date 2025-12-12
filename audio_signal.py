import numpy as np
from pyaudio import PyAudio
from PyQt6.QtCore import QRunnable, pyqtSlot


class AudioIO(QRunnable):
    """
    Worker thread to get audio io

    :params :
    """

    def __init__(
        self,
        p: PyAudio,
        block_len: int,
        width: int,
        channels: int,
        rate: int,
        pa_format: int,
        input: bool = True,
        output: bool = False,
    ):
        super().__init__()

        # PyAudio instance
        self.p = PyAudio()

        self.pa_format = pa_format
        self.channels = channels
        self.rate = rate
        self.input = True
        self.output = True
        self.frame_per_buffer = block_len

        self.is_running = True

    @pyqtSlot()
    def run(self):
        """
        Continuously get the audio input signal
        """
        print(self.rate)
        self.stream = self.p.open(
            format=self.pa_format,
            channels=self.channels,
            rate=self.rate,
            input=self.input,
            output=self.output,
            frames_per_buffer=self.frame_per_buffer,
        )

        # TODO: Need to close is_running flag
        while self.is_running:
            # Getting the Audio
            input_bytes = self.stream.read(
                self.frame_per_buffer, exception_on_overflow=False
            )
            x = np.frombuffer(input_bytes, dtype="int16")
            output: np.ndarray = x
            self.stream.write(output.tobytes(), self.frame_per_buffer)

    @pyqtSlot()
    def terminate(self):
        self.is_running = False
        self.stream.stop_stream()
        self.stream.close()
