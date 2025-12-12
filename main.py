import sys

import pyaudio
from PyQt6.QtCore import QSize, QThreadPool
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

from audio_signal import AudioIO

# PyAudio Value
# TODO: make the below constant value changable using the GUI input
BLOCKLEN = 64  # Number of frames per block
WIDTH = 2  # Bytes per sample
CHANNELS = 1  # Mono
RATE = 8000  # Frames per second

p = pyaudio.PyAudio()
PA_FORMAT = pyaudio.paInt16
# stream = p.open(
#     format=PA_FORMAT,
#     channels=CHANNELS,
#     rate=RATE,
#     input=False,
#     output=True,
#     frames_per_buffer=BLOCKLEN,
# )
# specify low frames_per_buffer to reduce latency


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Pedalboard Simulation")

        # TODO: Add device

        # Input Specification
        self.block_len = BLOCKLEN
        self.width = WIDTH
        self.channels = CHANNELS
        self.rate = RATE
        self.pa_format = PA_FORMAT

        # Buttons
        button = QPushButton("Press Me!")
        button.setCheckable(True)
        button.clicked.connect(self.audio_record)
        button.clicked.connect(self.test_func)

        # IO ThreadPool
        self.threadpool = QThreadPool()

        # Start the audio IO
        self.audio_io_worker = AudioIO(
            p=p,
            block_len=self.block_len,
            width=self.width,
            channels=self.channels,
            rate=self.rate,
            pa_format=self.pa_format,
        )
        self.threadpool.start(self.audio_io_worker)

        self.setFixedSize(QSize(680, 480))

        self.setCentralWidget(button)

    def audio_record(self):
        print("You Clicked Me!!")

    def test_func(self, checked):
        print(f"You've already clicked me!! {checked}")


print("Starting Core Application")
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
print("Closing Application. GoodBye!!")
