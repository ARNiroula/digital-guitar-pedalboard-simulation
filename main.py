import sys

import numpy as np
import pyaudio
import pyqtgraph as pg
from PyQt6.QtCore import QThreadPool, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
)

# import effects
from audio_signal import AudioIO
from visualizer import SpectrumAnalyzer
import ui


# PyAudio Value
# TODO: make the below constant value changable using the GUI input
BLOCKLEN = 64  # Number of frames per block
WIDTH = 2  # Bytes per sample
CHANNELS = 1  # Mono
RATE = 44100  # Frames per second

p = pyaudio.PyAudio()
PA_FORMAT = pyaudio.paInt16


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Pedalboard Simulation")
        self.resize(1200, 800)

        # Audio parameters
        self.sample_rate = 44100
        self.block_len = 1024

        # Spectrum analyzer
        self.analyzer = SpectrumAnalyzer(self.sample_rate, self.block_len)

        # Audio data buffers
        self.input_spectrum = np.zeros(len(self.analyzer.freqs))
        self.output_spectrum = np.zeros(len(self.analyzer.freqs))
        self.smoothing = 0.7

        # Create pedals
        self.distortion = ui.pedal.DistortionPedal()
        self.delay = ui.pedal.DelayPedal(self.sample_rate)

        self._setup_ui()

        # Audio setup
        self.threadpool = QThreadPool()
        self.audio_worker = None

        # Update timer
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plots)
        self.plot_timer.start(16)

    def _setup_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        pg.setConfigOptions(antialias=True)

        # Spectrum plots
        spectrum_layout = QHBoxLayout()

        self.input_plot = pg.PlotWidget(title="Input Spectrum")
        self.input_plot.setLabel("left", "Magnitude", units="dB")
        self.input_plot.setLabel("bottom", "Frequency", units="Hz")
        self.input_plot.setYRange(-80, 0)
        self.input_plot.setXRange(20, 20000)
        self.input_plot.setLogMode(x=True, y=False)
        self.input_curve = self.input_plot.plot(pen=pg.mkPen("cyan", width=1))
        spectrum_layout.addWidget(self.input_plot)

        self.output_plot = pg.PlotWidget(title="Output Spectrum")
        self.output_plot.setLabel("left", "Magnitude", units="dB")
        self.output_plot.setLabel("bottom", "Frequency", units="Hz")
        self.output_plot.setYRange(-80, 0)
        self.output_plot.setXRange(20, 20000)
        self.output_plot.setLogMode(x=True, y=False)
        self.output_curve = self.output_plot.plot(pen=pg.mkPen("lime", width=1))
        spectrum_layout.addWidget(self.output_plot)

        main_layout.addLayout(spectrum_layout)

        # Waveform plot
        self.waveform_plot = pg.PlotWidget(title="Waveform")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setLabel("bottom", "Samples")
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.addLegend()
        self.input_wave_curve = self.waveform_plot.plot(
            pen=pg.mkPen("cyan", width=1), name="Input"
        )
        self.output_wave_curve = self.waveform_plot.plot(
            pen=pg.mkPen("lime", width=1), name="Output"
        )
        main_layout.addWidget(self.waveform_plot)

        # Pedalboard section
        pedal_frame = QFrame()
        pedal_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        pedal_layout = QHBoxLayout(pedal_frame)
        pedal_layout.setSpacing(20)

        # Add pedals
        pedal_layout.addWidget(self.distortion)
        pedal_layout.addWidget(self.delay)
        pedal_layout.addStretch()

        main_layout.addWidget(pedal_frame)

        # Control buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Audio")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.toggle_audio)
        self.start_btn.setStyleSheet("""
            QPushButton {
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                background-color: #333333;
                color: white;
                border: 2px solid #555555;
                border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        button_layout.addWidget(self.start_btn)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)
        self.setCentralWidget(central)

        # Store latest audio data
        self.latest_input = np.zeros(self.block_len)
        self.latest_output = np.zeros(self.block_len)

    def toggle_audio(self, checked):
        if checked:
            self.start_btn.setText("Stop Audio")
            self.audio_worker = AudioIO(
                block_len=self.block_len,
                channels=1,
                rate=self.sample_rate,
            )
            # Add pedals to effects chain
            self.audio_worker.effects = [self.distortion, self.delay]

            self.audio_worker.signals.audio_data.connect(self._on_audio_data)
            self.audio_worker.signals.error.connect(self._on_audio_error)
            self.threadpool.start(self.audio_worker)
        else:
            self.start_btn.setText("Start Audio")
            if self.audio_worker:
                self.audio_worker.stop()
                self.audio_worker = None

    def _on_audio_data(self, audio_in: np.ndarray, audio_out: np.ndarray):
        self.latest_input = audio_in
        self.latest_output = audio_out

        _, in_spectrum = self.analyzer.compute(audio_in)
        _, out_spectrum = self.analyzer.compute(audio_out)

        self.input_spectrum = (
            self.smoothing * self.input_spectrum + (1 - self.smoothing) * in_spectrum
        )
        self.output_spectrum = (
            self.smoothing * self.output_spectrum + (1 - self.smoothing) * out_spectrum
        )

    def _on_audio_error(self, error_msg: str):
        print(f"Audio error: {error_msg}")
        self.start_btn.setChecked(False)
        self.start_btn.setText("Start Audio")

    def _update_plots(self):
        freqs = self.analyzer.freqs
        self.input_curve.setData(freqs[1:], self.input_spectrum[1:])
        self.output_curve.setData(freqs[1:], self.output_spectrum[1:])
        self.input_wave_curve.setData(self.latest_input)
        self.output_wave_curve.setData(self.latest_output)

    def closeEvent(self, event):
        self.plot_timer.stop()
        if self.audio_worker:
            self.audio_worker.stop()
        self.threadpool.waitForDone(1000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark theme for pyqtgraph
    pg.setConfigOption("background", "#2b2b2b")
    pg.setConfigOption("foreground", "#ffffff")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
