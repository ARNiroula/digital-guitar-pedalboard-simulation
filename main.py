import sys

import numpy as np
import pyaudio
import pyqtgraph as pg
from PyQt6.QtCore import QThreadPool, QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
    QTabWidget,
    QLabel,
)

# import effects
from ui.tabs import MainTab, SettingsTab, SpectrumTab, PedalboardTab
from audio_signal import AudioIO
from visualizer import SpectrumAnalyzer
from io_manager.synthetic_input import KarplusStrongSynth


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
        self.setWindowTitle("Guitar Pedalboard Simulator")
        self.resize(1000, 750)

        self.sample_rate = 44100
        self.block_len = 1024

        # Karplus-Strong Synthesizer
        self.synth = KarplusStrongSynth(self.sample_rate)

        # Initialize arrays
        self.analyzer = None
        self.input_spectrum = np.zeros(512)
        self.output_spectrum = np.zeros(512)
        self.input_peak = np.full(512, -80.0)
        self.output_peak = np.full(512, -80.0)
        self.latest_input = np.zeros(1024)
        self.latest_output = np.zeros(1024)
        self.smoothing = 0.7
        self.peak_decay = 0.99

        self.threadpool = QThreadPool()
        self.audio_worker = None

        self._setup_ui()
        self._init_analyzer()

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plots)
        self.plot_timer.start(16)

        # Keyboard mapping for playing notes
        self.key_note_map = {
            Qt.Key.Key_A: "E2",
            Qt.Key.Key_S: "A2",
            Qt.Key.Key_D: "D3",
            Qt.Key.Key_F: "G3",
            Qt.Key.Key_G: "B3",
            Qt.Key.Key_H: "E4",
            Qt.Key.Key_J: "A3",
            Qt.Key.Key_K: "C4",
            Qt.Key.Key_Z: "C3",
            Qt.Key.Key_X: "D3",
            Qt.Key.Key_C: "E3",
            Qt.Key.Key_V: "F3",
            Qt.Key.Key_B: "G3",
            Qt.Key.Key_N: "A3",
            Qt.Key.Key_M: "B3",
        }

    def _setup_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        pg.setConfigOptions(antialias=True)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #444444;
                border-radius: 10px;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #333333;
                color: white;
                padding: 10px 25px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: #00ff88;
                color: #1a1a1a;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #444444;
            }
        """)

        # Create tabs
        self.main_tab = MainTab()
        self.spectrum_tab = SpectrumTab()
        self.pedalboard_tab = PedalboardTab(self.sample_rate)
        self.settings_tab = SettingsTab()

        self.settings_tab.device_changed.connect(self._on_settings_changed)
        self.main_tab.note_triggered.connect(self._on_note_triggered)
        self.main_tab.damping_slider.valueChanged.connect(
            lambda v: self.synth.set_damping(v / 1000.0)
        )

        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.spectrum_tab, "Spectrum")
        self.tabs.addTab(self.pedalboard_tab, "Pedalboard")
        self.tabs.addTab(self.settings_tab, "Settings")

        main_layout.addWidget(self.tabs)

        # Control bar at bottom
        control_bar = QFrame()
        control_bar.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 8px;
            }
        """)
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(15, 10, 15, 10)

        self.start_btn = QPushButton("▶ Start Audio")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.toggle_audio)
        self.start_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 30px;
                font-size: 15px;
                font-weight: bold;
                background-color: #333333;
                color: white;
                border: 2px solid #555555;
                border-radius: 6px;
                min-width: 150px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                border-color: #4CAF50;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        control_layout.addWidget(self.start_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888; font-size: 13px;")
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()

        # Input source indicator
        self.source_label = QLabel("Source: Synth")
        self.source_label.setStyleSheet("color: #00bfff; font-size: 12px;")
        control_layout.addWidget(self.source_label)

        control_layout.addSpacing(15)

        self.info_label = QLabel("44100 Hz | 1024 samples")
        self.info_label.setStyleSheet("color: #00ff88; font-size: 12px;")
        control_layout.addWidget(self.info_label)

        main_layout.addWidget(control_bar)

        self.setCentralWidget(central)

    def _init_analyzer(self):
        self.sample_rate = self.settings_tab.get_sample_rate()
        self.block_len = self.settings_tab.get_buffer_size()

        self.analyzer = SpectrumAnalyzer(self.sample_rate, self.block_len)

        num_freqs = len(self.analyzer.freqs)
        self.input_spectrum = np.zeros(num_freqs)
        self.output_spectrum = np.zeros(num_freqs)
        self.input_peak = np.full(num_freqs, -80.0)
        self.output_peak = np.full(num_freqs, -80.0)

        self.latest_input = np.zeros(self.block_len)
        self.latest_output = np.zeros(self.block_len)

        # Update synth and pedalboard
        self.synth.set_sample_rate(self.sample_rate)
        self.pedalboard_tab.update_sample_rate(self.sample_rate)

        self.info_label.setText(f"{self.sample_rate} Hz | {self.block_len} samples")

        # Update source label
        source = "Synth" if self.settings_tab.use_synth() else "Mic"
        output = "Output ON" if self.settings_tab.output_enabled() else "Output OFF"
        self.source_label.setText(f"Source: {source} | {output}")

    def _on_settings_changed(self):
        was_running = self.audio_worker is not None

        if was_running:
            self.toggle_audio(False)
            self.start_btn.setChecked(False)

        self._init_analyzer()

    def _on_note_triggered(self, note: str):
        """Handle note trigger from virtual keyboard"""
        self.synth.pluck_note(note, 0.8)

    def toggle_audio(self, checked):
        if checked:
            self.start_btn.setText("■ Stop Audio")
            self.status_label.setText("Running...")
            self.status_label.setStyleSheet("color: #00ff88; font-size: 13px;")

            use_synth = self.settings_tab.use_synth()
            enable_output = self.settings_tab.output_enabled()

            input_device = self.settings_tab.get_input_device_index()
            output_device = self.settings_tab.get_output_device_index()
            sample_rate = self.settings_tab.get_sample_rate()
            buffer_size = self.settings_tab.get_buffer_size()

            self.audio_worker = AudioIO(
                block_len=buffer_size,
                channels=1,
                rate=sample_rate,
                input_device_index=input_device,
                output_device_index=output_device,
                use_synth=use_synth,
                synth=self.synth,
                enable_output=enable_output,
            )

            self.audio_worker.effects = self.pedalboard_tab.get_effects_chain()

            self.audio_worker.signals.audio_data.connect(self._on_audio_data)
            self.audio_worker.signals.error.connect(self._on_audio_error)
            self.threadpool.start(self.audio_worker)
        else:
            self.start_btn.setText("▶ Start Audio")
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("color: #888888; font-size: 13px;")
            if self.audio_worker:
                self.audio_worker.stop()
                self.audio_worker = None

    def _on_audio_data(self, audio_in: np.ndarray, audio_out: np.ndarray):
        self.latest_input = audio_in
        self.latest_output = audio_out

        self.main_tab.input_meter.set_level(np.max(np.abs(audio_in)))
        self.main_tab.output_meter.set_level(np.max(np.abs(audio_out)))

        if self.analyzer is None:
            return

        _, in_spectrum = self.analyzer.compute(audio_in)
        _, out_spectrum = self.analyzer.compute(audio_out)

        if len(in_spectrum) != len(self.input_spectrum):
            return

        self.input_spectrum = (
            self.smoothing * self.input_spectrum + (1 - self.smoothing) * in_spectrum
        )
        self.output_spectrum = (
            self.smoothing * self.output_spectrum + (1 - self.smoothing) * out_spectrum
        )

        self.input_peak = np.maximum(
            self.input_peak * self.peak_decay, self.input_spectrum
        )
        self.output_peak = np.maximum(
            self.output_peak * self.peak_decay, self.output_spectrum
        )

    def _on_audio_error(self, error_msg: str):
        print(f"Audio error: {error_msg}")
        self.start_btn.setChecked(False)
        self.start_btn.setText("▶ Start Audio")
        self.status_label.setText(f"Error: {error_msg[:30]}...")
        self.status_label.setStyleSheet("color: #ff4444; font-size: 13px;")

    def _update_plots(self):
        if self.analyzer is None:
            return

        if self.input_spectrum is None or self.output_spectrum is None:
            return

        if self.input_peak is None or self.output_peak is None:
            return

        try:
            freqs = self.analyzer.freqs

            if len(freqs) <= 1:
                return

            self.spectrum_tab.input_curve.setData(freqs[1:], self.input_spectrum[1:])
            self.spectrum_tab.input_peak_curve.setData(freqs[1:], self.input_peak[1:])
            self.spectrum_tab.output_curve.setData(freqs[1:], self.output_spectrum[1:])
            self.spectrum_tab.output_peak_curve.setData(freqs[1:], self.output_peak[1:])

            self.main_tab.input_wave_curve.setData(self.latest_input)
            self.main_tab.output_wave_curve.setData(self.latest_output)

        except (TypeError, IndexError, ValueError):
            pass

    def keyPressEvent(self, event):
        """Handle keyboard input for playing notes"""
        if event.isAutoRepeat():
            return

        key = event.key()
        if key in self.key_note_map:
            self.synth.pluck_note(self.key_note_map[key], 0.8)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.plot_timer.stop()
        if self.audio_worker:
            self.audio_worker.stop()
        self.threadpool.waitForDone(1000)
        self.settings_tab.device_manager.terminate()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark theme for pyqtgraph
    pg.setConfigOption("background", "#2b2b2b")
    pg.setConfigOption("foreground", "#ffffff")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
