import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QGridLayout,
    QComboBox,
    QGroupBox,
    QScrollArea,
)
import pyqtgraph as pg

from io_manager import AudioDeviceManager
from .meter import VUMeter
from .pedal import (
    CompressorPedal,
    EQPedal,
    DistortionPedal,
    DelayPedal,
    ChorusPedal,
    ReverbPedal,
)


class SettingsTab(QWidget):
    """Settings tab with audio device configuration"""

    device_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device_manager = AudioDeviceManager()

        # Initialize attributes first
        self.input_combo = None
        self.output_combo = None
        self.sample_rate_combo = None
        self.buffer_combo = None
        self.latency_label = None

        self._setup_ui()
        self._populate_devices()
        self._update_latency_display()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        combo_style = """
            QComboBox {
                background-color: #3a3a3a;
                color: white;
                border: 2px solid #555555;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                min-width: 300px;
            }
            QComboBox:hover {
                border-color: #00ff88;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 8px solid #00ff88;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: white;
                selection-background-color: #00ff88;
                selection-color: black;
                border: 1px solid #555555;
            }
        """

        label_style = "color: white; font-size: 14px; font-weight: bold;"

        group_style = """
            QGroupBox {
                color: #00ff88;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
            }
        """

        # Audio Devices Group
        devices_group = QGroupBox("Audio Devices")
        devices_group.setStyleSheet(group_style)
        devices_layout = QGridLayout(devices_group)
        devices_layout.setSpacing(15)
        devices_layout.setContentsMargins(20, 30, 20, 20)

        # Input device
        input_label = QLabel("Input Device:")
        input_label.setStyleSheet(label_style)
        devices_layout.addWidget(input_label, 0, 0)

        self.input_combo = QComboBox()
        self.input_combo.setStyleSheet(combo_style)
        devices_layout.addWidget(self.input_combo, 0, 1)

        # Output device
        output_label = QLabel("Output Device:")
        output_label.setStyleSheet(label_style)
        devices_layout.addWidget(output_label, 1, 0)

        self.output_combo = QComboBox()
        self.output_combo.setStyleSheet(combo_style)
        devices_layout.addWidget(self.output_combo, 1, 1)

        # Refresh button
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self._populate_devices)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 2px solid #555555;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #00ff88;
            }
        """)
        devices_layout.addWidget(refresh_btn, 0, 2, 2, 1)

        layout.addWidget(devices_group)

        # Audio Settings Group
        settings_group = QGroupBox("Audio Settings")
        settings_group.setStyleSheet(group_style)
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(15)
        settings_layout.setContentsMargins(20, 30, 20, 20)

        # Sample rate
        sample_rate_label = QLabel("Sample Rate:")
        sample_rate_label.setStyleSheet(label_style)
        settings_layout.addWidget(sample_rate_label, 0, 0)

        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.setStyleSheet(combo_style)
        for rate, name in [
            (22050, "22050 Hz"),
            (44100, "44100 Hz (CD Quality)"),
            (48000, "48000 Hz (Studio)"),
            (96000, "96000 Hz (High Res)"),
        ]:
            self.sample_rate_combo.addItem(name, rate)
        self.sample_rate_combo.setCurrentIndex(1)
        settings_layout.addWidget(self.sample_rate_combo, 0, 1)

        # Buffer size
        buffer_label = QLabel("Buffer Size:")
        buffer_label.setStyleSheet(label_style)
        settings_layout.addWidget(buffer_label, 1, 0)

        self.buffer_combo = QComboBox()
        self.buffer_combo.setStyleSheet(combo_style)
        for size, name in [
            (128, "128 samples (Low Latency)"),
            (256, "256 samples"),
            (512, "512 samples"),
            (1024, "1024 samples (Stable)"),
            (2048, "2048 samples (High Latency)"),
        ]:
            self.buffer_combo.addItem(name, size)
        self.buffer_combo.setCurrentIndex(3)
        settings_layout.addWidget(self.buffer_combo, 1, 1)

        # Latency display
        self.latency_label = QLabel("Estimated Latency: ~23ms")
        self.latency_label.setStyleSheet("color: #888888; font-size: 13px;")
        settings_layout.addWidget(self.latency_label, 2, 0, 1, 2)

        layout.addWidget(settings_group)
        layout.addStretch()

        # Connect signals AFTER all widgets are created
        self.input_combo.currentIndexChanged.connect(self._on_device_changed)
        self.output_combo.currentIndexChanged.connect(self._on_device_changed)
        self.sample_rate_combo.currentIndexChanged.connect(self._on_device_changed)
        self.buffer_combo.currentIndexChanged.connect(self._on_device_changed)

    def _populate_devices(self):
        current_input = self.get_input_device_index()
        current_output = self.get_output_device_index()

        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)

        self.input_combo.clear()
        self.output_combo.clear()

        input_devices = self.device_manager.get_input_devices()
        default_input = self.device_manager.get_default_input_device()
        default_input_idx = 0

        for i, (device_idx, name) in enumerate(input_devices):
            self.input_combo.addItem(name, device_idx)
            if device_idx == default_input:
                default_input_idx = i

        output_devices = self.device_manager.get_output_devices()
        default_output = self.device_manager.get_default_output_device()
        default_output_idx = 0

        for i, (device_idx, name) in enumerate(output_devices):
            self.output_combo.addItem(name, device_idx)
            if device_idx == default_output:
                default_output_idx = i

        if current_input >= 0:
            idx = self.input_combo.findData(current_input)
            self.input_combo.setCurrentIndex(idx if idx >= 0 else default_input_idx)
        else:
            self.input_combo.setCurrentIndex(default_input_idx)

        if current_output >= 0:
            idx = self.output_combo.findData(current_output)
            self.output_combo.setCurrentIndex(idx if idx >= 0 else default_output_idx)
        else:
            self.output_combo.setCurrentIndex(default_output_idx)

        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)

    def _on_device_changed(self):
        self._update_latency_display()
        self.device_changed.emit()

    def _update_latency_display(self):
        # Guard against being called before widgets exist
        if (
            self.buffer_combo is None
            or self.sample_rate_combo is None
            or self.latency_label is None
        ):
            return

        sample_rate = self.get_sample_rate()
        buffer_size = self.get_buffer_size()
        if sample_rate > 0:
            latency_ms = (buffer_size / sample_rate) * 1000 * 2
            self.latency_label.setText(f"Estimated Latency: ~{latency_ms:.1f}ms")

    def get_input_device_index(self) -> int:
        if self.input_combo is None or self.input_combo.currentIndex() < 0:
            return -1
        return self.input_combo.currentData()

    def get_output_device_index(self) -> int:
        if self.output_combo is None or self.output_combo.currentIndex() < 0:
            return -1
        return self.output_combo.currentData()

    def get_sample_rate(self) -> int:
        if self.sample_rate_combo is None or self.sample_rate_combo.currentIndex() < 0:
            return 44100
        return self.sample_rate_combo.currentData()

    def get_buffer_size(self) -> int:
        if self.buffer_combo is None or self.buffer_combo.currentIndex() < 0:
            return 1024
        return self.buffer_combo.currentData()


class MainTab(QWidget):
    """Main tab with VU meters, waveform, and controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # VU Meters section
        meters_group = QGroupBox("Levels")
        meters_group.setStyleSheet("""
            QGroupBox {
                color: #00ff88;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
            }
        """)
        meters_layout = QHBoxLayout(meters_group)
        meters_layout.setContentsMargins(20, 25, 20, 15)

        self.input_meter = VUMeter("INPUT")
        self.output_meter = VUMeter("OUTPUT")

        meters_layout.addStretch()
        meters_layout.addWidget(self.input_meter)
        meters_layout.addSpacing(30)
        meters_layout.addWidget(self.output_meter)
        meters_layout.addStretch()

        layout.addWidget(meters_group)

        # Waveform section
        waveform_group = QGroupBox("Waveform")
        waveform_group.setStyleSheet(meters_group.styleSheet())
        waveform_layout = QVBoxLayout(waveform_group)
        waveform_layout.setContentsMargins(15, 25, 15, 15)

        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setLabel("bottom", "Samples")
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.addLegend(offset=(60, 10))
        self.waveform_plot.setMinimumHeight(200)
        self.input_wave_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#00bfff", width=1), name="Input"
        )
        self.output_wave_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#00ff88", width=1), name="Output"
        )

        waveform_layout.addWidget(self.waveform_plot)
        layout.addWidget(waveform_group)


class SpectrumTab(QWidget):
    """Spectrum analyzer tab"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        group_style = """
            QGroupBox {
                color: #00ff88;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
            }
        """

        # Input spectrum
        input_group = QGroupBox("Input Spectrum")
        input_group.setStyleSheet(group_style)
        input_layout = QVBoxLayout(input_group)
        input_layout.setContentsMargins(15, 25, 15, 15)

        self.input_plot = pg.PlotWidget()
        self.input_plot.setLabel("left", "Magnitude (dB)")
        self.input_plot.setLabel("bottom", "Frequency (Hz)")
        self.input_plot.setYRange(-80, 0)
        self.input_plot.setXRange(20, 20000)
        self.input_plot.setLogMode(x=True, y=False)
        self.input_curve = self.input_plot.plot(pen=pg.mkPen("#00bfff", width=1.5))
        self.input_peak_curve = self.input_plot.plot(
            pen=pg.mkPen("#00bfff", width=1, style=Qt.PenStyle.DotLine)
        )

        input_layout.addWidget(self.input_plot)
        layout.addWidget(input_group)

        # Output spectrum
        output_group = QGroupBox("Output Spectrum")
        output_group.setStyleSheet(group_style)
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(15, 25, 15, 15)

        self.output_plot = pg.PlotWidget()
        self.output_plot.setLabel("left", "Magnitude (dB)")
        self.output_plot.setLabel("bottom", "Frequency (Hz)")
        self.output_plot.setYRange(-80, 0)
        self.output_plot.setXRange(20, 20000)
        self.output_plot.setLogMode(x=True, y=False)
        self.output_curve = self.output_plot.plot(pen=pg.mkPen("#00ff88", width=1.5))
        self.output_peak_curve = self.output_plot.plot(
            pen=pg.mkPen("#00ff88", width=1, style=Qt.PenStyle.DotLine)
        )

        output_layout.addWidget(self.output_plot)
        layout.addWidget(output_group)


class PedalboardTab(QWidget):
    """Pedalboard tab with all effects"""

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Signal chain label
        chain_label = QLabel(
            "Signal Chain: Input → Compressor → EQ → Distortion → Chorus → Delay → Reverb → Output"
        )
        chain_label.setStyleSheet("color: #888888; font-size: 12px; padding: 5px;")
        chain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(chain_label)

        # Pedals in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 10px;
            }
        """)

        pedal_container = QWidget()
        pedal_layout = QGridLayout(pedal_container)
        pedal_layout.setSpacing(20)
        pedal_layout.setContentsMargins(20, 20, 20, 20)

        # Create pedals
        self.compressor = CompressorPedal(self.sample_rate)
        self.eq = EQPedal(self.sample_rate)
        self.distortion = DistortionPedal()
        self.chorus = ChorusPedal(self.sample_rate)
        self.delay = DelayPedal(self.sample_rate)
        self.reverb = ReverbPedal(self.sample_rate)

        # Arrange in grid (2 rows x 3 columns)
        pedal_layout.addWidget(self.compressor, 0, 0)
        pedal_layout.addWidget(self.eq, 0, 1)
        pedal_layout.addWidget(self.distortion, 0, 2)
        pedal_layout.addWidget(self.chorus, 1, 0)
        pedal_layout.addWidget(self.delay, 1, 1)
        pedal_layout.addWidget(self.reverb, 1, 2)

        scroll.setWidget(pedal_container)
        layout.addWidget(scroll)

        # Bypass all button
        self.bypass_all_btn = QPushButton("Bypass All Effects")
        self.bypass_all_btn.setCheckable(True)
        self.bypass_all_btn.clicked.connect(self._toggle_bypass_all)
        self.bypass_all_btn.setStyleSheet("""
            QPushButton {
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
                background-color: #333333;
                color: white;
                border: 2px solid #555555;
                border-radius: 6px;
            }
            QPushButton:checked {
                background-color: #ff6600;
                border-color: #ff6600;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        layout.addWidget(self.bypass_all_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _toggle_bypass_all(self, checked):
        pedals = [
            self.compressor,
            self.eq,
            self.distortion,
            self.chorus,
            self.delay,
            self.reverb,
        ]
        for pedal in pedals:
            pedal.enabled = not checked
            pedal.bypass_btn.setChecked(not checked)
            pedal.bypass_btn.setText("ON" if not checked else "OFF")

        self.bypass_all_btn.setText(
            "Enable All Effects" if checked else "Bypass All Effects"
        )

    def get_effects_chain(self) -> list:
        return [
            self.compressor,
            self.eq,
            self.distortion,
            self.chorus,
            self.delay,
            self.reverb,
        ]

    def update_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate
        for pedal in [self.compressor, self.eq, self.chorus, self.delay, self.reverb]:
            pedal.sample_rate = sample_rate

        # Reset delay buffers
        self.delay.max_delay_samples = sample_rate
        self.delay.buffer = np.zeros(sample_rate)
        self.delay.write_idx = 0
