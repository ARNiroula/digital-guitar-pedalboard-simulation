import pyaudio
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QPushButton,
    QLabel,
    QGridLayout,
    QComboBox,
    QGroupBox,
)


class AudioDeviceManager:
    def __init__(self):
        self._p = pyaudio.PyAudio()

    def get_device_count(self) -> int:
        return self._p.get_device_count()

    def get_device_info(self, index: int) -> dict:
        return self._p.get_device_info_by_index(index)

    def get_input_devices(self) -> list[tuple[int, str]]:
        devices = []
        for i in range(self.get_device_count()):
            info = self.get_device_info(i)
            if info["maxInputChannels"] > 0:
                name = f"{info['name']} ({int(info['defaultSampleRate'])} Hz)"
                devices.append((i, name))
        return devices

    def get_output_devices(self) -> list[tuple[int, str]]:
        devices = []
        for i in range(self.get_device_count()):
            info = self.get_device_info(i)
            if info["maxOutputChannels"] > 0:
                name = f"{info['name']} ({int(info['defaultSampleRate'])} Hz)"
                devices.append((i, name))
        return devices

    def get_default_input_device(self) -> int:
        try:
            return self._p.get_default_input_device_info()["index"]
        except IOError:
            return -1

    def get_default_output_device(self) -> int:
        try:
            return self._p.get_default_output_device_info()["index"]
        except IOError:
            return -1

    def terminate(self):
        self._p.terminate()


class DeviceSelector(QGroupBox):
    """Widget for selecting input/output audio devices and sample rate"""

    device_changed = pyqtSignal()  # Emitted when device or sample rate changes

    def __init__(self, parent=None):
        super().__init__("Audio Settings", parent)

        self.device_manager = AudioDeviceManager()

        self._setup_ui()
        self._populate_devices()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(10)

        # Common stylesheet for combo boxes
        combo_style = """
            QComboBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
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
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #00ff88;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: white;
                selection-background-color: #00ff88;
                selection-color: black;
                border: 1px solid #555555;
            }
        """

        label_style = "color: white; font-weight: bold;"

        # Input device selector
        input_label = QLabel("Input Device:")
        input_label.setStyleSheet(label_style)
        layout.addWidget(input_label, 0, 0)

        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(300)
        self.input_combo.currentIndexChanged.connect(self._on_device_changed)
        self.input_combo.setStyleSheet(combo_style)
        layout.addWidget(self.input_combo, 0, 1)

        # Output device selector
        output_label = QLabel("Output Device:")
        output_label.setStyleSheet(label_style)
        layout.addWidget(output_label, 1, 0)

        self.output_combo = QComboBox()
        self.output_combo.setMinimumWidth(300)
        self.output_combo.currentIndexChanged.connect(self._on_device_changed)
        self.output_combo.setStyleSheet(combo_style)
        layout.addWidget(self.output_combo, 1, 1)

        # Sample rate selector
        sample_rate_label = QLabel("Sample Rate:")
        sample_rate_label.setStyleSheet(label_style)
        layout.addWidget(sample_rate_label, 2, 0)

        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.setMinimumWidth(150)
        self.sample_rate_combo.currentIndexChanged.connect(self._on_device_changed)
        self.sample_rate_combo.setStyleSheet(combo_style)

        # Common sample rates
        sample_rates = [
            (8000, "8000 Hz (Low)"),
            (16000, "16000 Hz"),
            (22050, "22050 Hz"),
            (44100, "44100 Hz (CD Quality)"),
            (48000, "48000 Hz (Studio)"),
            (96000, "96000 Hz (High Res)"),
        ]

        for rate, name in sample_rates:
            self.sample_rate_combo.addItem(name, rate)

        # Default to 44100 Hz
        default_idx = self.sample_rate_combo.findData(44100)
        if default_idx >= 0:
            self.sample_rate_combo.setCurrentIndex(default_idx)

        layout.addWidget(self.sample_rate_combo, 2, 1)

        # Buffer size selector
        buffer_label = QLabel("Buffer Size:")
        buffer_label.setStyleSheet(label_style)
        layout.addWidget(buffer_label, 3, 0)

        self.buffer_combo = QComboBox()
        self.buffer_combo.setMinimumWidth(150)
        self.buffer_combo.currentIndexChanged.connect(self._on_device_changed)
        self.buffer_combo.setStyleSheet(combo_style)

        # Common buffer sizes (in samples)
        buffer_sizes = [
            (64, "64 samples (~1.5ms)"),
            (128, "128 samples (~3ms)"),
            (256, "256 samples (~6ms)"),
            (512, "512 samples (~12ms)"),
            (1024, "1024 samples (~23ms)"),
            (2048, "2048 samples (~46ms)"),
        ]

        for size, name in buffer_sizes:
            self.buffer_combo.addItem(name, size)

        # Default to 1024
        default_idx = self.buffer_combo.findData(1024)
        if default_idx >= 0:
            self.buffer_combo.setCurrentIndex(default_idx)

        layout.addWidget(self.buffer_combo, 3, 1)

        # Refresh button
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self._populate_devices)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #00ff88;
            }
        """)
        layout.addWidget(refresh_btn, 0, 2, 2, 1)

        # Latency display
        self.latency_label = QLabel("Latency: ~23ms")
        self.latency_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.latency_label, 2, 2, 2, 1)

        # Update latency display when settings change
        self.sample_rate_combo.currentIndexChanged.connect(self._update_latency_display)
        self.buffer_combo.currentIndexChanged.connect(self._update_latency_display)

        # Style the group box
        self.setStyleSheet("""
            QGroupBox {
                color: #00ff88;
                font-weight: bold;
                border: 2px solid #333333;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

    def _populate_devices(self):
        """Refresh device lists"""
        current_input = self.get_input_device_index()
        current_output = self.get_output_device_index()

        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)

        self.input_combo.clear()
        self.output_combo.clear()

        # Populate input devices
        input_devices = self.device_manager.get_input_devices()
        default_input = self.device_manager.get_default_input_device()
        default_input_idx = 0

        for i, (device_idx, name) in enumerate(input_devices):
            self.input_combo.addItem(name, device_idx)
            if device_idx == default_input:
                default_input_idx = i

        # Populate output devices
        output_devices = self.device_manager.get_output_devices()
        default_output = self.device_manager.get_default_output_device()
        default_output_idx = 0

        for i, (device_idx, name) in enumerate(output_devices):
            self.output_combo.addItem(name, device_idx)
            if device_idx == default_output:
                default_output_idx = i

        # Restore previous selection or use default
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
        self.device_changed.emit()

    def _update_latency_display(self):
        """Calculate and display approximate latency"""
        sample_rate = self.get_sample_rate()
        buffer_size = self.get_buffer_size()

        if sample_rate > 0:
            # Round-trip latency is roughly 2x buffer size (input + output)
            latency_ms = (buffer_size / sample_rate) * 1000 * 2
            self.latency_label.setText(f"Latency: ~{latency_ms:.1f}ms")

    def get_input_device_index(self) -> int:
        """Returns the PyAudio device index for selected input"""
        return (
            self.input_combo.currentData()
            if self.input_combo.currentIndex() >= 0
            else -1
        )

    def get_output_device_index(self) -> int:
        """Returns the PyAudio device index for selected output"""
        return (
            self.output_combo.currentData()
            if self.output_combo.currentIndex() >= 0
            else -1
        )

    def get_sample_rate(self) -> int:
        """Returns the selected sample rate"""
        return (
            self.sample_rate_combo.currentData()
            if self.sample_rate_combo.currentIndex() >= 0
            else 44100
        )

    def get_buffer_size(self) -> int:
        """Returns the selected buffer size"""
        return (
            self.buffer_combo.currentData()
            if self.buffer_combo.currentIndex() >= 0
            else 1024
        )
