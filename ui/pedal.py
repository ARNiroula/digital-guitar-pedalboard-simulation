import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QLabel, QFrame, QGridLayout

from .knob import Knob


class Pedal(QFrame):
    """Base class for effect pedals"""

    def __init__(self, name: str, color: str = "#444444", parent=None):
        super().__init__(parent)
        self.name = name
        self.enabled = True

        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet(f"""
            Pedal {{
                background-color: {color};
                border: 2px solid #222222;
                border-radius: 10px;
            }}
        """)
        self.setMinimumSize(150, 200)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        # Pedal name
        title = QLabel(self.name)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        layout.addWidget(title)

        # Knobs container
        self.knobs_layout = QGridLayout()
        self.knobs_layout.setSpacing(10)
        layout.addLayout(self.knobs_layout)

        layout.addStretch()

        # Bypass button (footswitch style)
        self.bypass_btn = QPushButton("ON")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.setChecked(True)
        self.bypass_btn.clicked.connect(self._toggle_bypass)
        self.bypass_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #00ff88;
                border: 2px solid #333333;
                border-radius: 20px;
                padding: 10px;
                font-weight: bold;
                min-height: 40px;
            }
            QPushButton:checked {
                background-color: #00ff88;
                color: #1a1a1a;
            }
        """)
        layout.addWidget(self.bypass_btn)

    def _toggle_bypass(self, checked):
        self.enabled = checked
        self.bypass_btn.setText("ON" if checked else "OFF")

    def add_knob(self, knob: Knob, row: int = 0, col: int = 0):
        self.knobs_layout.addWidget(knob, row, col, Qt.AlignmentFlag.AlignCenter)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Override in subclass"""
        return audio


class DistortionPedal(Pedal):
    """Distortion/Overdrive effect"""

    def __init__(self, parent=None):
        super().__init__("DISTORTION", "#8B0000", parent)

        # Parameters
        self.gain = 5.0
        self.tone = 0.5
        self.level = 0.7

        # Create knobs
        self.gain_knob = Knob("GAIN", 1.0, 20.0, self.gain)
        self.gain_knob.valueChanged.connect(lambda v: setattr(self, "gain", v))
        self.add_knob(self.gain_knob, 0, 0)

        self.tone_knob = Knob("TONE", 0.0, 1.0, self.tone)
        self.tone_knob.valueChanged.connect(lambda v: setattr(self, "tone", v))
        self.add_knob(self.tone_knob, 0, 1)

        self.level_knob = Knob("LEVEL", 0.0, 1.0, self.level)
        self.level_knob.valueChanged.connect(lambda v: setattr(self, "level", v))
        self.add_knob(self.level_knob, 1, 0)

        # Simple low-pass filter state
        self._lp_state = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        # Apply gain
        signal = audio * self.gain

        # Soft clipping (tanh distortion)
        signal = np.tanh(signal)

        # Simple one-pole low-pass filter for tone control
        # tone=0 -> darker, tone=1 -> brighter
        cutoff = 0.1 + 0.9 * self.tone  # Filter coefficient
        output = np.zeros_like(signal)
        state = self._lp_state

        for i, sample in enumerate(signal):
            state = state + cutoff * (sample - state)
            output[i] = state

        self._lp_state = state

        # Blend filtered and original based on tone
        signal = output * (1 - self.tone * 0.5) + signal * (self.tone * 0.5)

        # Apply output level
        signal = signal * self.level

        return signal


class DelayPedal(Pedal):
    """Simple delay effect"""

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("DELAY", "#00008B", parent)

        self.sample_rate = sample_rate

        # Parameters
        self.time_ms = 300.0
        self.feedback = 0.4
        self.mix = 0.5

        # Delay buffer (max 1 second)
        self.max_delay_samples = sample_rate
        self.buffer = np.zeros(self.max_delay_samples)
        self.write_idx = 0

        # Create knobs
        self.time_knob = Knob("TIME", 50.0, 1000.0, self.time_ms)
        self.time_knob.valueChanged.connect(lambda v: setattr(self, "time_ms", v))
        self.add_knob(self.time_knob, 0, 0)

        self.feedback_knob = Knob("FDBK", 0.0, 0.9, self.feedback)
        self.feedback_knob.valueChanged.connect(lambda v: setattr(self, "feedback", v))
        self.add_knob(self.feedback_knob, 0, 1)

        self.mix_knob = Knob("MIX", 0.0, 1.0, self.mix)
        self.mix_knob.valueChanged.connect(lambda v: setattr(self, "mix", v))
        self.add_knob(self.mix_knob, 1, 0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        delay_samples = int(self.time_ms * self.sample_rate / 1000)
        delay_samples = min(delay_samples, self.max_delay_samples - 1)

        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            # Read from delay buffer
            read_idx = (self.write_idx - delay_samples) % self.max_delay_samples
            delayed = self.buffer[read_idx]

            # Write to delay buffer (input + feedback)
            self.buffer[self.write_idx] = sample + delayed * self.feedback

            # Mix dry and wet
            output[i] = sample * (1 - self.mix) + delayed * self.mix

            # Advance write position
            self.write_idx = (self.write_idx + 1) % self.max_delay_samples

        return output
