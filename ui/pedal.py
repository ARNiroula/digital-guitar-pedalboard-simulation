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
        self.setMinimumSize(300, 350)

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


class ReverbPedal(Pedal):
    """Simple reverb using multiple delay lines (Schroeder reverb)"""

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("REVERB", "#4B0082", parent)

        self.sample_rate = sample_rate
        self.room_size = 0.5
        self.damping = 0.5
        self.mix = 0.3

        # Comb filter delays (in ms)
        comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
        self.comb_delays = [int(d * sample_rate / 1000) for d in comb_delays_ms]
        self.comb_buffers = [np.zeros(d + 1000) for d in self.comb_delays]
        self.comb_indices = [0] * 4
        self.comb_filters = [0.0] * 4

        # Allpass filter delays
        allpass_delays_ms = [5.0, 1.7]
        self.allpass_delays = [int(d * sample_rate / 1000) for d in allpass_delays_ms]
        self.allpass_buffers = [np.zeros(d + 100) for d in self.allpass_delays]
        self.allpass_indices = [0] * 2

        self.room_knob = Knob("ROOM", 0.0, 1.0, self.room_size)
        self.room_knob.valueChanged.connect(lambda v: setattr(self, "room_size", v))
        self.add_knob(self.room_knob, 0, 0)

        self.damp_knob = Knob("DAMP", 0.0, 1.0, self.damping)
        self.damp_knob.valueChanged.connect(lambda v: setattr(self, "damping", v))
        self.add_knob(self.damp_knob, 0, 1)

        self.mix_knob = Knob("MIX", 0.0, 1.0, self.mix)
        self.mix_knob.valueChanged.connect(lambda v: setattr(self, "mix", v))
        self.add_knob(self.mix_knob, 1, 0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        output = np.zeros_like(audio)
        feedback = 0.7 + 0.28 * self.room_size
        damp = self.damping * 0.4

        for i, sample in enumerate(audio):
            # Parallel comb filters
            comb_out = 0.0
            for j in range(4):
                delay = self.comb_delays[j]
                buf = self.comb_buffers[j]
                idx = self.comb_indices[j]

                read_idx = (idx - delay) % len(buf)
                delayed = buf[read_idx]

                # Low-pass filter in feedback
                self.comb_filters[j] = (
                    delayed * (1 - damp) + self.comb_filters[j] * damp
                )
                buf[idx] = sample + self.comb_filters[j] * feedback

                comb_out += delayed
                self.comb_indices[j] = (idx + 1) % len(buf)

            comb_out *= 0.25

            # Series allpass filters
            allpass_out = comb_out
            for j in range(2):
                delay = self.allpass_delays[j]
                buf = self.allpass_buffers[j]
                idx = self.allpass_indices[j]

                read_idx = (idx - delay) % len(buf)
                delayed = buf[read_idx]

                buf[idx] = allpass_out + delayed * 0.5
                allpass_out = delayed - allpass_out * 0.5

                self.allpass_indices[j] = (idx + 1) % len(buf)

            output[i] = sample * (1 - self.mix) + allpass_out * self.mix

        return output


class ChorusPedal(Pedal):
    """Chorus effect using modulated delay"""

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("CHORUS", "#006400", parent)

        self.sample_rate = sample_rate
        self.rate = 1.5  # LFO rate in Hz
        self.depth = 0.5
        self.mix = 0.5

        # Delay buffer (max ~30ms)
        self.max_delay = int(0.03 * sample_rate)
        self.buffer = np.zeros(self.max_delay)
        self.write_idx = 0

        # LFO phase
        self.lfo_phase = 0.0

        self.rate_knob = Knob("RATE", 0.1, 5.0, self.rate)
        self.rate_knob.valueChanged.connect(lambda v: setattr(self, "rate", v))
        self.add_knob(self.rate_knob, 0, 0)

        self.depth_knob = Knob("DEPTH", 0.0, 1.0, self.depth)
        self.depth_knob.valueChanged.connect(lambda v: setattr(self, "depth", v))
        self.add_knob(self.depth_knob, 0, 1)

        self.mix_knob = Knob("MIX", 0.0, 1.0, self.mix)
        self.mix_knob.valueChanged.connect(lambda v: setattr(self, "mix", v))
        self.add_knob(self.mix_knob, 1, 0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        output = np.zeros_like(audio)
        lfo_inc = 2 * np.pi * self.rate / self.sample_rate

        # Base delay ~7ms, modulation depth ~3ms
        base_delay = int(0.007 * self.sample_rate)
        mod_depth = int(0.003 * self.sample_rate * self.depth)

        for i, sample in enumerate(audio):
            # Write to buffer
            self.buffer[self.write_idx] = sample

            # Calculate modulated delay
            lfo = np.sin(self.lfo_phase)
            delay = base_delay + int(lfo * mod_depth)
            delay = max(1, min(delay, self.max_delay - 1))

            # Read with linear interpolation
            read_pos = (self.write_idx - delay) % self.max_delay
            read_idx = int(read_pos)
            frac = read_pos - read_idx

            delayed = self.buffer[read_idx] * (1 - frac)
            delayed += self.buffer[(read_idx + 1) % self.max_delay] * frac

            output[i] = sample * (1 - self.mix) + delayed * self.mix

            self.write_idx = (self.write_idx + 1) % self.max_delay
            self.lfo_phase += lfo_inc
            if self.lfo_phase > 2 * np.pi:
                self.lfo_phase -= 2 * np.pi

        return output


class EQPedal(Pedal):
    """3-band EQ"""

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("EQ", "#8B4513", parent)

        self.sample_rate = sample_rate
        self.low = 0.5  # 0-1, 0.5 = flat
        self.mid = 0.5
        self.high = 0.5

        # Filter states
        self._lp_state = 0.0
        self._hp_state = 0.0
        self._bp_state1 = 0.0
        self._bp_state2 = 0.0

        self.low_knob = Knob("LOW", 0.0, 1.0, self.low)
        self.low_knob.valueChanged.connect(lambda v: setattr(self, "low", v))
        self.add_knob(self.low_knob, 0, 0)

        self.mid_knob = Knob("MID", 0.0, 1.0, self.mid)
        self.mid_knob.valueChanged.connect(lambda v: setattr(self, "mid", v))
        self.add_knob(self.mid_knob, 0, 1)

        self.high_knob = Knob("HIGH", 0.0, 1.0, self.high)
        self.high_knob.valueChanged.connect(lambda v: setattr(self, "high", v))
        self.add_knob(self.high_knob, 1, 0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        output = np.zeros_like(audio)

        # Filter coefficients (approximate cutoffs: 200Hz, 2kHz)
        lp_coef = 0.05  # Low-pass for bass
        hp_coef = 0.1  # High-pass for treble

        # Gain multipliers (convert 0-1 to 0.25-4x)
        low_gain = 0.25 + self.low * 3.75
        mid_gain = 0.25 + self.mid * 3.75
        high_gain = 0.25 + self.high * 3.75

        for i, sample in enumerate(audio):
            # Low-pass for bass
            self._lp_state += lp_coef * (sample - self._lp_state)
            low_band = self._lp_state

            # High-pass for treble
            self._hp_state += hp_coef * (sample - self._hp_state)
            high_band = sample - self._hp_state

            # Mid is what's left
            mid_band = sample - low_band - high_band

            output[i] = (
                low_band * low_gain + mid_band * mid_gain + high_band * high_gain
            )

        return np.clip(output, -1.0, 1.0)


class CompressorPedal(Pedal):
    """Dynamic range compressor"""

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("COMP", "#2F4F4F", parent)

        self.sample_rate = sample_rate
        self.threshold = 0.5  # 0-1 maps to -40 to 0 dB
        self.ratio = 0.5  # 0-1 maps to 1:1 to 20:1
        self.gain = 0.5  # Makeup gain

        # Envelope follower state
        self._envelope = 0.0

        self.thresh_knob = Knob("THRESH", 0.0, 1.0, self.threshold)
        self.thresh_knob.valueChanged.connect(lambda v: setattr(self, "threshold", v))
        self.add_knob(self.thresh_knob, 0, 0)

        self.ratio_knob = Knob("RATIO", 0.0, 1.0, self.ratio)
        self.ratio_knob.valueChanged.connect(lambda v: setattr(self, "ratio", v))
        self.add_knob(self.ratio_knob, 0, 1)

        self.gain_knob = Knob("GAIN", 0.0, 1.0, self.gain)
        self.gain_knob.valueChanged.connect(lambda v: setattr(self, "gain", v))
        self.add_knob(self.gain_knob, 1, 0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        output = np.zeros_like(audio)

        # Convert parameters
        thresh_db = -40 + self.threshold * 40  # -40 to 0 dB
        thresh_lin = 10 ** (thresh_db / 20)
        ratio = 1 + self.ratio * 19  # 1:1 to 20:1
        makeup = 1 + self.gain * 3  # 1x to 4x

        # Attack and release (in samples)
        attack = 0.01  # Fast attack
        release = 0.1  # Slower release

        for i, sample in enumerate(audio):
            # Envelope follower
            input_level = abs(sample)
            if input_level > self._envelope:
                self._envelope += attack * (input_level - self._envelope)
            else:
                self._envelope += release * (input_level - self._envelope)

            # Compute gain reduction
            if self._envelope > thresh_lin:
                # How much over threshold (in dB)
                over_db = 20 * np.log10(self._envelope / thresh_lin + 1e-10)
                # Reduce by ratio
                reduce_db = over_db * (1 - 1 / ratio)
                gain_reduction = 10 ** (-reduce_db / 20)
            else:
                gain_reduction = 1.0

            output[i] = sample * gain_reduction * makeup

        return np.clip(output, -1.0, 1.0)
