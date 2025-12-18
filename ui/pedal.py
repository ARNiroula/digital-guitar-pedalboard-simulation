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


class DelayPedal(Pedal):
    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("DELAY", "#00008B", parent)

        self.sample_rate = sample_rate
        self.time_ms = 300.0
        self.feedback = 0.4
        self.mix = 0.5

        self.max_delay_samples = sample_rate
        self.buffer = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.write_idx = 0

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

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        delay_samples = int(self.time_ms * self.sample_rate / 1000)
        delay_samples = max(1, min(delay_samples, self.max_delay_samples - 1))

        for i in range(num_samples):
            read_idx = (self.write_idx - delay_samples) % self.max_delay_samples
            delayed = self.buffer[read_idx]
            self.buffer[self.write_idx] = audio[i] + delayed * self.feedback
            output[i] = audio[i] * (1 - self.mix) + delayed * self.mix
            self.write_idx = (self.write_idx + 1) % self.max_delay_samples

        return output


class ReverbPedal(Pedal):
    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("REVERB", "#4B0082", parent)

        self.sample_rate = sample_rate
        self.room_size = 0.5
        self.damping = 0.5
        self.mix = 0.3

        self._init_buffers(sample_rate)

        self.room_knob = Knob("ROOM", 0.0, 1.0, self.room_size)
        self.room_knob.valueChanged.connect(lambda v: setattr(self, "room_size", v))
        self.add_knob(self.room_knob, 0, 0)

        self.damp_knob = Knob("DAMP", 0.0, 1.0, self.damping)
        self.damp_knob.valueChanged.connect(lambda v: setattr(self, "damping", v))
        self.add_knob(self.damp_knob, 0, 1)

        self.mix_knob = Knob("MIX", 0.0, 1.0, self.mix)
        self.mix_knob.valueChanged.connect(lambda v: setattr(self, "mix", v))
        self.add_knob(self.mix_knob, 1, 0)

    def _init_buffers(self, sample_rate: int):
        comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
        self.comb_delays = [max(1, int(d * sample_rate / 1000)) for d in comb_delays_ms]
        self.comb_buffers = [
            np.zeros(d + 1000, dtype=np.float32) for d in self.comb_delays
        ]
        self.comb_indices = [0] * 4
        self.comb_filters = [0.0] * 4

        allpass_delays_ms = [5.0, 1.7]
        self.allpass_delays = [
            max(1, int(d * sample_rate / 1000)) for d in allpass_delays_ms
        ]
        self.allpass_buffers = [
            np.zeros(d + 100, dtype=np.float32) for d in self.allpass_delays
        ]
        self.allpass_indices = [0] * 2

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)
        feedback = 0.7 + 0.28 * self.room_size
        damp = self.damping * 0.4

        for i in range(num_samples):
            sample = audio[i]
            comb_out = 0.0

            for j in range(4):
                delay = self.comb_delays[j]
                buf = self.comb_buffers[j]
                buf_len = len(buf)
                idx = self.comb_indices[j] % buf_len

                read_idx = (idx - delay) % buf_len
                delayed = buf[read_idx]

                self.comb_filters[j] = (
                    delayed * (1 - damp) + self.comb_filters[j] * damp
                )
                buf[idx] = sample + self.comb_filters[j] * feedback

                comb_out += delayed
                self.comb_indices[j] = (idx + 1) % buf_len

            comb_out *= 0.25

            allpass_out = comb_out
            for j in range(2):
                delay = self.allpass_delays[j]
                buf = self.allpass_buffers[j]
                buf_len = len(buf)
                idx = self.allpass_indices[j] % buf_len

                read_idx = (idx - delay) % buf_len
                delayed = buf[read_idx]

                buf[idx] = allpass_out + delayed * 0.5
                allpass_out = delayed - allpass_out * 0.5

                self.allpass_indices[j] = (idx + 1) % buf_len

            output[i] = sample * (1 - self.mix) + allpass_out * self.mix

        return output


class ChorusPedal(Pedal):
    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("CHORUS", "#006400", parent)

        self.sample_rate = sample_rate
        self.rate = 1.5
        self.depth = 0.5
        self.mix = 0.5

        self.max_delay = int(0.03 * sample_rate)
        self.buffer = np.zeros(self.max_delay, dtype=np.float32)
        self.write_idx = 0
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

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)
        lfo_inc = 2 * np.pi * self.rate / self.sample_rate

        base_delay = int(0.007 * self.sample_rate)
        mod_depth = int(0.003 * self.sample_rate * self.depth)

        for i in range(num_samples):
            self.buffer[self.write_idx] = audio[i]

            lfo = np.sin(self.lfo_phase)
            delay = base_delay + int(lfo * mod_depth)
            delay = max(1, min(delay, self.max_delay - 2))

            read_idx = (self.write_idx - delay) % self.max_delay
            next_read_idx = (read_idx + 1) % self.max_delay
            frac = (self.write_idx - delay) - int(self.write_idx - delay)

            delayed = (
                self.buffer[read_idx] * (1 - frac) + self.buffer[next_read_idx] * frac
            )

            output[i] = audio[i] * (1 - self.mix) + delayed * self.mix

            self.write_idx = (self.write_idx + 1) % self.max_delay
            self.lfo_phase += lfo_inc
            if self.lfo_phase > 2 * np.pi:
                self.lfo_phase -= 2 * np.pi

        return output


class FlangerPedal(Pedal):
    """Flanger effect - short modulated delay with feedback"""

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("FLANGER", "#FF6347", parent)

        self.sample_rate = sample_rate
        self.rate = 0.5  # LFO rate in Hz (slow sweep)
        self.depth = 0.7  # Modulation depth
        self.feedback = 0.7  # Feedback amount (creates resonance)
        self.mix = 0.5  # Dry/wet mix

        # Delay buffer
        self.max_delay_ms = 10.0
        self.max_delay_samples = int(self.max_delay_ms * sample_rate / 1000)
        self.buffer = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.write_idx = 0

        # LFO phase
        self.lfo_phase = 0.0

        # Create knobs
        self.rate_knob = Knob("RATE", 0.05, 5.0, self.rate)
        self.rate_knob.valueChanged.connect(lambda v: setattr(self, "rate", v))
        self.add_knob(self.rate_knob, 0, 0)

        self.depth_knob = Knob("DEPTH", 0.0, 1.0, self.depth)
        self.depth_knob.valueChanged.connect(lambda v: setattr(self, "depth", v))
        self.add_knob(self.depth_knob, 0, 1)

        self.feedback_knob = Knob("FDBK", 0.0, 0.95, self.feedback)
        self.feedback_knob.valueChanged.connect(lambda v: setattr(self, "feedback", v))
        self.add_knob(self.feedback_knob, 1, 0)

        self.mix_knob = Knob("MIX", 0.0, 1.0, self.mix)
        self.mix_knob.valueChanged.connect(lambda v: setattr(self, "mix", v))
        self.add_knob(self.mix_knob, 1, 1)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        # LFO increment per sample
        lfo_inc = 2.0 * np.pi * self.rate / self.sample_rate

        # Delay range: 0.1ms to max_delay_ms
        min_delay_samples = int(0.1 * self.sample_rate / 1000)  # 0.1ms minimum
        max_mod_samples = int(self.max_delay_ms * self.sample_rate / 1000 * self.depth)

        for i in range(num_samples):
            # Calculate modulated delay using sine LFO
            lfo = (np.sin(self.lfo_phase) + 1.0) * 0.5

            # Calculate delay in samples
            delay_samples = min_delay_samples + int(lfo * max_mod_samples)
            delay_samples = max(1, min(delay_samples, self.max_delay_samples - 2))

            # Read from delay buffer with linear interpolation
            read_pos = self.write_idx - delay_samples
            if read_pos < 0:
                read_pos += self.max_delay_samples

            read_idx = int(read_pos) % self.max_delay_samples
            next_idx = (read_idx + 1) % self.max_delay_samples
            frac = read_pos - int(read_pos)

            # Linear interpolation for smooth delay modulation
            delayed = (
                self.buffer[read_idx] * (1.0 - frac) + self.buffer[next_idx] * frac
            )

            # Input + feedback
            input_sample = audio[i] + delayed * self.feedback

            # Soft clip feedback to prevent runaway
            input_sample = np.tanh(input_sample)

            # Write to delay buffer
            self.buffer[self.write_idx] = input_sample

            # Mix dry and wet signals
            output[i] = audio[i] * (1.0 - self.mix) + delayed * self.mix

            # Advance write index
            self.write_idx = (self.write_idx + 1) % self.max_delay_samples

            # Advance LFO phase
            self.lfo_phase += lfo_inc
            if self.lfo_phase >= 2.0 * np.pi:
                self.lfo_phase -= 2.0 * np.pi

        return output

    def reset(self):
        """Reset the effect state"""
        self.buffer = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.write_idx = 0
        self.lfo_phase = 0.0


class EQPedal(Pedal):
    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("EQ", "#8B4513", parent)

        self.sample_rate = sample_rate
        self.low = 0.5
        self.mid = 0.5
        self.high = 0.5

        self._lp_state = 0.0
        self._hp_state = 0.0

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

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        lp_coef = 0.05
        hp_coef = 0.1

        low_gain = 0.25 + self.low * 3.75
        mid_gain = 0.25 + self.mid * 3.75
        high_gain = 0.25 + self.high * 3.75

        for i in range(num_samples):
            sample = audio[i]

            self._lp_state += lp_coef * (sample - self._lp_state)
            low_band = self._lp_state

            self._hp_state += hp_coef * (sample - self._hp_state)
            high_band = sample - self._hp_state

            mid_band = sample - low_band - high_band

            output[i] = (
                low_band * low_gain + mid_band * mid_gain + high_band * high_gain
            )

        return np.clip(output, -1.0, 1.0)


class CompressorPedal(Pedal):
    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("COMPRESSOR", "#2F4F4F", parent)

        self.sample_rate = sample_rate
        self.threshold = 0.5
        self.ratio = 0.5
        self.gain = 0.5

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

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        thresh_db = -40 + self.threshold * 40
        thresh_lin = 10 ** (thresh_db / 20)
        ratio = 1 + self.ratio * 19
        makeup = 1 + self.gain * 3

        attack = 0.01
        release = 0.1

        for i in range(num_samples):
            input_level = abs(audio[i])
            if input_level > self._envelope:
                self._envelope += attack * (input_level - self._envelope)
            else:
                self._envelope += release * (input_level - self._envelope)

            if self._envelope > thresh_lin:
                over_db = 20 * np.log10(self._envelope / thresh_lin + 1e-10)
                reduce_db = over_db * (1 - 1 / ratio)
                gain_reduction = 10 ** (-reduce_db / 20)
            else:
                gain_reduction = 1.0

            output[i] = audio[i] * gain_reduction * makeup

        return np.clip(output, -1.0, 1.0)


class DistortionPedal(Pedal):
    def __init__(self, parent=None):
        super().__init__("DISTORTION", "#8B0000", parent)

        self.gain = 5.0
        self.tone = 0.5
        self.level = 0.7

        self.gain_knob = Knob("GAIN", 1.0, 20.0, self.gain)
        self.gain_knob.valueChanged.connect(lambda v: setattr(self, "gain", v))
        self.add_knob(self.gain_knob, 0, 0)

        self.tone_knob = Knob("TONE", 0.0, 1.0, self.tone)
        self.tone_knob.valueChanged.connect(lambda v: setattr(self, "tone", v))
        self.add_knob(self.tone_knob, 0, 1)

        self.level_knob = Knob("LEVEL", 0.0, 1.0, self.level)
        self.level_knob.valueChanged.connect(lambda v: setattr(self, "level", v))
        self.add_knob(self.level_knob, 1, 0)

        self._lp_state = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        cutoff = 0.1 + 0.9 * self.tone

        # Clipping threshold
        clip_threshold = 0.8

        for i in range(num_samples):
            # Apply gain
            sample = audio[i] * self.gain

            # Hard clipping
            if sample > clip_threshold:
                sample = clip_threshold
            elif sample < -clip_threshold:
                sample = -clip_threshold

            # Simple low-pass filter for tone control
            self._lp_state += cutoff * (sample - self._lp_state)

            # Blend filtered and original based on tone
            filtered = self._lp_state * (1 - self.tone * 0.5) + sample * (
                self.tone * 0.5
            )

            output[i] = filtered * self.level

        return output


class OverdrivePedal(Pedal):
    """Overdrive effect with soft clipping (tube-like)"""

    def __init__(self, parent=None):
        super().__init__("OVERDRIVE", "#CD853F", parent)

        self.gain = 3.0
        self.tone = 0.5
        self.level = 0.7

        self.gain_knob = Knob("DRIVE", 1.0, 15.0, self.gain)
        self.gain_knob.valueChanged.connect(lambda v: setattr(self, "gain", v))
        self.add_knob(self.gain_knob, 0, 0)

        self.tone_knob = Knob("TONE", 0.0, 1.0, self.tone)
        self.tone_knob.valueChanged.connect(lambda v: setattr(self, "tone", v))
        self.add_knob(self.tone_knob, 0, 1)

        self.level_knob = Knob("LEVEL", 0.0, 1.0, self.level)
        self.level_knob.valueChanged.connect(lambda v: setattr(self, "level", v))
        self.add_knob(self.level_knob, 1, 0)

        self._lp_state = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        cutoff = 0.1 + 0.9 * self.tone

        for i in range(num_samples):
            # Apply gain
            sample = audio[i] * self.gain

            # Soft clipping using tanh
            sample = np.tanh(sample)

            # Simple low-pass filter for tone control
            self._lp_state += cutoff * (sample - self._lp_state)

            # Blend filtered and original based on tone
            filtered = self._lp_state * (1 - self.tone * 0.5) + sample * (
                self.tone * 0.5
            )

            output[i] = filtered * self.level

        return output
