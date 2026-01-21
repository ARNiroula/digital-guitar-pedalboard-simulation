import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QLabel, QFrame, QGridLayout

from .knob import Knob, DiscreteKnob


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
    """
    Feedback Delay Network Reverb with Early Reflections
    """

    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("REVERB", "#4B0082", parent)

        self.sample_rate = sample_rate
        self.decay = 0.5
        self.damping = 0.5
        self.mix = 0.3  # Dry/wet mix
        self.size = 0.5  # Room size

        # Number of FDN delay lines
        self.num_delays = 4

        self._init_early_reflections(sample_rate)
        self._init_fdn_buffers(sample_rate)
        self._init_matrix()

        # Knobs
        self.decay_knob = Knob("DECAY", 0.0, 1.0, self.decay)
        self.decay_knob.valueChanged.connect(lambda v: setattr(self, "decay", v))
        self.add_knob(self.decay_knob, 0, 0)

        self.size_knob = Knob("SIZE", 0.0, 1.0, self.size)
        self.size_knob.valueChanged.connect(self._on_size_changed)
        self.add_knob(self.size_knob, 0, 1)

        self.damp_knob = Knob("DAMP", 0.0, 1.0, self.damping)
        self.damp_knob.valueChanged.connect(lambda v: setattr(self, "damping", v))
        self.add_knob(self.damp_knob, 1, 0)

        self.mix_knob = Knob("MIX", 0.0, 1.0, self.mix)
        self.mix_knob.valueChanged.connect(lambda v: setattr(self, "mix", v))
        self.add_knob(self.mix_knob, 1, 1)

    def _init_early_reflections(self, sample_rate: int):
        """
        Initialize early reflection tap delays
        Simulates first reflections off walls, ceiling, floor
        """
        # Room size scale (0.5x to 2.0x)
        size_scale = 0.5 + self.size * 1.5

        # Early reflection configuration
        # Delays based on typical room reflection patterns
        # Gains decrease with time (inverse square law approximation)
        # Pan values: -1 (left) to 1 (right)
        self.er_config = [
            # (delay_ms, gain, description)
            (7.0, 0.85, "Front wall"),
            (11.0, 0.75, "Side wall L"),
            (13.0, 0.72, "Side wall R"),
            (19.0, 0.65, "Ceiling"),
            (23.0, 0.58, "Back wall"),
            (29.0, 0.50, "Floor bounce"),
            (37.0, 0.42, "Corner 1"),
            (41.0, 0.38, "Corner 2"),
            (47.0, 0.32, "Double bounce 1"),
            (53.0, 0.28, "Double bounce 2"),
            (59.0, 0.22, "Triple bounce 1"),
            (67.0, 0.18, "Triple bounce 2"),
        ]

        # Calculate delay times in samples (scaled by room size)
        self.er_delays = [
            max(1, int(d * size_scale * sample_rate / 1000))
            for d, g, _ in self.er_config
        ]

        # Gains (also scale slightly with room size - larger rooms = more absorption)
        absorption_factor = 1.0 - (self.size * 0.2)  # Larger rooms absorb more
        self.er_gains = [g * absorption_factor for _, g, _ in self.er_config]

        # Single delay buffer for all early reflections
        max_er_delay = max(self.er_delays) + 100
        self.er_buffer = np.zeros(max_er_delay, dtype=np.float32)
        self.er_write_idx = 0

        # Low-pass filter states for each ER tap (simulate wall absorption)
        self.er_lp_states = [0.0] * len(self.er_config)

        # Pre-delay (time before any reflections, creates sense of distance)
        self.predelay_ms = 10.0 * size_scale
        self.predelay_samples = max(1, int(self.predelay_ms * sample_rate / 1000))
        self.predelay_buffer = np.zeros(self.predelay_samples + 100, dtype=np.float32)
        self.predelay_write_idx = 0

    def _init_fdn_buffers(self, sample_rate: int):
        """Initialize FDN delay buffers with prime-length delays"""
        # Room size scale
        size_scale = 0.5 + self.size * 1.5

        # Base delay times in ms (mutually prime for maximum density)
        # Longer than ER delays to create smooth transition
        base_delays_ms = [67.0, 79.0, 89.0, 97.0]  # Prime numbers

        self.delay_times_ms = [d * size_scale for d in base_delays_ms]
        self.delay_samples = [
            max(1, int(d * sample_rate / 1000)) for d in self.delay_times_ms
        ]

        # Create delay buffers
        self.delay_buffers = [
            np.zeros(d + 100, dtype=np.float32) for d in self.delay_samples
        ]
        self.write_indices = [0] * self.num_delays

        # Low-pass filter states for damping in FDN
        self.lp_states = [0.0] * self.num_delays

    def _init_matrix(self):
        """
        Initialize Hadamard feedback matrix
        Orthogonal and energy-preserving
        """
        # 4x4 Hadamard matrix (normalized)
        self.feedback_matrix = (
            np.array(
                [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
                dtype=np.float32,
            )
            * 0.5
        )

    def _on_size_changed(self, value: float):
        """Reinitialize buffers when size changes"""
        self.size = value
        self._init_early_reflections(self.sample_rate)
        self._init_fdn_buffers(self.sample_rate)

    def _apply_matrix(self, inputs: list) -> list:
        """Apply Hadamard feedback matrix"""
        outputs = [0.0] * self.num_delays
        for i in range(self.num_delays):
            for j in range(self.num_delays):
                outputs[i] += self.feedback_matrix[i, j] * inputs[j]
        return outputs

    def _process_early_reflections(self, sample: float) -> float:
        """
        Process early reflections using multi-tap delay
        Returns the sum of all early reflection taps
        """
        er_buffer = self.er_buffer
        buf_len = len(er_buffer)
        write_idx = self.er_write_idx

        # Write input to ER buffer
        er_buffer[write_idx] = sample

        # Sum all reflection taps
        er_output = 0.0

        # High-frequency damping coefficient for ER
        # Walls absorb high frequencies
        er_damp = 0.3 + self.damping * 0.4

        for i, delay in enumerate(self.er_delays):
            # Read from delay buffer
            read_idx = (write_idx - delay) % buf_len
            tap_output = er_buffer[read_idx]

            # Apply per-tap low-pass filter (wall absorption)
            self.er_lp_states[i] = (
                tap_output * (1.0 - er_damp) + self.er_lp_states[i] * er_damp
            )

            # Add to output with gain
            er_output += self.er_lp_states[i] * self.er_gains[i]

        # Advance write index
        self.er_write_idx = (write_idx + 1) % buf_len

        # Normalize (prevent clipping)
        er_output *= 0.3

        return er_output

    def _process_predelay(self, sample: float) -> float:
        """Simple pre-delay buffer"""
        buf = self.predelay_buffer
        buf_len = len(buf)
        write_idx = self.predelay_write_idx

        # Read delayed sample
        read_idx = (write_idx - self.predelay_samples) % buf_len
        output = buf[read_idx]

        # Write new sample
        buf[write_idx] = sample

        # Advance write index
        self.predelay_write_idx = (write_idx + 1) % buf_len

        return output

    def _process_fdn(self, sample: float) -> float:
        """
        Process through Feedback Delay Network
        """
        # Calculate decay parameters
        t60 = 0.5 + self.decay * 4.5  # Reverb time 0.5s to 5s
        damp_coef = self.damping * 0.7

        # Read from all delay lines
        delay_outputs = []
        for j in range(self.num_delays):
            buf = self.delay_buffers[j]
            buf_len = len(buf)
            delay = self.delay_samples[j]
            write_idx = self.write_indices[j]

            read_idx = (write_idx - delay) % buf_len
            delay_outputs.append(buf[read_idx])

        # Apply feedback matrix (Hadamard mixing)
        mixed = self._apply_matrix(delay_outputs)

        # Process each delay line
        for j in range(self.num_delays):
            buf = self.delay_buffers[j]
            buf_len = len(buf)
            write_idx = self.write_indices[j]
            delay_time = self.delay_times_ms[j] / 1000.0

            # Calculate feedback gain for uniform decay
            feedback_gain = 10.0 ** (-3.0 * delay_time / t60)
            feedback_gain = min(feedback_gain, 0.98)

            # Apply damping (low-pass filter)
            filtered = mixed[j] * (1.0 - damp_coef) + self.lp_states[j] * damp_coef
            self.lp_states[j] = filtered

            # Apply feedback gain and soft clip
            feedback_sample = np.tanh(filtered * feedback_gain)

            # Write input + feedback to buffer
            buf[write_idx] = sample * 0.25 + feedback_sample

            # Advance write index
            self.write_indices[j] = (write_idx + 1) % buf_len

        # Sum delay outputs for wet signal
        fdn_output = sum(delay_outputs) * 0.25

        return fdn_output

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            dry_sample = audio[i]

            # Pre-delay
            predelayed = self._process_predelay(dry_sample)

            # Early reflections (distinct room bounces)
            er_output = self._process_early_reflections(predelayed)

            # FDN late reverb (dense tail)
            fdn_input = predelayed + er_output * 0.5
            fdn_output = self._process_fdn(fdn_input)

            # Combine ER and FDN
            wet = er_output + fdn_output

            # Soft clip wet signal
            wet = np.tanh(wet)

            # Mix dry and wet
            output[i] = dry_sample * (1.0 - self.mix) + wet * self.mix * 1.5

        return output

    def reset(self):
        """Reset all buffers and states"""
        # Reset ER buffers
        self.er_buffer.fill(0.0)
        self.er_write_idx = 0
        self.er_lp_states = [0.0] * len(self.er_config)

        # Reset pre-delay
        self.predelay_buffer.fill(0.0)
        self.predelay_write_idx = 0

        # Reset FDN buffers
        for buf in self.delay_buffers:
            buf.fill(0.0)
        self.write_indices = [0] * self.num_delays
        self.lp_states = [0.0] * self.num_delays


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


class TremoloPedal(Pedal):
    def __init__(self, sample_rate: int = 44100, parent=None):
        super().__init__("Tremelo", "#FF6123", parent)

        self.sample_rate = sample_rate
        self.rate = 0.5  # LFO rate
        self.depth = 0.7  # Modulation Depth
        self.wave = "Sine"
        # LFO Phase
        self.lfo_phase = 0.0

        # Create knobs
        self.rate_knob = Knob("RATE", 0.05, 5.0, self.rate)
        self.rate_knob.valueChanged.connect(lambda v: setattr(self, "rate", v))
        self.add_knob(self.rate_knob, 0, 0)

        self.depth_knob = Knob("DEPTH", 0.0, 1.0, self.depth)
        self.depth_knob.valueChanged.connect(lambda v: setattr(self, "depth", v))
        self.add_knob(self.depth_knob, 0, 1)

        self.wave_knob = DiscreteKnob(
            label="WAVE",
            values=["Sine", "Square", "Saw", "Triangle"],
            default_index=0,
        )
        self.wave_knob.valueChangedDiscrete.connect(lambda v: setattr(self, "wave", v))
        self.add_knob(self.wave_knob, 1, 0)

    def _get_lfo_value(self, phase: float) -> float:
        """Generate LFO value based on waveform"""

        # Normalize Phase
        norm_phase = phase / (2.0 * np.pi)
        aug_phase = norm_phase
        if self.wave == "Sine":
            aug_phase = phase

        waves = {
            "Sine": lambda x: np.sin(x),
            "Square": lambda x: 1.0 if x % 1.0 < 0.5 else -1,
            "Saw": lambda x: 2.0 * (x % 1.0) - 1.0,
            "Triangle": lambda x: 4.0 * abs((x % 1.0) - 0.5) - 1.0,
        }

        return waves.get(self.wave, "Sine")(aug_phase)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio

        num_samples = len(audio)
        output = np.zeros(num_samples, dtype=np.float32)

        # LFO increments
        lfo_inc = 2.0 * np.pi * self.rate / self.sample_rate

        for i in range(num_samples):
            # Get LFO value (-1 to 1)
            lfo = self._get_lfo_value(self.lfo_phase)

            # Convert LFO to amplitude modulation
            # depth = 0; no effect
            # depth = 1; max effect
            amplitude = 1.0 - (self.depth * 0.5 * (1.0 + lfo))

            # Apply output
            output[i] = audio[i] * amplitude

            # Update LFO phase
            self.lfo_phase += lfo_inc

            # Out of bound check
            if self.lfo_phase >= 2.0 * np.pi:
                self.lfo_phase -= 2.0 * np.pi

        return output

    def reset(self):
        """Reset the effect"""
        self.lfo_phase = 0.0


class PhaserPedal(Pedal):
    def __init__():
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        pass

    def reset(self):
        pass
