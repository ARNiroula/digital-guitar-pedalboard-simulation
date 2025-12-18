import threading

import numpy as np


class KarplusStrongSynth:
    """Karplus-Strong algorithm for plucked string synthesis"""

    GUITAR_NOTES = {
        "E2": 82.41,
        "F2": 87.31,
        "F#2": 92.50,
        "G2": 98.00,
        "G#2": 103.83,
        "A2": 110.00,
        "A#2": 116.54,
        "B2": 123.47,
        "C3": 130.81,
        "C#3": 138.59,
        "D3": 146.83,
        "D#3": 155.56,
        "E3": 164.81,
        "F3": 174.61,
        "F#3": 185.00,
        "G3": 196.00,
        "G#3": 207.65,
        "A3": 220.00,
        "A#3": 233.08,
        "B3": 246.94,
        "C4": 261.63,
        "C#4": 277.18,
        "D4": 293.66,
        "D#4": 311.13,
        "E4": 329.63,
        "F4": 349.23,
        "F#4": 369.99,
        "G4": 392.00,
        "G#4": 415.30,
        "A4": 440.00,
    }

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.active_strings = []
        self.damping = 0.996
        self._lock = threading.Lock()
        self._max_polyphony = 12
        self._string_id_counter = 0

    def pluck(self, frequency: float, amplitude: float = 0.8):
        """Trigger a new plucked string"""
        period = int(self.sample_rate / frequency)
        if period < 2:
            return

        # Initialize with noise burst
        noise = np.random.uniform(-amplitude, amplitude, period).astype(np.float32)
        envelope = np.linspace(1.0, 0.8, period, dtype=np.float32)
        noise = noise * envelope

        with self._lock:
            self._string_id_counter += 1
            string = {
                "id": self._string_id_counter,  # Unique ID for comparison
                "buffer": noise.copy(),
                "index": 0,
                "period": period,
                "active": True,
                "age": 0,
            }

            self.active_strings.append(string)

            # Limit polyphony
            while len(self.active_strings) > self._max_polyphony:
                self.active_strings.pop(0)

    def pluck_note(self, note_name: str, amplitude: float = 0.8):
        """Pluck a note by name"""
        if note_name in self.GUITAR_NOTES:
            self.pluck(self.GUITAR_NOTES[note_name], amplitude)

    def pluck_midi(self, midi_note: int, amplitude: float = 0.8):
        """Pluck a note by MIDI number"""
        frequency = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        self.pluck(frequency, amplitude)

    def generate(self, num_samples: int) -> np.ndarray:
        """Generate audio samples - always returns exactly num_samples"""
        output = np.zeros(num_samples, dtype=np.float32)

        with self._lock:
            if not self.active_strings:
                return output

            # Collect IDs of strings to remove
            ids_to_remove = []

            for string in self.active_strings:
                if not string["active"]:
                    ids_to_remove.append(string["id"])
                    continue

                buffer = string["buffer"]
                period = string["period"]

                if buffer is None or period <= 0:
                    ids_to_remove.append(string["id"])
                    continue

                idx = string["index"] % period

                for i in range(num_samples):
                    sample = buffer[idx]
                    output[i] += sample

                    # Karplus-Strong filter
                    next_idx = (idx + 1) % period
                    new_sample = self.damping * 0.5 * (sample + buffer[next_idx])
                    buffer[idx] = new_sample

                    idx = next_idx
                    string["age"] += 1

                string["index"] = idx

                # Check decay
                if (
                    np.max(np.abs(buffer)) < 0.001
                    or string["age"] > self.sample_rate * 10
                ):
                    string["active"] = False
                    ids_to_remove.append(string["id"])

            # Remove dead strings by ID (not by object comparison)
            self.active_strings = [
                s for s in self.active_strings if s["id"] not in ids_to_remove
            ]

        # Normalize
        max_out = np.max(np.abs(output))
        if max_out > 1.0:
            output = output / max_out

        return output

    def set_damping(self, value: float):
        self.damping = max(0.9, min(0.9999, value))

    def set_sample_rate(self, sample_rate: int):
        with self._lock:
            self.sample_rate = sample_rate
            self.active_strings.clear()

    def clear(self):
        with self._lock:
            self.active_strings.clear()
