from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)


class VirtualKeyboard(QWidget):
    """Virtual keyboard for triggering Karplus-Strong notes"""

    note_triggered = pyqtSignal(str)  # Emits note name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        # Guitar string buttons (standard tuning)
        strings_layout = QHBoxLayout()

        self.string_buttons = []
        guitar_strings = [
            ("E2", "E (Low)"),
            ("A2", "A"),
            ("D3", "D"),
            ("G3", "G"),
            ("B3", "B"),
            ("E4", "E (High)"),
        ]

        for note, label in guitar_strings:
            btn = QPushButton(label)
            btn.setProperty("note", note)
            btn.clicked.connect(lambda checked, n=note: self.note_triggered.emit(n))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a4a4a;
                    color: white;
                    border: 2px solid #666666;
                    border-radius: 8px;
                    padding: 15px 10px;
                    font-size: 12px;
                    font-weight: bold;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #5a5a5a;
                    border-color: #00ff88;
                }
                QPushButton:pressed {
                    background-color: #00ff88;
                    color: #1a1a1a;
                }
            """)
            strings_layout.addWidget(btn)
            self.string_buttons.append(btn)

        layout.addLayout(strings_layout)

        # Chromatic note buttons
        chromatic_layout = QHBoxLayout()

        chromatic_notes = [
            "C3",
            "C#3",
            "D3",
            "D#3",
            "E3",
            "F3",
            "F#3",
            "G3",
            "G#3",
            "A3",
            "A#3",
            "B3",
        ]

        for note in chromatic_notes:
            btn = QPushButton(note)
            btn.setProperty("note", note)
            btn.clicked.connect(lambda checked, n=note: self.note_triggered.emit(n))

            is_sharp = "#" in note
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {"#2a2a2a" if is_sharp else "#f0f0f0"};
                    color: {"white" if is_sharp else "black"};
                    border: 1px solid #444444;
                    border-radius: 4px;
                    padding: 10px 5px;
                    font-size: 10px;
                    min-width: 35px;
                }}
                QPushButton:hover {{
                    background-color: {"#3a3a3a" if is_sharp else "#e0e0e0"};
                    border-color: #00ff88;
                }}
                QPushButton:pressed {{
                    background-color: #00ff88;
                    color: #1a1a1a;
                }}
            """)
            chromatic_layout.addWidget(btn)

        layout.addLayout(chromatic_layout)
