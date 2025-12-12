import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
)


class VUMeter(QWidget):
    """Vertical VU meter with peak hold"""

    def __init__(self, label: str = "Level", parent=None):
        super().__init__(parent)
        self.label = label
        self._level = -60.0  # dB
        self._peak = -60.0
        self._peak_hold_time = 0
        self._peak_hold_max = 30  # frames to hold peak

        self.setFixedSize(40, 150)

        # Colors for gradient
        self.color_low = QColor("#00ff88")
        self.color_mid = QColor("#ffff00")
        self.color_high = QColor("#ff0000")

    def set_level(self, linear_value: float):
        """Set level from linear amplitude (0-1)"""
        if linear_value > 0:
            self._level = 20 * np.log10(linear_value + 1e-10)
        else:
            self._level = -60.0

        # Update peak
        if self._level > self._peak:
            self._peak = self._level
            self._peak_hold_time = 0
        else:
            self._peak_hold_time += 1
            if self._peak_hold_time > self._peak_hold_max:
                self._peak -= 1.0  # Decay peak

        self._peak = max(self._peak, -60.0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            width = self.width()
            height = self.height()

            meter_x = 5
            meter_width = width - 10
            meter_top = 20
            meter_height = height - 40

            # Background
            painter.fillRect(
                meter_x, meter_top, meter_width, meter_height, QColor("#1a1a1a")
            )

            # Calculate level height (map -60 to 0 dB to 0 to meter_height)
            level_normalized = (self._level + 60) / 60
            level_normalized = max(0, min(1, level_normalized))
            level_height = int(meter_height * level_normalized)

            # Draw level gradient
            if level_height > 0:
                gradient = QLinearGradient(0, meter_top + meter_height, 0, meter_top)
                gradient.setColorAt(0.0, self.color_low)
                gradient.setColorAt(0.6, self.color_low)
                gradient.setColorAt(0.8, self.color_mid)
                gradient.setColorAt(1.0, self.color_high)

                painter.fillRect(
                    meter_x,
                    meter_top + meter_height - level_height,
                    meter_width,
                    level_height,
                    QBrush(gradient),
                )

            # Draw peak indicator
            peak_normalized = (self._peak + 60) / 60
            peak_normalized = max(0, min(1, peak_normalized))
            peak_y = int(meter_top + meter_height - meter_height * peak_normalized)

            if self._peak > -6:
                painter.setPen(QPen(self.color_high, 2))
            else:
                painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(meter_x, peak_y, meter_x + meter_width, peak_y)

            # Draw scale marks
            painter.setPen(QColor("#555555"))
            for db in [0, -6, -12, -24, -48]:
                y = int(meter_top + meter_height * (1 - (db + 60) / 60))
                painter.drawLine(meter_x - 3, y, meter_x, y)

            # Draw label
            painter.setPen(QColor("#ffffff"))
            font = QFont("Arial", 8)
            painter.setFont(font)
            painter.drawText(
                QRectF(0, height - 18, width, 18),
                Qt.AlignmentFlag.AlignCenter,
                self.label,
            )

            # Draw dB value
            painter.setPen(QColor("#888888"))
            db_text = f"{self._level:.0f}"
            painter.drawText(
                QRectF(0, 2, width, 16), Qt.AlignmentFlag.AlignCenter, db_text
            )

        finally:
            painter.end()
