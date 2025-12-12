import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal, Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QConicalGradient


class Knob(QWidget):
    """Rotary knob control widget"""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str = "Knob",
        min_val: float = 0.0,
        max_val: float = 1.0,
        default: float = 0.5,
        parent=None,
    ):
        super().__init__(parent)

        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self._value = default

        # Knob appearance
        self.knob_radius = 30
        self.setFixedSize(80, 100)

        # Interaction state
        self._dragging = False
        self._last_y = 0

        # Colors
        self.knob_color = QColor("#3a3a3a")
        self.indicator_color = QColor("#00ff88")
        self.arc_bg_color = QColor("#1a1a1a")
        self.arc_fg_color = QColor("#00ff88")
        self.label_color = QColor("#ffffff")

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, val: float):
        val = max(self.min_val, min(self.max_val, val))
        if val != self._value:
            self._value = val
            self.valueChanged.emit(self._value)
            self.update()

    def normalized_value(self) -> float:
        """Returns value normalized to 0-1 range"""
        return (self._value - self.min_val) / (self.max_val - self.min_val)

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            width = self.width()
            center_x = width // 2
            center_y = 45  # Knob center

            # Draw arc background (270 degrees, from -225 to 45)
            arc_rect = QRectF(
                center_x - self.knob_radius - 5,
                center_y - self.knob_radius - 5,
                (self.knob_radius + 5) * 2,
                (self.knob_radius + 5) * 2,
            )

            pen = QPen(self.arc_bg_color, 4)
            painter.setPen(pen)
            painter.drawArc(arc_rect, -225 * 16, 270 * 16)

            # Draw arc foreground (value indicator)
            pen = QPen(self.arc_fg_color, 4)
            painter.setPen(pen)
            span_angle = int(270 * self.normalized_value())
            painter.drawArc(arc_rect, -225 * 16, span_angle * 16)

            # Draw knob body
            knob_gradient = QConicalGradient(center_x, center_y, -45)
            knob_gradient.setColorAt(0.0, QColor("#4a4a4a"))
            knob_gradient.setColorAt(0.5, QColor("#2a2a2a"))
            knob_gradient.setColorAt(1.0, QColor("#4a4a4a"))

            painter.setPen(QPen(QColor("#1a1a1a"), 2))
            painter.setBrush(QBrush(knob_gradient))
            painter.drawEllipse(
                center_x - self.knob_radius,
                center_y - self.knob_radius,
                self.knob_radius * 2,
                self.knob_radius * 2,
            )

            # Draw indicator line
            angle = np.radians(-225 + 270 * self.normalized_value())
            indicator_length = self.knob_radius - 8
            end_x = center_x + indicator_length * np.cos(angle)
            end_y = center_y + indicator_length * np.sin(angle)

            pen = QPen(
                self.indicator_color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
            )
            painter.setPen(pen)
            painter.drawLine(
                int(center_x + 8 * np.cos(angle)),
                int(center_y + 8 * np.sin(angle)),
                int(end_x),
                int(end_y),
            )

            # Draw label
            painter.setPen(self.label_color)
            font = QFont("Arial", 9)
            painter.setFont(font)
            painter.drawText(
                QRectF(0, 78, width, 20), Qt.AlignmentFlag.AlignCenter, self.label
            )

            # Draw value
            font = QFont("Arial", 8)
            painter.setFont(font)
            painter.setPen(QColor("#888888"))
            display_val = (
                f"{self._value:.1f}" if self.max_val > 10 else f"{self._value:.2f}"
            )
            painter.drawText(
                QRectF(0, center_y - 8, width, 16),
                Qt.AlignmentFlag.AlignCenter,
                display_val,
            )
        finally:
            painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_y = event.position().y()
            self.setCursor(Qt.CursorShape.BlankCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.unsetCursor()

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = self._last_y - event.position().y()
            self._last_y = event.position().y()

            # Sensitivity based on range
            sensitivity = (self.max_val - self.min_val) / 100
            self.value = self._value + delta * sensitivity

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120  # Standard wheel step
        sensitivity = (self.max_val - self.min_val) / 20
        self.value = self._value + delta * sensitivity
