"""
app/ui/widgets/progress_pill.py
Minimalist Apple-style status indicator pill with status dot and progress bar.
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt


class StatusDot(QWidget):
    """Circular status indicator dot."""

    def __init__(self, color: str = "#10B981", size: int = 8, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._color = color
        self._size = size
        self.setFixedSize(size, size)

    def set_color(self, color: str):
        self._color = color
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QBrush, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(self._color)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self._size, self._size)


class ProgressPill(QWidget):
    """
    Apple HIG Status & Progress Pill.
    Combines status dot, textual description, and miniature progress bar.
    """
    COLORS = {
        "idle": "#71717A",
        "ready": "#10B981",
        "processing": "#3B82F6",
        "warning": "#F59E0B",
        "completed": "#10B981",
        "failed": "#EF4444",
        "error": "#EF4444",
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("progressPill")
        self.setStyleSheet("""
            #progressPill {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 2px 8px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        self._dot = StatusDot(color=self.COLORS["ready"], size=8, parent=self)
        layout.addWidget(self._dot)

        self._label = QLabel("就绪", self)
        self._label.setStyleSheet("font-size: 12px; font-weight: 500;")
        layout.addWidget(self._label)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setFixedSize(60, 6)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

    def set_status(self, state: str, text: str, progress: int = 0):
        """Updates the status pill state, message text, and progress bar value."""
        color = self.COLORS.get(state.lower(), self.COLORS["idle"])
        self._dot.set_color(color)
        self._label.setText(text)

        if state.lower() == "processing":
            self._progress_bar.show()
            self._progress_bar.setValue(max(0, min(100, progress)))
        else:
            self._progress_bar.hide()

    def text(self) -> str:
        return self._label.text()

    def is_progress_visible(self) -> bool:
        return not self._progress_bar.isHidden()
