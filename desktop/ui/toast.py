from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QColor

class Toast(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(16, 10, 16, 10)
        self.layout.setSpacing(10)

        self.icon_label = QLabel("✨", self)
        self.icon_label.setStyleSheet("font-size: 15px; background: transparent;")
        self.layout.addWidget(self.icon_label)

        self.msg_label = QLabel("", self)
        self.msg_label.setStyleSheet("color: #FFFFFF; font-size: 13px; font-weight: 500; background: transparent;")
        self.layout.addWidget(self.msg_label)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 32, 0.94);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 20px;
            }
        """)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)

        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._fade_out)

    def show_message(self, message: str, toast_type: str = "info", duration_ms: int = 2500):
        icons = {
            "success": "✓",
            "error": "✕",
            "warning": "⚠",
            "info": "ℹ"
        }
        colors = {
            "success": "#30D158",
            "error": "#FF453A",
            "warning": "#FF9F0A",
            "info": "#0A84FF"
        }
        self.icon_label.setText(icons.get(toast_type, "ℹ"))
        self.icon_label.setStyleSheet(f"color: {colors.get(toast_type, '#0A84FF')}; font-size: 15px; font-weight: bold; background: transparent;")
        self.msg_label.setText(message)
        self.adjustSize()

        if self.parent():
            parent_rect = self.parent().rect()
            x = (parent_rect.width() - self.width()) // 2
            y = parent_rect.height() - self.height() - 40
            self.move(x, y)

        self.show()
        self.raise_()

        # Fade in animation
        self.anim_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim_in.setDuration(220)
        self.anim_in.setStartValue(0.0)
        self.anim_in.setEndValue(1.0)
        self.anim_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim_in.start()

        self.hide_timer.start(duration_ms)

    def _fade_out(self):
        self.anim_out = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim_out.setDuration(280)
        self.anim_out.setStartValue(1.0)
        self.anim_out.setEndValue(0.0)
        self.anim_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self.anim_out.finished.connect(self.hide)
        self.anim_out.start()
