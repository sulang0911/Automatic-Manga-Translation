"""
app/ui/widgets/drag_overlay.py
Full-window drag & drop visual overlay adhering to Apple HIG.
Provides responsive visual feedback whenever files/folders are dragged over the window.
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPainter

from app.ui.theme.icons import render_svg_pixmap


class DragDropOverlay(QWidget):
    """
    Frosted-glass translucent overlay displayed during drag-and-drop operations.
    Shows animated drop target with clear format guidelines.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setWindowFlags(Qt.WindowType.SubWindow)
        self.hide()

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Center Card
        self.card = QFrame(self)
        self.card.setObjectName("dragCard")
        self.card.setStyleSheet("""
            #dragCard {
                background-color: rgba(24, 24, 28, 0.92);
                border: 2px dashed #0A84FF;
                border-radius: 16px;
                padding: 36px 48px;
            }
        """)

        card_layout = QVBoxLayout(self.card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(12)

        # Upload icon
        self.icon_lbl = QLabel(self.card)
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = render_svg_pixmap("upload_cloud", color="#0A84FF", size=56)
        self.icon_lbl.setPixmap(pix)
        card_layout.addWidget(self.icon_lbl)

        # Main prompt
        self.title_lbl = QLabel("释放以导入漫画文件或文件夹", self.card)
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_lbl.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: 600;")
        card_layout.addWidget(self.title_lbl)

        # Subtitle formats
        self.subtitle_lbl = QLabel("支持多选导入 JPG · PNG · WEBP · ZIP 压缩包 · 章节文件夹", self.card)
        self.subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_lbl.setStyleSheet("color: #A1A1AA; font-size: 12px;")
        card_layout.addWidget(self.subtitle_lbl)

        main_layout.addWidget(self.card)

    def paintEvent(self, event):
        """Draws semi-transparent backdrop."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 140))
        painter.end()

    def show_overlay(self):
        """Displays the overlay matching parent geometry."""
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def hide_overlay(self):
        """Hides the overlay."""
        self.hide()
