"""
app/ui/widgets/card.py
Apple HIG Card container widget with crisp 1px borders and rounded corners.
"""
from typing import Optional
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PyQt6.QtCore import Qt


class CardWidget(QFrame):
    """
    An elegant, elevated card container adhering to Apple Human Interface Guidelines.
    Features subtle 1px border, 10-12px rounded corners, and optional title/subtitle header.
    """

    def __init__(self, title: Optional[str] = None, subtitle: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setObjectName("cardWidget")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)

        self._header_layout: Optional[QHBoxLayout] = None
        self._title_label: Optional[QLabel] = None
        self._subtitle_label: Optional[QLabel] = None

        if title or subtitle:
            self._setup_header(title, subtitle)

    def _setup_header(self, title: Optional[str], subtitle: Optional[str]):
        header_widget = QWidget(self)
        self._header_layout = QHBoxLayout(header_widget)
        self._header_layout.setContentsMargins(0, 0, 0, 4)
        self._header_layout.setSpacing(6)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)

        if title:
            self._title_label = QLabel(title, header_widget)
            self._title_label.setStyleSheet("font-weight: 600; font-size: 13px;")
            title_col.addWidget(self._title_label)

        if subtitle:
            self._subtitle_label = QLabel(subtitle, header_widget)
            self._subtitle_label.setStyleSheet("font-size: 11px; opacity: 0.7;")
            title_col.addWidget(self._subtitle_label)

        self._header_layout.addLayout(title_col)
        self._header_layout.addStretch()
        self._layout.addWidget(header_widget)

    def set_content_widget(self, widget: QWidget):
        """Sets the primary content widget of the card."""
        self._layout.addWidget(widget)

    def content_layout(self) -> QVBoxLayout:
        """Returns the internal QVBoxLayout for adding child widgets."""
        return self._layout
