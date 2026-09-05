"""
app/ui/widgets/segmented_control.py
Apple-inspired segmented pill control for switching view modes and states.
"""
from typing import Dict, Optional
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QToolButton, QButtonGroup, QWidget
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QIcon


class SegmentedControl(QFrame):
    """
    Apple HIG Segmented Control:
    A capsule/pill container holding mutually exclusive toggle buttons.
    """
    sig_segment_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("segmentedControl")
        self.setStyleSheet("""
            #segmentedControl {
                background-color: #18181B;
                border: 1px solid #2D2D32;
                border-radius: 3px;
                padding: 1px;
            }
        """)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._button_group.idClicked.connect(self._on_button_clicked)

        self._buttons: Dict[str, QToolButton] = {}
        self._key_by_id: Dict[int, str] = {}
        self._current_key: Optional[str] = None
        self._next_id = 0

    def add_segment(self, key: str, label: str, icon: Optional[QIcon] = None) -> QToolButton:
        """Adds a segment button with given unique key, text label, and optional icon."""
        btn = QToolButton(self)
        btn.setProperty("class", "segment-btn")
        btn.setText(label)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon if icon else Qt.ToolButtonStyle.ToolButtonTextOnly)

        if icon:
            btn.setIcon(icon)

        btn_id = self._next_id
        self._next_id += 1

        self._button_group.addButton(btn, btn_id)
        self._buttons[key] = btn
        self._key_by_id[btn_id] = key
        self._layout.addWidget(btn)

        if len(self._buttons) == 1:
            btn.setChecked(True)
            self._current_key = key

        return btn

    def set_selected(self, key: str):
        """Programmatically select a segment by key."""
        if key in self._buttons:
            btn = self._buttons[key]
            btn.setChecked(True)
            self._current_key = key

    def current_segment(self) -> Optional[str]:
        """Returns the currently active segment key."""
        return self._current_key

    def get_button(self, key: str) -> Optional[QToolButton]:
        """Returns the QToolButton associated with key."""
        return self._buttons.get(key)

    def buttons(self) -> list[QToolButton]:
        """Returns list of all segment buttons."""
        return list(self._buttons.values())

    def _on_button_clicked(self, btn_id: int):
        key = self._key_by_id.get(btn_id)
        if key and key != self._current_key:
            self._current_key = key
            self.sig_segment_changed.emit(key)
