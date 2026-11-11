"""
app/ui/sidebar/nav_rail.py
Compact macOS auxiliary navigation strip.
"""
from typing import Optional, Dict
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QToolButton, QButtonGroup, QWidget
from PyQt6.QtCore import pyqtSignal, Qt, QSize

from app.ui.theme.icons import get_icon


class NavRail(QFrame):
    """
    Compact vertical navigation bar providing programmatic and quick access
    to Pages, Inspector, and Settings.
    """
    sig_nav_changed = pyqtSignal(str)
    sig_toggle_sidebar = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("navRail")
        self.setFixedWidth(40)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 8, 0, 8)
        self._layout.setSpacing(4)

        # Collapse toggle button
        self._toggle_btn = QToolButton(self)
        self._toggle_btn.setIcon(get_icon("sidebar_left", color="#98989D", size=16))
        self._toggle_btn.setIconSize(QSize(16, 16))
        self._toggle_btn.setToolTip("折叠/展开侧边栏 (Ctrl+B)")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self.sig_toggle_sidebar.emit)
        self._layout.addWidget(self._toggle_btn)

        self._layout.addSpacing(6)

        # Navigation group
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._buttons: Dict[str, QToolButton] = {}

        self._add_nav_item("pages", "layers", "页面列表", checked=True)
        self._add_nav_item("inspector", "sidebar_right", "属性检查器")

        self._layout.addStretch()

        # Bottom items
        self._add_nav_item("settings", "settings", "系统设置", is_action=True)

    def _add_nav_item(self, key: str, icon_name: str, tooltip: str, checked: bool = False, is_action: bool = False):
        btn = QToolButton(self)
        btn.setIcon(get_icon(icon_name, color="#98989D", active_color="#0A84FF", size=17))
        btn.setIconSize(QSize(17, 17))
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        if not is_action:
            btn.setCheckable(True)
            btn.setChecked(checked)
            self._btn_group.addButton(btn)

        btn.clicked.connect(lambda: self.sig_nav_changed.emit(key))
        self._buttons[key] = btn
        self._layout.addWidget(btn)

    def set_active_section(self, key: str):
        """Selects a navigation section programmatically."""
        if key in self._buttons:
            self._buttons[key].setChecked(True)

    def set_collapsed_icon(self, is_collapsed: bool):
        """Updates toggle chevron orientation."""
        icon_name = "chevron_right" if is_collapsed else "chevron_left"
        self._toggle_btn.setIcon(get_icon(icon_name, color="#98989D", size=16))
