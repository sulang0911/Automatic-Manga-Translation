"""
app/ui/sidebar/nav_rail.py
Professional IDE Activity Bar (VS Code / JetBrains Fleet style).
"""
from typing import Optional, Dict
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QToolButton, QButtonGroup, QWidget
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtGui import QIcon

from app.ui.theme.icons import get_icon


class NavRail(QFrame):
    """
    Vertical icon Activity Bar providing fast access to Pages, Inspector, and Settings.
    Supports collapsing/expanding the adjacent sidebar drawer with active left border indicator.
    """
    sig_nav_changed = pyqtSignal(str)
    sig_toggle_sidebar = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("navRail")
        self.setFixedWidth(46)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 8, 0, 8)
        self._layout.setSpacing(2)

        # Collapse toggle button
        self._toggle_btn = QToolButton(self)
        self._toggle_btn.setIcon(get_icon("chevron_left", color="#A1A1AA", size=16))
        self._toggle_btn.setIconSize(QSize(16, 16))
        self._toggle_btn.setToolTip("折叠/展开侧边栏 (Explorer)")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self.sig_toggle_sidebar.emit)
        self._layout.addWidget(self._toggle_btn)

        self._layout.addSpacing(8)

        # Navigation group
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._buttons: Dict[str, QToolButton] = {}

        self._add_nav_item("pages", "layers", "页面队列 (Pages / Explorer)", checked=True)
        self._add_nav_item("inspector", "eye", "属性检查器 (Inspector)")

        self._layout.addStretch()

        # Bottom items
        self._add_nav_item("settings", "settings", "首选项设置 (Preferences)", is_action=True)

    def _add_nav_item(self, key: str, icon_name: str, tooltip: str, checked: bool = False, is_action: bool = False):
        btn = QToolButton(self)
        btn.setIcon(get_icon(icon_name, color="#A1A1AA", active_color="#007ACC", size=18))
        btn.setIconSize(QSize(18, 18))
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
        self._toggle_btn.setIcon(get_icon(icon_name, color="#A1A1AA", size=16))

