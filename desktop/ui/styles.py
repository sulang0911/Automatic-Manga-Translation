"""
Apple macOS Human Interface Guidelines (Sonoma / Sequoia) Design System for Legacy Desktop UI.
"""

DARK_THEME = """
QWidget {
    background-color: #18181A;
    color: #F5F5F7;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif;
    font-size: 12px;
    selection-background-color: #0A84FF;
    selection-color: #FFFFFF;
}

QMainWindow, QDialog {
    background-color: #18181A;
}

QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 6px;
    margin: 2px 1px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.18);
    min-height: 24px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 0.32);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: transparent;
    height: 6px;
    margin: 1px 2px;
}

QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 0.18);
    min-width: 24px;
    border-radius: 3px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(255, 255, 255, 0.32);
}

QToolBar {
    background: #1E1E22;
    border-bottom: 1px solid #2C2C30;
    padding: 4px 12px;
    spacing: 8px;
}

QFrame#cardFrame, QFrame#sidebarCard, QFrame#inspectorCard {
    background-color: #1C1C1F;
    border: 1px solid #2C2C30;
    border-radius: 10px;
}

QFrame#glassHeader {
    background-color: #202024;
    border-bottom: 1px solid #2C2C30;
}

QPushButton {
    background-color: rgba(255, 255, 255, 0.07);
    color: #F5F5F7;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 7px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2C2C31;
    border-color: #3A3A40;
}

QPushButton:pressed {
    background-color: #36363C;
}

QPushButton:disabled {
    background-color: transparent;
    color: #636366;
    border: 1px solid #2C2C30;
}

QPushButton#primaryBtn {
    background-color: #0A84FF;
    color: #FFFFFF;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 5px 14px;
    font-weight: 600;
}

QPushButton#primaryBtn:hover {
    background-color: #0071E3;
}

QPushButton#dangerBtn {
    background-color: rgba(255, 69, 58, 0.14);
    color: #FF453A;
    border: 1px solid rgba(255, 69, 58, 0.28);
    border-radius: 7px;
}

QPushButton#dangerBtn:hover {
    background-color: rgba(255, 69, 58, 0.24);
}

QFrame#segmentedControl {
    background-color: rgba(0, 0, 0, 0.28);
    border: 1px solid #2C2C30;
    border-radius: 8px;
    padding: 2px;
}

QPushButton[class="segmentedItem"] {
    background-color: transparent;
    color: #98989D;
    border: none;
    border-radius: 6px;
    padding: 4px 11px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton[class="segmentedItem"]:hover {
    color: #F5F5F7;
}

QPushButton[class="segmentedItem"]:checked {
    background-color: #0A84FF;
    color: #FFFFFF;
    font-weight: 600;
}

QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: rgba(255, 255, 255, 0.05);
    color: #F5F5F7;
    border: 1px solid #2C2C30;
    border-radius: 7px;
    padding: 5px 9px;
    font-size: 12px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
    border-color: #0A84FF;
    background-color: rgba(255, 255, 255, 0.08);
}

QListWidget {
    background-color: transparent;
    border: none;
    outline: none;
}

QListWidget::item {
    border-radius: 8px;
    padding: 5px 8px;
    margin: 1px 2px;
}

QListWidget::item:hover {
    background-color: #2C2C31;
}

QListWidget::item:selected {
    background-color: rgba(10, 132, 255, 0.16);
    color: #0A84FF;
    font-weight: 600;
}

QTabWidget::pane {
    border: none;
    background-color: transparent;
}

QTabBar::tab {
    background-color: transparent;
    color: #98989D;
    border: none;
    border-radius: 6px;
    padding: 5px 12px;
    margin: 2px;
    font-size: 12px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #36363C;
    color: #F5F5F7;
    font-weight: 600;
}
"""

LIGHT_THEME = """
QWidget {
    background-color: #F5F5F7;
    color: #1D1D1F;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif;
    font-size: 12px;
    selection-background-color: #0071E3;
    selection-color: #FFFFFF;
}

QMainWindow, QDialog {
    background-color: #F5F5F7;
}

QFrame#cardFrame, QFrame#sidebarCard, QFrame#inspectorCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E2E7;
    border-radius: 10px;
}

QFrame#glassHeader {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E2E7;
}

QPushButton {
    background-color: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid rgba(0, 0, 0, 0.10);
    border-radius: 7px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #E5E5EA;
}

QPushButton#primaryBtn {
    background-color: #0071E3;
    color: #FFFFFF;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 5px 14px;
    font-weight: 600;
}

QPushButton#primaryBtn:hover {
    background-color: #0058B6;
}
"""


def get_stylesheet(theme: str = "dark") -> str:
    if isinstance(theme, str) and theme.lower() == "light":
        return LIGHT_THEME
    return DARK_THEME
