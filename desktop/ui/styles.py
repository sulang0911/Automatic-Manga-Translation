"""
Professional IDE Developer Design System for PyQt6 (VS Code / JetBrains / Linear Style)
Features:
- Dark charcoal / zinc surface palette (#18181B, #121214, #1E1E22)
- Crisp 1px hairline panel borders (#2D2D32)
- Tight 3px - 4px precision corner radius
- Developer typography with JetBrains Mono / Cascadia Code monospace support
- Modern IDE electric blue accent (#007ACC)
"""

DARK_THEME = """
/* Global Window & Base */
QWidget {
    background-color: #18181B;
    color: #ECECEF;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
    selection-background-color: #007ACC;
    selection-color: #FFFFFF;
}

QMainWindow, QDialog {
    background-color: #18181B;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.14);
    min-height: 20px;
    border-radius: 2px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 0.28);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: transparent;
    height: 8px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 0.14);
    min-width: 20px;
    border-radius: 2px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(255, 255, 255, 0.28);
}

/* Header & Toolbars */
QToolBar {
    background: #1E1E22;
    border-bottom: 1px solid #2D2D32;
    padding: 3px 8px;
    spacing: 6px;
}

/* Cards & Frames */
QFrame#cardFrame, QFrame#sidebarCard, QFrame#inspectorCard {
    background-color: #1E1E22;
    border: 1px solid #2D2D32;
    border-radius: 4px;
}

QFrame#glassHeader {
    background-color: #1E1E22;
    border-bottom: 1px solid #2D2D32;
}

/* Buttons */
QPushButton {
    background-color: #26262B;
    color: #ECECEF;
    border: 1px solid #2D2D32;
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #303036;
    border-color: #007ACC;
}

QPushButton:pressed {
    background-color: rgba(0, 122, 204, 0.16);
    border-color: #007ACC;
}

QPushButton:disabled {
    background-color: #18181B;
    color: #71717A;
    border: 1px solid transparent;
}

/* Primary Accent Buttons */
QPushButton#primaryBtn {
    background-color: #007ACC;
    color: #FFFFFF;
    border: 1px solid #0062A3;
    border-radius: 3px;
    padding: 5px 14px;
    font-weight: 600;
}

QPushButton#primaryBtn:hover {
    background-color: #0062A3;
    border-color: #0062A3;
}

QPushButton#primaryBtn:pressed {
    background-color: #0062A3;
}

/* Danger Buttons */
QPushButton#dangerBtn {
    background-color: rgba(244, 63, 94, 0.15);
    color: #F43F5E;
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-radius: 3px;
    padding: 4px 10px;
}

QPushButton#dangerBtn:hover {
    background-color: rgba(244, 63, 94, 0.28);
}

/* Segmented Control (IDE style tab switch) */
QFrame#segmentedControl {
    background-color: #18181B;
    border-radius: 3px;
    border: 1px solid #2D2D32;
    padding: 1px;
}

QPushButton.segmentedItem {
    background-color: transparent;
    color: #A1A1AA;
    border: none;
    border-radius: 3px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 500;
}

QPushButton.segmentedItem:hover {
    color: #ECECEF;
    background-color: #26262B;
}

QPushButton.segmentedItem:checked {
    background-color: #007ACC;
    color: #FFFFFF;
    font-weight: 600;
}

/* Input Fields & TextEdit */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #18181B;
    color: #ECECEF;
    border: 1px solid #2D2D32;
    border-radius: 3px;
    padding: 5px 8px;
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
    border: 1px solid #007ACC;
    background-color: #101012;
}

/* Dropdown ComboBox */
QComboBox {
    padding-right: 20px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: none;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #A1A1AA;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background-color: #1E1E22;
    border: 1px solid #2D2D32;
    border-radius: 3px;
    padding: 2px;
    selection-background-color: #007ACC;
    selection-color: #FFFFFF;
}

/* Sliders */
QSlider::groove:horizontal {
    border: none;
    height: 3px;
    background: #2D2D32;
    border-radius: 1px;
}

QSlider::sub-page:horizontal {
    background: #007ACC;
    border-radius: 1px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #007ACC;
    width: 12px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 6px;
}

QSlider::handle:horizontal:hover {
    background: #F0F0F0;
}

/* Progress Bar */
QProgressBar {
    background-color: #18181B;
    border: 1px solid #2D2D32;
    border-radius: 2px;
    text-align: center;
    color: #ECECEF;
    font-size: 9px;
    font-family: "JetBrains Mono", "Cascadia Code", monospace;
    font-weight: 600;
    height: 8px;
}

QProgressBar::chunk {
    background: #007ACC;
    border-radius: 1px;
}

/* TabWidget */
QTabWidget::pane {
    border: 1px solid #2D2D32;
    background: #1E1E22;
}

QTabBar::tab {
    background: #18181B;
    color: #A1A1AA;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 500;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:hover {
    color: #ECECEF;
    background: #26262B;
}

QTabBar::tab:selected {
    color: #ECECEF;
    background: #1E1E22;
    border-bottom: 2px solid #007ACC;
    font-weight: 600;
}

/* List / Tree Widgets */
QListWidget, QTreeWidget {
    background-color: transparent;
    border: none;
    outline: none;
    padding: 2px;
}

QListWidget::item {
    border-radius: 3px;
    padding: 6px 8px;
    margin: 1px 2px;
    border: 1px solid transparent;
}

QListWidget::item:hover {
    background-color: #26262B;
}

QListWidget::item:selected {
    background-color: rgba(0, 122, 204, 0.16);
    color: #ECECEF;
    border: 1px solid #007ACC;
}

/* GroupBox & Labels */
QGroupBox {
    border: 1px solid #2D2D32;
    border-radius: 3px;
    margin-top: 12px;
    padding-top: 8px;
    font-size: 11px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: #ECECEF;
    font-size: 11px;
}

QLabel#headingLabel {
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
}

QLabel#subLabel {
    font-size: 11px;
    color: #71717A;
}

/* Status Badge */
QLabel.statusBadge {
    border-radius: 3px;
    padding: 2px 6px;
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    font-weight: 600;
}

QLabel.statusPending {
    background-color: rgba(234, 179, 8, 0.18);
    color: #EAB308;
    border: 1px solid rgba(234, 179, 8, 0.3);
}

QLabel.statusProcessing {
    background-color: rgba(0, 122, 204, 0.18);
    color: #007ACC;
    border: 1px solid rgba(0, 122, 204, 0.3);
}

QLabel.statusCompleted {
    background-color: rgba(34, 197, 94, 0.18);
    color: #22C55E;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

QLabel.statusFailed {
    background-color: rgba(244, 63, 94, 0.18);
    color: #F43F5E;
    border: 1px solid rgba(244, 63, 94, 0.3);
}
"""

def get_stylesheet(theme: str = "dark") -> str:
    return DARK_THEME

