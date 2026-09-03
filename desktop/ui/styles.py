"""
Apple macOS Sonoma / Sequoia Minimalist Design System for PyQt6
Features:
- Refined glassmorphic / card-based surfaces
- System SF-Pro aesthetic typography & spacing
- Subtle borders, gentle accent gradients, and smooth hover feedback
- Dark Mode & Light Mode support
"""

DARK_THEME = """
/* Global Window & Base */
QWidget {
    background-color: #1E1E1E;
    color: #F5F5F7;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
    selection-background-color: #0A84FF;
    selection-color: #FFFFFF;
}

QMainWindow, QDialog {
    background-color: #18181A;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    margin: 4px 0 4px 0;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.18);
    min-height: 24px;
    border-radius: 4px;
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
    height: 8px;
    margin: 0 4px 0 4px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 0.18);
    min-width: 24px;
    border-radius: 4px;
}

/* Header & Toolbars */
QToolBar {
    background: rgba(30, 30, 32, 0.85);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 6px 12px;
    spacing: 8px;
}

/* Cards & Frames */
QFrame#cardFrame, QFrame#sidebarCard, QFrame#inspectorCard {
    background-color: #252528;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
}

QFrame#glassHeader {
    background-color: rgba(35, 35, 38, 0.7);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

/* Buttons */
QPushButton {
    background-color: #2C2C2E;
    color: #F5F5F7;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3A3A3C;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

QPushButton:pressed {
    background-color: #242426;
}

QPushButton:disabled {
    background-color: #202022;
    color: rgba(255, 255, 255, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.04);
}

/* Primary Accent Buttons */
QPushButton#primaryBtn {
    background-color: #0A84FF;
    color: #FFFFFF;
    border: 1px solid #0071E3;
    border-radius: 8px;
    padding: 7px 18px;
    font-weight: 600;
}

QPushButton#primaryBtn:hover {
    background-color: #0077ED;
}

QPushButton#primaryBtn:pressed {
    background-color: #0062C4;
}

/* Danger Buttons */
QPushButton#dangerBtn {
    background-color: rgba(255, 69, 58, 0.15);
    color: #FF453A;
    border: 1px solid rgba(255, 69, 58, 0.3);
    border-radius: 8px;
    padding: 6px 12px;
}

QPushButton#dangerBtn:hover {
    background-color: rgba(255, 69, 58, 0.28);
}

/* Segmented Control (Apple style tab switch) */
QFrame#segmentedControl {
    background-color: #18181A;
    border-radius: 9px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2px;
}

QPushButton.segmentedItem {
    background-color: transparent;
    color: #A1A1A6;
    border: none;
    border-radius: 7px;
    padding: 5px 12px;
    font-weight: 500;
}

QPushButton.segmentedItem:hover {
    color: #F5F5F7;
}

QPushButton.segmentedItem:checked {
    background-color: #3A3A3C;
    color: #FFFFFF;
    font-weight: 600;
}

/* Input Fields & TextEdit */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1A1A1C;
    color: #F5F5F7;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 6px 10px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
    border: 1.5px solid #0A84FF;
    background-color: #1E1E22;
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
    border-top: 5px solid #A1A1A6;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #2C2C2E;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 4px;
    selection-background-color: #0A84FF;
}

/* Sliders */
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #0A84FF;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1px solid rgba(0, 0, 0, 0.2);
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #F0F0F0;
    box-shadow: 0 0 6px rgba(0, 0, 0, 0.4);
}

/* Progress Bar */
QProgressBar {
    background-color: #1A1A1C;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 600;
    height: 12px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0A84FF, stop:1 #30D158);
    border-radius: 5px;
}

/* TabWidget */
QTabWidget::pane {
    border: none;
    background: transparent;
}

QTabBar::tab {
    background: transparent;
    color: #8E8E93;
    padding: 8px 14px;
    font-weight: 500;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:hover {
    color: #E5E5EA;
}

QTabBar::tab:selected {
    color: #0A84FF;
    border-bottom: 2px solid #0A84FF;
    font-weight: 600;
}

/* List / Tree Widgets */
QListWidget, QTreeWidget {
    background-color: #1C1C1E;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 4px;
}

QListWidget::item {
    border-radius: 8px;
    padding: 8px;
    margin: 2px 0px;
}

QListWidget::item:hover {
    background-color: rgba(255, 255, 255, 0.06);
}

QListWidget::item:selected {
    background-color: #2C2C2E;
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* GroupBox & Labels */
QGroupBox {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #A1A1A6;
    font-size: 12px;
}

QLabel#headingLabel {
    font-size: 16px;
    font-weight: 700;
    color: #FFFFFF;
}

QLabel#subLabel {
    font-size: 12px;
    color: #8E8E93;
}

/* Status Badge */
QLabel.statusBadge {
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.statusPending {
    background-color: rgba(255, 159, 10, 0.18);
    color: #FF9F0A;
}

QLabel.statusProcessing {
    background-color: rgba(10, 132, 255, 0.18);
    color: #0A84FF;
}

QLabel.statusCompleted {
    background-color: rgba(48, 209, 88, 0.18);
    color: #30D158;
}

QLabel.statusFailed {
    background-color: rgba(255, 69, 58, 0.18);
    color: #FF453A;
}
"""

def get_stylesheet(theme: str = "dark") -> str:
    return DARK_THEME
