"""
Apple macOS Human Interface Guidelines (Sonoma / Sequoia) Design System for Legacy Desktop UI.
"""
import os


def _ensure_legacy_svg_assets():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    asset_dir = os.path.join(base_dir, "app", "ui", "theme", "assets")
    svg_files = {
        "check_white": (
            "check_white.svg",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="12" height="12">'
            '<polyline points="3.5 8.5 6.5 11.5 12.5 4.5" stroke="#FFFFFF" stroke-width="2.2" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>',
        ),
        "check_disabled": (
            "check_disabled.svg",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="12" height="12">'
            '<polyline points="3.5 8.5 6.5 11.5 12.5 4.5" stroke="#8E8E93" stroke-width="2.0" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>',
        ),
        "chevron_down_dark": (
            "chevron_down_dark.svg",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="12" height="12">'
            '<polyline points="4 6 8 10 12 6" stroke="#98989D" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>',
        ),
        "chevron_down_light": (
            "chevron_down_light.svg",
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="12" height="12">'
            '<polyline points="4 6 8 10 12 6" stroke="#6E6E73" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>',
        ),
    }
    paths = {}
    try:
        os.makedirs(asset_dir, exist_ok=True)
        for key, (fname, content) in svg_files.items():
            full_path = os.path.join(asset_dir, fname)
            if not os.path.exists(full_path):
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
            paths[key] = full_path.replace("\\", "/")
    except Exception:
        for key, (fname, _) in svg_files.items():
            paths[key] = os.path.join(asset_dir, fname).replace("\\", "/")
    return paths


_SVG_PATHS = _ensure_legacy_svg_assets()
_CHECK_WHITE_SVG = _SVG_PATHS["check_white"]
_CHECK_DISABLED_SVG = _SVG_PATHS["check_disabled"]
_CHEVRON_DARK_SVG = _SVG_PATHS["chevron_down_dark"]
_CHEVRON_LIGHT_SVG = _SVG_PATHS["chevron_down_light"]

DARK_THEME = f"""
QWidget {{
    background-color: #18181A;
    color: #F5F5F7;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif;
    font-size: 12px;
    selection-background-color: #0A84FF;
    selection-color: #FFFFFF;
}}

QMainWindow, QDialog {{
    background-color: #18181A;
}}

QLabel {{
    background-color: transparent;
    color: #F5F5F7;
}}

QLabel#headingLabel {{
    font-size: 15px;
    font-weight: 600;
    color: #F5F5F7;
    padding-bottom: 4px;
}}

QScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget,
QWidget#settingsScrollContent {{
    background-color: transparent;
    border: none;
}}

QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 6px;
    margin: 2px 1px;
}}

QScrollBar::handle:vertical {{
    background: rgba(255, 255, 255, 0.18);
    min-height: 24px;
    border-radius: 3px;
}}

QScrollBar::handle:vertical:hover {{
    background: rgba(255, 255, 255, 0.32);
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
    height: 0px;
}}

QScrollBar:horizontal {{
    border: none;
    background: transparent;
    height: 6px;
    margin: 1px 2px;
}}

QScrollBar::handle:horizontal {{
    background: rgba(255, 255, 255, 0.18);
    min-width: 24px;
    border-radius: 3px;
}}

QScrollBar::handle:horizontal:hover {{
    background: rgba(255, 255, 255, 0.32);
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
    width: 0px;
}}

QToolBar {{
    background: #1E1E22;
    border-bottom: 1px solid #2C2C30;
    padding: 4px 12px;
    spacing: 8px;
}}

QFrame#cardFrame, QFrame#sidebarCard, QFrame#inspectorCard {{
    background-color: #1C1C1F;
    border: 1px solid #2C2C30;
    border-radius: 10px;
}}

QFrame#glassHeader {{
    background-color: #202024;
    border-bottom: 1px solid #2C2C30;
}}

QPushButton {{
    background-color: #26262B;
    color: #F5F5F7;
    border: 1px solid #38383E;
    border-radius: 7px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: #2C2C31;
    border-color: #484850;
}}

QPushButton:pressed {{
    background-color: #36363C;
}}

QPushButton:disabled {{
    background-color: transparent;
    color: #636366;
    border: 1px solid #2C2C30;
}}

QPushButton#primaryBtn, QPushButton[class="primaryBtn"] {{
    background-color: #0A84FF;
    color: #FFFFFF;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 5px 14px;
    font-weight: 600;
}}

QPushButton#primaryBtn:hover, QPushButton[class="primaryBtn"]:hover {{
    background-color: #0071E3;
}}

QPushButton#dangerBtn {{
    background-color: rgba(255, 69, 58, 0.14);
    color: #FF453A;
    border: 1px solid rgba(255, 69, 58, 0.28);
    border-radius: 7px;
}}

QPushButton#dangerBtn:hover {{
    background-color: rgba(255, 69, 58, 0.24);
}}

QFrame#segmentedControl {{
    background-color: rgba(0, 0, 0, 0.28);
    border: 1px solid #2C2C30;
    border-radius: 8px;
    padding: 2px;
}}

QPushButton[class="segmentedItem"] {{
    background-color: transparent;
    color: #98989D;
    border: none;
    border-radius: 6px;
    padding: 4px 11px;
    font-size: 12px;
    font-weight: 500;
}}

QPushButton[class="segmentedItem"]:hover {{
    color: #F5F5F7;
}}

QPushButton[class="segmentedItem"]:checked {{
    background-color: #0A84FF;
    color: #FFFFFF;
    font-weight: 600;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: #141417;
    color: #F5F5F7;
    border: 1px solid #38383E;
    border-radius: 7px;
    padding: 5px 9px;
    font-size: 12px;
    min-height: 20px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: #0A84FF;
    background-color: #1C1C21;
}}

QComboBox {{
    background-color: #26262B;
    color: #F5F5F7;
    border: 1px solid #38383E;
    border-radius: 7px;
    padding: 5px 28px 5px 10px;
    min-height: 22px;
    font-size: 12px;
}}

QComboBox:hover, QComboBox:focus {{
    background-color: #2C2C31;
    border-color: #0A84FF;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border: none;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: url("{_CHEVRON_DARK_SVG}");
    width: 12px;
    height: 12px;
}}

QComboBoxPrivateContainer,
QComboBox QAbstractItemView,
QComboBox QListView {{
    background-color: #202024;
    color: #F5F5F7;
    border: 1px solid #38383E;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #0A84FF;
    selection-color: #FFFFFF;
    outline: none;
}}

QComboBox QAbstractItemView::item,
QComboBox QListView::item {{
    background-color: #202024;
    color: #F5F5F7;
    min-height: 26px;
    padding: 4px 10px;
    border-radius: 5px;
}}

QComboBox QAbstractItemView::item:hover,
QComboBox QListView::item:hover {{
    background-color: #2C2C31;
}}

QComboBox QAbstractItemView::item:selected,
QComboBox QListView::item:selected {{
    background-color: #0A84FF;
    color: #FFFFFF;
}}

QGroupBox {{
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid #2C2C30;
    border-radius: 10px;
    margin-top: 22px;
    padding: 14px 12px 12px 12px;
    color: #F5F5F7;
    font-size: 12px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 6px;
    top: 2px;
    padding: 0px 4px;
    color: #98989D;
    font-size: 12px;
    font-weight: 600;
}}

QCheckBox {{
    spacing: 8px;
    font-size: 12px;
    color: #F5F5F7;
    background-color: transparent;
    min-height: 20px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #38383E;
    background-color: #141417;
}}

QCheckBox::indicator:hover:unchecked {{
    border-color: #0A84FF;
    background-color: #2C2C31;
}}

QCheckBox::indicator:checked {{
    background-color: #0A84FF;
    border-color: #0A84FF;
    image: url("{_CHECK_WHITE_SVG}");
}}

QCheckBox::indicator:checked:disabled {{
    background-color: #36363C;
    border-color: #2C2C30;
    image: url("{_CHECK_DISABLED_SVG}");
}}

QSlider:horizontal {{
    min-height: 22px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: #2C2C30;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: #0A84FF;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 2px solid #0A84FF;
    width: 12px;
    height: 12px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
}}

QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
}}

QListWidget::item {{
    border-radius: 8px;
    padding: 5px 8px;
    margin: 1px 2px;
}}

QListWidget::item:hover {{
    background-color: #2C2C31;
}}

QListWidget::item:selected {{
    background-color: rgba(10, 132, 255, 0.16);
    color: #0A84FF;
    font-weight: 600;
}}

QTabWidget::pane {{
    border: none;
    background-color: transparent;
}}

QTabBar::tab {{
    background-color: transparent;
    color: #98989D;
    border: none;
    border-radius: 6px;
    padding: 5px 12px;
    margin: 2px;
    font-size: 12px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: #36363C;
    color: #F5F5F7;
    font-weight: 600;
}}
"""

LIGHT_THEME = f"""
QWidget {{
    background-color: #F5F5F7;
    color: #1D1D1F;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif;
    font-size: 12px;
    selection-background-color: #0071E3;
    selection-color: #FFFFFF;
}}

QMainWindow, QDialog {{
    background-color: #F5F5F7;
}}

QLabel {{
    background-color: transparent;
    color: #1D1D1F;
}}

QLabel#headingLabel {{
    font-size: 15px;
    font-weight: 600;
    color: #1D1D1F;
    padding-bottom: 4px;
}}

QScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget,
QWidget#settingsScrollContent {{
    background-color: transparent;
    border: none;
}}

QFrame#cardFrame, QFrame#sidebarCard, QFrame#inspectorCard {{
    background-color: #FFFFFF;
    border: 1px solid #E2E2E7;
    border-radius: 10px;
}}

QFrame#glassHeader {{
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E2E7;
}}

QPushButton {{
    background-color: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 7px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: #E5E5EA;
}}

QPushButton#primaryBtn, QPushButton[class="primaryBtn"] {{
    background-color: #0071E3;
    color: #FFFFFF;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 5px 14px;
    font-weight: 600;
}}

QPushButton#primaryBtn:hover, QPushButton[class="primaryBtn"]:hover {{
    background-color: #0058B6;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid #D1D1D6;
    border-radius: 7px;
    padding: 5px 9px;
    font-size: 12px;
    min-height: 20px;
}}

QComboBox {{
    background-color: #FFFFFF;
    color: #1D1D1F;
    border: 1px solid #D1D1D6;
    border-radius: 7px;
    padding: 5px 28px 5px 10px;
    min-height: 22px;
    font-size: 12px;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border: none;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: url("{_CHEVRON_LIGHT_SVG}");
    width: 12px;
    height: 12px;
}}

QGroupBox {{
    background-color: #FFFFFF;
    border: 1px solid #E2E2E7;
    border-radius: 10px;
    margin-top: 22px;
    padding: 14px 12px 12px 12px;
    color: #1D1D1F;
    font-size: 12px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 6px;
    top: 2px;
    padding: 0px 4px;
    color: #6E6E73;
    font-size: 12px;
    font-weight: 600;
}}

QCheckBox {{
    spacing: 8px;
    font-size: 12px;
    color: #1D1D1F;
    background-color: transparent;
    min-height: 20px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #D1D1D6;
    background-color: #FFFFFF;
}}

QCheckBox::indicator:checked {{
    background-color: #0071E3;
    border-color: #0071E3;
    image: url("{_CHECK_WHITE_SVG}");
}}
"""


def get_stylesheet(theme: str = "dark") -> str:
    if isinstance(theme, str) and theme.lower() == "light":
        return LIGHT_THEME
    return DARK_THEME
