"""
app/ui/theme/tokens.py
Apple macOS Human Interface Guidelines (Sonoma / Sequoia) Design Tokens and Stylesheet Generator.
Supports macOS Dark Vibrancy (#18181B) and macOS Studio Light (#F8F9FA) themes.
"""
import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ThemeTokens:
    name: str
    bg_base: str
    bg_sidebar: str
    bg_surface: str
    bg_surface_hover: str
    bg_surface_active: str
    border_subtle: str
    border_focus: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent_primary: str
    accent_hover: str
    accent_subtle: str
    status_success: str
    status_warning: str
    status_error: str
    canvas_bg: str
    radius_sm: int = 7
    radius_md: int = 10
    radius_lg: int = 12
    radius_pill: int = 9999
    border_width: int = 1
    font_family: str = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif'
    font_mono: str = '"SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace'


DARK_TOKENS = ThemeTokens(
    name="dark",
    bg_base="#18181B",            # macOS dark window background
    bg_sidebar="#1C1C1F",         # Soft translucent-feel sidebar background
    bg_surface="#202024",         # Elevated toolbar, cards & inspector surface
    bg_surface_hover="#2C2C31",   # Gentle macOS control hover
    bg_surface_active="#36363C",  # Active / pressed pill surface
    border_subtle="#2C2C30",      # Hairline divider border
    border_focus="#3B82F6",       # Apple System Blue focus ring
    text_primary="#F5F5F7",       # Apple primary alabaster label
    text_secondary="#98989D",     # Apple secondary label
    text_muted="#636366",         # Apple tertiary / placeholder label
    accent_primary="#3B82F6",     # System Blue
    accent_hover="#2563EB",       # System Blue hover
    accent_subtle="rgba(59, 130, 246, 0.16)",
    status_success="#30D158",     # Apple System Green
    status_warning="#FF9F0A",     # Apple System Orange
    status_error="#FF453A",       # Apple System Red
    canvas_bg="#121214",          # Deep neutral canvas backdrop
    radius_sm=7,                  # macOS button & input corner radius
    radius_md=10,                 # macOS card & group corner radius
    radius_lg=12,                 # macOS window / modal radius
    radius_pill=9999,
    border_width=1,
    font_family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif',
    font_mono='"SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace',
)

LIGHT_TOKENS = ThemeTokens(
    name="light",
    bg_base="#F8F9FA",            # macOS light window background
    bg_sidebar="#F2F2F7",         # macOS Finder sidebar light gray
    bg_surface="#FFFFFF",         # Pure white elevated surface
    bg_surface_hover="#E5E5EA",   # Gentle hover fill
    bg_surface_active="#D1D1D6",  # Active / pressed fill
    border_subtle="#E2E2E7",      # Soft hairline divider
    border_focus="#2563EB",       # System Blue focus
    text_primary="#1D1D1F",       # Apple primary dark label
    text_secondary="#6E6E73",     # Apple secondary label
    text_muted="#8E8E93",         # Apple tertiary label
    accent_primary="#2563EB",     # System Blue
    accent_hover="#1D4ED8",       # Darker blue hover
    accent_subtle="rgba(37, 99, 235, 0.12)",
    status_success="#34C759",
    status_warning="#FF9500",
    status_error="#FF3B30",
    canvas_bg="#EBEBF0",
    radius_sm=7,
    radius_md=10,
    radius_lg=12,
    radius_pill=9999,
    border_width=1,
    font_family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif',
    font_mono='"SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace',
)

THEMES: Dict[str, ThemeTokens] = {
    "dark": DARK_TOKENS,
    "light": LIGHT_TOKENS,
}


def _ensure_theme_svg_assets() -> Dict[str, str]:
    """Ensures vector SVG indicator icons exist on disk for QSS url() references."""
    asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
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
    paths: Dict[str, str] = {}
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


def get_tokens(theme_name: str = "dark") -> ThemeTokens:
    """Retrieves design tokens for the specified theme, defaulting to dark."""
    return THEMES.get(theme_name.lower(), DARK_TOKENS)


def build_stylesheet(tokens: ThemeTokens) -> str:
    """
    Generates an authentic Apple macOS Sonoma / Sequoia QSS stylesheet.
    Emphasizes airy spacing, soft pill controls, borderless source lists,
    unified toolbar vibrancy, and zero nested box clutter.
    """
    is_dark = tokens.name == "dark"
    svg_paths = _ensure_theme_svg_assets()
    check_white_svg = svg_paths["check_white"]
    check_disabled_svg = svg_paths["check_disabled"]
    chevron_down_svg = svg_paths["chevron_down_dark"] if is_dark else svg_paths["chevron_down_light"]

    group_bg = "rgba(255, 255, 255, 0.03)" if is_dark else "#FFFFFF"
    input_bg = "#141417" if is_dark else "#FFFFFF"
    input_focus_bg = "#1C1C21" if is_dark else "#FFFFFF"
    control_border = "#38383E" if is_dark else "#D1D1D6"
    seg_track_bg = "rgba(0, 0, 0, 0.28)" if is_dark else "rgba(0, 0, 0, 0.06)"
    btn_bg = "#26262B" if is_dark else "#FFFFFF"
    btn_border = "#38383E" if is_dark else "rgba(0, 0, 0, 0.12)"
    scroll_thumb = "rgba(255, 255, 255, 0.18)" if is_dark else "rgba(0, 0, 0, 0.18)"
    scroll_thumb_hover = "rgba(255, 255, 255, 0.32)" if is_dark else "rgba(0, 0, 0, 0.32)"
    nav_list_bg = "#1E1E22" if is_dark else "#F2F2F7"
    table_header_bg = "#242429" if is_dark else "#F2F2F7"

    return f"""
    /* Global Application Typography & Reset */
    QWidget {{
        font-family: {tokens.font_family};
        font-size: 12px;
        color: {tokens.text_primary};
        background-color: transparent;
        selection-background-color: {tokens.accent_primary};
        selection-color: #FFFFFF;
        outline: none;
    }}

    QMainWindow, QDialog {{
        background-color: {tokens.bg_base};
    }}

    QLabel {{
        color: {tokens.text_primary};
        background-color: transparent;
    }}

    QScrollArea,
    QScrollArea > QWidget,
    QScrollArea > QWidget > QWidget,
    QWidget#settingsScrollContent {{
        background-color: transparent;
        border: none;
    }}

    /* Elevated Card Containers */
    QFrame[class="card"], QWidget[class="card"],
    QFrame#cardFrame, QFrame#sidebarCard {{
        background-color: {tokens.bg_surface};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_md}px;
    }}

    /* Inspector Panel — Clean Flush macOS Right Sidebar without boxy border */
    QFrame#inspectorCard {{
        background-color: {tokens.bg_sidebar};
        border: none;
        border-left: 1px solid {tokens.border_subtle};
        border-radius: 0px;
    }}

    /* Left Source List Sidebar */
    QWidget[class="sidebar"], QFrame[class="sidebar"] {{
        background-color: {tokens.bg_sidebar};
        border-right: 1px solid {tokens.border_subtle};
    }}

    /* Settings Dialog Left Category Navigation List */
    QListWidget#settingsNavList {{
        background-color: {nav_list_bg};
        border: 1px solid {tokens.border_subtle};
        border-radius: 10px;
        padding: 6px;
        outline: none;
    }}

    QListWidget#settingsNavList::item {{
        border-radius: 7px;
        padding: 6px 10px;
        margin: 2px 0px;
        font-size: 12px;
        font-weight: 500;
        color: {tokens.text_primary};
    }}

    QListWidget#settingsNavList::item:hover:!selected {{
        background-color: {tokens.bg_surface_hover};
    }}

    QListWidget#settingsNavList::item:selected {{
        background-color: {tokens.accent_primary};
        color: #FFFFFF;
        font-weight: 600;
    }}

    /* Optional NavRail (compact & unobtrusive) */
    #navRail {{
        background-color: {tokens.bg_sidebar};
        border-right: 1px solid {tokens.border_subtle};
        min-width: 40px;
        max-width: 40px;
    }}

    #navRail QToolButton {{
        background: transparent;
        border: none;
        border-radius: {tokens.radius_sm}px;
        padding: 7px;
        margin: 2px 4px;
        color: {tokens.text_secondary};
    }}

    #navRail QToolButton:hover {{
        background-color: {tokens.bg_surface_hover};
        color: {tokens.text_primary};
    }}

    #navRail QToolButton:checked {{
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
    }}

    /* Unified macOS Window Titlebar & Toolbar */
    QFrame[class="toolbar"] {{
        background-color: {tokens.bg_surface};
        border-bottom: 1px solid {tokens.border_subtle};
        padding: 4px 12px;
    }}

    QToolBar {{
        background-color: transparent;
        border: none;
        spacing: 6px;
        padding: 2px 0px;
    }}

    QToolBar QToolButton {{
        background-color: {btn_bg};
        color: {tokens.text_primary};
        border: 1px solid {btn_border};
        border-radius: {tokens.radius_sm}px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 500;
    }}

    QToolBar QToolButton:hover {{
        background-color: {tokens.bg_surface_hover};
        border-color: {tokens.border_focus};
    }}

    QToolBar QToolButton:pressed {{
        background-color: {tokens.bg_surface_active};
    }}

    /* Split Comparison Bottom Slider Bar */
    #sliderBar {{
        background-color: {tokens.bg_surface};
        border-top: 1px solid {tokens.border_subtle};
    }}

    /* Lightweight macOS Finder-Style Footer Bar */
    #statusBar {{
        background-color: {tokens.bg_sidebar};
        border-top: 1px solid {tokens.border_subtle};
        padding: 0px 12px;
    }}

    #statusLabel, .ide-status-text {{
        font-family: {tokens.font_family};
        font-size: 11px;
        color: {tokens.text_secondary};
    }}

    /* Authentic macOS NSSegmentedControl Pill Container */
    #segmentedControl {{
        background-color: {seg_track_bg};
        border: 1px solid {tokens.border_subtle};
        border-radius: 8px;
        padding: 2px;
    }}

    QToolButton[class="segment-btn"] {{
        background-color: transparent;
        color: {tokens.text_secondary};
        border: none;
        border-radius: 6px;
        padding: 4px 11px;
        font-weight: 500;
        font-size: 12px;
    }}

    QToolButton[class="segment-btn"]:hover {{
        color: {tokens.text_primary};
    }}

    QToolButton[class="segment-btn"]:checked {{
        background-color: {tokens.accent_primary};
        color: #FFFFFF;
        font-weight: 600;
    }}

    /* Standard macOS Push Buttons */
    QPushButton {{
        background-color: {btn_bg};
        color: {tokens.text_primary};
        border: 1px solid {btn_border};
        border-radius: {tokens.radius_sm}px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 500;
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {tokens.bg_surface_hover};
        border-color: {control_border};
    }}

    QPushButton:pressed {{
        background-color: {tokens.bg_surface_active};
    }}

    QPushButton:disabled {{
        color: {tokens.text_muted};
        background-color: transparent;
        border-color: {tokens.border_subtle};
    }}

    /* Apple System Blue Primary Action Button */
    QPushButton[class="primaryBtn"], QPushButton#primaryBtn {{
        background-color: {tokens.accent_primary};
        color: #FFFFFF;
        border: 1px solid transparent;
        border-radius: {tokens.radius_sm}px;
        font-weight: 600;
    }}

    QPushButton[class="primaryBtn"]:hover, QPushButton#primaryBtn:hover {{
        background-color: {tokens.accent_hover};
    }}

    QPushButton[class="primaryBtn"]:pressed, QPushButton#primaryBtn:pressed {{
        background-color: {tokens.accent_hover};
    }}

    /* Subtle macOS Toolbar Icon Buttons */
    QToolButton[class="icon-action-btn"], QPushButton[class="icon-action-btn"],
    .icon-btn {{
        background-color: transparent;
        color: {tokens.text_secondary};
        border: none;
        border-radius: {tokens.radius_sm}px;
        padding: 5px;
        min-width: 26px;
        min-height: 26px;
    }}

    QToolButton[class="icon-action-btn"]:hover, QPushButton[class="icon-action-btn"]:hover,
    .icon-btn:hover {{
        background-color: {tokens.bg_surface_hover};
        color: {tokens.text_primary};
    }}

    QToolButton[class="icon-action-btn"]:pressed, QPushButton[class="icon-action-btn"]:pressed,
    .icon-btn:pressed {{
        background-color: {tokens.bg_surface_active};
    }}

    QToolButton[class="icon-action-btn"]:checked {{
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
    }}

    /* Sidebar Drop Zone — Soft macOS Intake Card */
    #dropZone {{
        border: 1px dashed {tokens.border_subtle};
        border-radius: {tokens.radius_md}px;
        background-color: {group_bg};
        padding: 10px;
    }}

    #dropZone:hover {{
        border-color: {tokens.accent_primary};
        background-color: {tokens.accent_subtle};
    }}

    #dropZoneTitle {{
        font-weight: 600;
        font-size: 12px;
        color: {tokens.text_primary};
    }}

    #dropZoneSub {{
        font-size: 11px;
        color: {tokens.text_secondary};
    }}

    /* Soft Pill Count Badge */
    #countBadge, .ide-badge {{
        font-family: {tokens.font_family};
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
        border: none;
        border-radius: 9px;
        padding: 1px 8px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* Inspector Detail Container — Clean & Uncluttered */
    #detailFrame {{
        background-color: transparent;
        border: none;
        border-top: 1px solid {tokens.border_subtle};
        border-radius: 0px;
        padding-top: 8px;
    }}

    #blockTitle {{
        font-family: {tokens.font_family};
        font-weight: 600;
        font-size: 12px;
        color: {tokens.text_primary};
    }}

    #sizeValLabel {{
        font-family: {tokens.font_family};
        color: {tokens.accent_primary};
        font-weight: 600;
    }}

    /* Ultra-Slim macOS Overlay Scrollbars */
    QScrollBar:vertical {{
        border: none;
        background: transparent;
        width: 6px;
        margin: 2px 1px;
    }}

    QScrollBar::handle:vertical {{
        background: {scroll_thumb};
        min-height: 24px;
        border-radius: 3px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {scroll_thumb_hover};
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
        background: {scroll_thumb};
        min-width: 24px;
        border-radius: 3px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {scroll_thumb_hover};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
        width: 0px;
    }}

    /* macOS Finder / Preview Source Lists */
    QListWidget {{
        background-color: transparent;
        border: none;
        outline: none;
    }}

    QListWidget::item {{
        padding: 5px 8px;
        border-radius: 8px;
        margin: 1px 2px;
        color: {tokens.text_primary};
        border: none;
    }}

    QListWidget::item:hover {{
        background-color: {tokens.bg_surface_hover};
    }}

    QListWidget::item:selected {{
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
        font-weight: 600;
    }}

    /* Soft Rounded Text Inputs & SpinBoxes (Natural SF Pro Typography) */
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {input_bg};
        color: {tokens.text_primary};
        border: 1px solid {control_border};
        border-radius: {tokens.radius_sm}px;
        padding: 5px 9px;
        font-family: {tokens.font_family};
        font-size: 12px;
        min-height: 20px;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {tokens.border_focus};
        background-color: {input_focus_bg};
    }}

    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: {tokens.text_muted};
        background-color: transparent;
        border-color: {tokens.border_subtle};
    }}

    /* Glossary & Data Tables */
    QTableWidget, QTableView {{
        background-color: {input_bg};
        alternate-background-color: {group_bg};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        gridline-color: {tokens.border_subtle};
        selection-background-color: {tokens.accent_subtle};
        selection-color: {tokens.text_primary};
        outline: none;
    }}

    QTableWidget::item, QTableView::item {{
        padding: 4px 8px;
        color: {tokens.text_primary};
        border: none;
    }}

    QTableWidget::item:selected, QTableView::item:selected {{
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
    }}

    QTableWidget QLineEdit, QTableView QLineEdit {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_focus};
        border-radius: 4px;
        padding: 1px 6px;
        margin: 0px;
        min-height: 18px;
    }}

    QHeaderView {{
        background-color: transparent;
        border: none;
    }}

    QHeaderView::section {{
        background-color: {table_header_bg};
        color: {tokens.text_secondary};
        font-weight: 600;
        font-size: 11px;
        padding: 6px 8px;
        border: none;
        border-bottom: 1px solid {tokens.border_subtle};
        border-right: 1px solid {tokens.border_subtle};
    }}

    QTableCornerButton::section {{
        background-color: {table_header_bg};
        border: none;
        border-bottom: 1px solid {tokens.border_subtle};
        border-right: 1px solid {tokens.border_subtle};
    }}

    /* macOS Rounded Context Menus */
    QMenu {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_subtle};
        border-radius: {tokens.radius_md}px;
        padding: 5px;
    }}

    QMenu::item {{
        background-color: transparent;
        color: {tokens.text_primary};
        padding: 6px 20px 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        min-width: 140px;
    }}

    QMenu::item:selected {{
        background-color: {tokens.accent_primary};
        color: #FFFFFF;
    }}

    QMenu::item:disabled {{
        color: {tokens.text_muted};
        background-color: transparent;
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {tokens.border_subtle};
        margin: 4px 8px;
    }}

    /* macOS Pop-Up Button / ComboBox */
    QComboBox {{
        background-color: {btn_bg};
        color: {tokens.text_primary};
        border: 1px solid {btn_border};
        border-radius: {tokens.radius_sm}px;
        padding: 5px 28px 5px 10px;
        min-height: 22px;
        font-size: 12px;
    }}

    QComboBox:hover, QComboBox:focus {{
        background-color: {tokens.bg_surface_hover};
        border-color: {tokens.border_focus};
    }}

    QComboBox:disabled {{
        color: {tokens.text_muted};
        background-color: transparent;
        border-color: {tokens.border_subtle};
    }}

    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border: none;
        background: transparent;
    }}

    QComboBox::down-arrow {{
        image: url("{chevron_down_svg}");
        width: 12px;
        height: 12px;
    }}

    QComboBoxPrivateContainer,
    QComboBox QAbstractItemView,
    QComboBox QListView {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: 1px solid {control_border};
        border-radius: 6px;
        padding: 4px;
        selection-background-color: {tokens.accent_primary};
        selection-color: #FFFFFF;
        outline: none;
    }}

    QComboBox QAbstractItemView::item,
    QComboBox QListView::item {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        min-height: 26px;
        padding: 4px 10px;
        border-radius: 5px;
    }}

    QComboBox QAbstractItemView::item:hover,
    QComboBox QListView::item:hover {{
        background-color: {tokens.bg_surface_hover};
        color: {tokens.text_primary};
    }}

    QComboBox QAbstractItemView::item:selected,
    QComboBox QListView::item:selected {{
        background-color: {tokens.accent_primary};
        color: #FFFFFF;
    }}

    /* macOS Segmented Inspector Tabs */
    QTabWidget::pane {{
        border: none;
        background-color: transparent;
        margin-top: 4px;
    }}

    QTabBar {{
        background-color: {seg_track_bg};
        border-radius: 7px;
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {tokens.text_secondary};
        border: none;
        border-radius: 6px;
        padding: 5px 14px;
        margin: 2px;
        font-size: 12px;
        font-weight: 500;
    }}

    QTabBar::tab:selected {{
        background-color: {tokens.bg_surface_active};
        color: {tokens.text_primary};
        font-weight: 600;
    }}

    QTabBar::tab:hover:!selected {{
        color: {tokens.text_primary};
    }}

    /* Clean macOS Form Sections */
    QGroupBox {{
        background-color: {group_bg};
        border: 1px solid {tokens.border_subtle};
        border-radius: {tokens.radius_md}px;
        margin-top: 22px;
        padding: 14px 12px 12px 12px;
        color: {tokens.text_primary};
        font-size: 12px;
        font-weight: 600;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 6px;
        top: 2px;
        padding: 0px 4px;
        color: {tokens.text_secondary};
        font-size: 12px;
        font-weight: 600;
    }}

    /* Smooth macOS Sliders */
    QSlider:horizontal {{
        min-height: 22px;
    }}

    QSlider::groove:horizontal {{
        height: 4px;
        background: {tokens.border_subtle};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal {{
        background: {tokens.accent_primary};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal:disabled {{
        background: {tokens.border_subtle};
    }}

    QSlider::handle:horizontal {{
        background: #FFFFFF;
        border: 2px solid {tokens.accent_primary};
        width: 12px;
        height: 12px;
        margin-top: -6px;
        margin-bottom: -6px;
        border-radius: 8px;
    }}

    QSlider::handle:horizontal:disabled {{
        background: {tokens.bg_surface};
        border: 2px solid {tokens.text_muted};
    }}

    /* Slim Pill Progress Bar */
    QProgressBar {{
        background-color: {tokens.border_subtle};
        border: none;
        border-radius: 3px;
        height: 6px;
        text-align: center;
        font-size: 9px;
        color: {tokens.text_primary};
    }}

    QProgressBar::chunk {{
        background-color: {tokens.accent_primary};
        border-radius: 3px;
    }}

    /* macOS CheckBox */
    QCheckBox {{
        spacing: 8px;
        font-size: 12px;
        color: {tokens.text_primary};
        background-color: transparent;
        min-height: 20px;
    }}

    QCheckBox:disabled {{
        color: {tokens.text_muted};
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid {control_border};
        background-color: {input_bg};
    }}

    QCheckBox::indicator:hover:unchecked {{
        border-color: {tokens.accent_primary};
        background-color: {tokens.bg_surface_hover};
    }}

    QCheckBox::indicator:checked {{
        background-color: {tokens.accent_primary};
        border-color: {tokens.accent_primary};
        image: url("{check_white_svg}");
    }}

    QCheckBox::indicator:checked:disabled {{
        background-color: {tokens.bg_surface_active};
        border-color: {tokens.border_subtle};
        image: url("{check_disabled_svg}");
    }}

    QCheckBox::indicator:unchecked:disabled {{
        background-color: transparent;
        border-color: {tokens.border_subtle};
    }}

    /* Tooltips */
    QToolTip {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_subtle};
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 11px;
        font-family: {tokens.font_family};
    }}

    /* Crisp 1px Splitter Hairline */
    QSplitter::handle {{
        background-color: {tokens.border_subtle};
    }}

    QSplitter::handle:hover {{
        background-color: {tokens.accent_primary};
    }}

    QSplitter::handle:horizontal {{
        width: 1px;
    }}

    QSplitter::handle:vertical {{
        height: 1px;
    }}

    /* Floating Canvas Zoom HUD */
    #canvasZoomHud {{
        background-color: rgba(28, 28, 32, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
    }}
    """
