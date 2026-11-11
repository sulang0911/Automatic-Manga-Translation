"""
app/ui/theme/tokens.py
Apple macOS Human Interface Guidelines (Sonoma / Sequoia) Design Tokens and Stylesheet Generator.
Supports macOS Dark Vibrancy (#18181B) and macOS Studio Light (#F8F9FA) themes.
"""
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
    input_bg = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.04)"
    input_focus_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "#FFFFFF"
    seg_track_bg = "rgba(0, 0, 0, 0.28)" if is_dark else "rgba(0, 0, 0, 0.06)"
    btn_bg = "rgba(255, 255, 255, 0.07)" if is_dark else "#FFFFFF"
    btn_border = "rgba(255, 255, 255, 0.09)" if is_dark else "rgba(0, 0, 0, 0.10)"
    scroll_thumb = "rgba(255, 255, 255, 0.18)" if is_dark else "rgba(0, 0, 0, 0.18)"
    scroll_thumb_hover = "rgba(255, 255, 255, 0.32)" if is_dark else "rgba(0, 0, 0, 0.32)"

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
        padding: 5px 11px;
        font-size: 12px;
        font-weight: 500;
    }}

    QPushButton:hover {{
        background-color: {tokens.bg_surface_hover};
        border-color: {tokens.border_subtle};
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
        background-color: {input_bg};
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

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
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

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
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

    /* Soft Rounded Text Inputs (Natural SF Pro Typography) */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {input_bg};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        padding: 6px 9px;
        font-family: {tokens.font_family};
        font-size: 12px;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {tokens.border_focus};
        background-color: {input_focus_bg};
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
        padding: 4px 10px;
        min-height: 20px;
        font-size: 12px;
    }}

    QComboBox:hover, QComboBox:focus {{
        background-color: {tokens.bg_surface_hover};
        border-color: {tokens.border_focus};
    }}

    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 0px;
    }}

    QComboBoxPrivateContainer,
    QComboBox QAbstractItemView,
    QComboBox QListView {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_subtle};
        border-radius: 8px;
        padding: 4px;
        selection-background-color: {tokens.accent_primary};
        selection-color: #FFFFFF;
        outline: none;
    }}

    QComboBox QAbstractItemView::item,
    QComboBox QListView::item {{
        background-color: transparent;
        color: {tokens.text_primary};
        min-height: 24px;
        padding: 4px 10px;
        border-radius: 5px;
    }}

    QComboBox QAbstractItemView::item:hover,
    QComboBox QListView::item:hover {{
        background-color: {tokens.bg_surface_hover};
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

    /* Clean Borderless / Flat macOS Form Sections (No Box-in-Box Clutter) */
    QGroupBox {{
        background-color: {input_bg};
        border: 1px solid {tokens.border_subtle};
        border-radius: {tokens.radius_md}px;
        margin-top: 16px;
        padding: 12px 10px 10px 10px;
        color: {tokens.text_secondary};
        font-size: 11px;
        font-weight: 600;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 4px;
        top: 0px;
        padding: 0 4px;
        color: {tokens.text_secondary};
        font-weight: 600;
    }}

    /* Smooth macOS Sliders */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {tokens.border_subtle};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal {{
        background: {tokens.accent_primary};
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: #FFFFFF;
        border: 1.5px solid {tokens.accent_primary};
        width: 14px;
        height: 14px;
        margin-top: -6px;
        margin-bottom: -6px;
        border-radius: 7px;
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
        spacing: 6px;
        font-size: 12px;
        color: {tokens.text_primary};
    }}

    QCheckBox::indicator {{
        width: 15px;
        height: 15px;
        border-radius: 4px;
        border: 1px solid {tokens.border_subtle};
        background-color: {input_bg};
    }}

    QCheckBox::indicator:checked {{
        background-color: {tokens.accent_primary};
        border-color: {tokens.accent_primary};
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
