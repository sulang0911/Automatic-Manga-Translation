"""
app/ui/theme/tokens.py
Apple Human Interface Guidelines Design Tokens and Stylesheet Generator.
Supports Warm Zinc Dark (#18181B) and Clean Studio Light (#F8FAFC) themes.
Fully compliant with WCAG 2.1 AA contrast standards (>= 4.5:1 for normal text).
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
    radius_sm: int = 6
    radius_md: int = 10
    radius_lg: int = 12
    radius_pill: int = 9999
    border_width: int = 1
    font_family: str = '-apple-system, BlinkMacSystemFont, "Segoe UI Variable Display", "Segoe UI", "Inter", sans-serif'
    font_mono: str = '"Cascadia Code", "SF Mono", "Consolas", monospace'


DARK_TOKENS = ThemeTokens(
    name="dark",
    bg_base="#18181B",
    bg_sidebar="#121214",
    bg_surface="#202024",
    bg_surface_hover="#27272D",
    bg_surface_active="#323238",
    border_subtle="rgba(255, 255, 255, 0.10)",
    border_focus="#3B82F6",
    text_primary="#F4F4F5",
    text_secondary="#D4D4D8",     # High contrast (contrast ~10:1 on #18181B)
    text_muted="#A1A1AA",         # WCAG AA compliant (contrast > 5.5:1 on #18181B)
    accent_primary="#3B82F6",
    accent_hover="#2563EB",
    accent_subtle="rgba(59, 130, 246, 0.18)",
    status_success="#10B981",
    status_warning="#F59E0B",
    status_error="#EF4444",
    canvas_bg="#141416",
    radius_sm=6,
    radius_md=10,
    radius_lg=12,
    radius_pill=9999,
    border_width=1,
)

LIGHT_TOKENS = ThemeTokens(
    name="light",
    bg_base="#F8F9FA",
    bg_sidebar="#F0F1F3",
    bg_surface="#FFFFFF",
    bg_surface_hover="#F4F4F6",
    bg_surface_active="#EAECEF",
    border_subtle="rgba(0, 0, 0, 0.10)",
    border_focus="#2563EB",
    text_primary="#0F172A",       # Slate-900 (contrast > 14:1 on white)
    text_secondary="#334155",     # Slate-700 (contrast > 8.5:1 on white)
    text_muted="#64748B",         # Slate-500 (contrast > 4.6:1 on white, passes WCAG AA)
    accent_primary="#2563EB",     # Blue-600 (contrast > 5.0:1 on white)
    accent_hover="#1D4ED8",
    accent_subtle="rgba(37, 99, 235, 0.12)",
    status_success="#059669",
    status_warning="#D97706",
    status_error="#DC2626",
    canvas_bg="#E2E8F0",
    radius_sm=6,
    radius_md=10,
    radius_lg=12,
    radius_pill=9999,
    border_width=1,
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
    Generates a complete QSS stylesheet conforming to Apple HIG design principles.
    Uses precise 1px borders, subtle surface contrast, and elegant rounded cards.
    Fully unifies all sub-panels, controls, drop zones, toolbars, and status bars.
    """
    return f"""
    /* Global Application Reset & Typography */
    QWidget {{
        font-family: {tokens.font_family};
        font-size: 13px;
        color: {tokens.text_primary};
        background-color: transparent;
        selection-background-color: {tokens.accent_primary};
        selection-color: #FFFFFF;
        outline: none;
    }}

    QMainWindow, QDialog {{
        background-color: {tokens.bg_base};
    }}

    /* Card Containers & Panels */
    QFrame[class="card"], QWidget[class="card"] {{
        background-color: {tokens.bg_surface};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_lg}px;
    }}

    QFrame[class="sidebar"] {{
        background-color: {tokens.bg_sidebar};
        border-right: {tokens.border_width}px solid {tokens.border_subtle};
    }}

    /* Vertical Navigation Rail */
    #navRail {{
        background-color: {tokens.bg_sidebar};
        border-right: {tokens.border_width}px solid {tokens.border_subtle};
    }}

    #navRail QToolButton {{
        background: transparent;
        border: none;
        border-radius: {tokens.radius_sm}px;
        padding: 8px;
        margin: 4px 0px;
        color: {tokens.text_secondary};
    }}

    #navRail QToolButton:hover {{
        background-color: {tokens.bg_surface_active};
        color: {tokens.text_primary};
    }}

    #navRail QToolButton:checked {{
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
    }}

    /* Action Toolbar & Controls */
    QFrame[class="toolbar"] {{
        background-color: {tokens.bg_surface};
        border-bottom: {tokens.border_width}px solid {tokens.border_subtle};
        padding: 4px 12px;
    }}

    /* Slider Bar for Split Comparison */
    #sliderBar {{
        background-color: {tokens.bg_surface};
        border-top: {tokens.border_width}px solid {tokens.border_subtle};
    }}

    /* Bottom Status Bar */
    #statusBar {{
        background-color: {tokens.bg_sidebar};
        border-top: {tokens.border_width}px solid {tokens.border_subtle};
        padding: 0px 12px;
    }}

    #statusLabel {{
        font-size: 11px;
        color: {tokens.text_secondary};
    }}

    /* Segmented Viewport Controls */
    #segmentedControl {{
        background-color: {tokens.bg_surface_active};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        padding: 2px;
    }}

    QToolButton[class="segment-btn"] {{
        background-color: transparent;
        color: {tokens.text_secondary};
        border: none;
        border-radius: 4px;
        padding: 4px 10px;
        font-weight: 500;
        font-size: 12px;
    }}

    QToolButton[class="segment-btn"]:hover {{
        color: {tokens.text_primary};
        background-color: {tokens.bg_surface_hover};
    }}

    QToolButton[class="segment-btn"]:checked {{
        background-color: {tokens.accent_primary};
        color: #FFFFFF;
        font-weight: 600;
    }}

    /* Push Buttons */
    QPushButton {{
        background-color: {tokens.bg_surface_hover};
        color: {tokens.text_primary};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        padding: 6px 14px;
        font-weight: 500;
    }}

    QPushButton:hover {{
        background-color: {tokens.bg_surface_active};
        border-color: {tokens.border_focus};
    }}

    QPushButton:pressed {{
        background-color: {tokens.accent_subtle};
    }}

    QPushButton:disabled {{
        color: {tokens.text_muted};
        background-color: {tokens.bg_surface};
        border-color: transparent;
    }}

    QPushButton[class="primaryBtn"], QPushButton#primaryBtn {{
        background-color: {tokens.accent_primary};
        color: #FFFFFF;
        border: none;
        font-weight: 600;
    }}

    QPushButton[class="primaryBtn"]:hover, QPushButton#primaryBtn:hover {{
        background-color: {tokens.accent_hover};
    }}

    QPushButton[class="primaryBtn"]:pressed, QPushButton#primaryBtn:pressed {{
        background-color: {tokens.accent_hover};
    }}

    /* Drop Zone */
    #dropZone {{
        border: 2px dashed {tokens.border_focus};
        border-radius: {tokens.radius_md}px;
        background-color: {tokens.accent_subtle};
        padding: 16px;
    }}

    #dropZoneTitle {{
        font-weight: 600;
        font-size: 12px;
        color: {tokens.text_primary};
    }}

    #dropZoneSub {{
        font-size: 11px;
        color: {tokens.text_muted};
    }}

    /* Count Badge */
    #countBadge {{
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
        border-radius: {tokens.radius_pill}px;
        padding: 1px 8px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* Detail Frame in Inspector */
    #detailFrame {{
        background-color: {tokens.bg_surface_hover};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        padding: 6px;
    }}

    #blockTitle {{
        font-weight: 600;
        font-size: 12px;
        color: {tokens.accent_primary};
    }}

    #sizeValLabel {{
        color: {tokens.accent_primary};
        font-weight: bold;
    }}

    /* Scrollbars */
    QScrollBar:vertical {{
        border: none;
        background: transparent;
        width: 8px;
        margin: 0px;
    }}

    QScrollBar::handle:vertical {{
        background: {tokens.border_subtle};
        min-height: 24px;
        border-radius: 4px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {tokens.text_muted};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        border: none;
        background: transparent;
        height: 8px;
        margin: 0px;
    }}

    QScrollBar::handle:horizontal {{
        background: {tokens.border_subtle};
        min-width: 24px;
        border-radius: 4px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {tokens.text_muted};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* QListWidget and Table Items */
    QListWidget {{
        background-color: transparent;
        border: none;
        outline: none;
    }}

    QListWidget::item {{
        padding: 8px 10px;
        border-radius: {tokens.radius_md}px;
        margin: 2px 4px;
        color: {tokens.text_primary};
    }}

    QListWidget::item:hover {{
        background-color: {tokens.bg_surface_hover};
    }}

    QListWidget::item:selected {{
        background-color: {tokens.accent_subtle};
        color: {tokens.accent_primary};
        font-weight: 600;
        border: 1px solid {tokens.border_focus};
    }}

    /* Text Inputs */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {tokens.bg_surface_hover};
        color: {tokens.text_primary};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        padding: 6px 10px;
    }}

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {tokens.border_focus};
        background-color: {tokens.bg_surface};
    }}

    /* QMenu - Context Menus and Cascading Submenus (一级与二级右键菜单) */
    QMenu {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_focus};
        border-radius: {tokens.radius_sm}px;
        padding: 4px;
    }}

    QMenu::item {{
        background-color: transparent;
        color: {tokens.text_primary};
        padding: 6px 26px 6px 12px;
        border-radius: 4px;
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
        margin: 4px 6px;
    }}

    QMenu::right-arrow {{
        margin-right: 8px;
    }}

    /* Dropdown Combo Boxes (二级下拉列表) */
    QComboBox {{
        background-color: {tokens.bg_surface_hover};
        color: {tokens.text_primary};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        padding: 5px 10px;
        min-height: 22px;
    }}

    QComboBox:focus, QComboBox:hover {{
        border-color: {tokens.border_focus};
    }}

    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border-left-width: 0px;
        border-top-right-radius: {tokens.radius_sm}px;
        border-bottom-right-radius: {tokens.radius_sm}px;
    }}

    /* QComboBox Popup Container & Item View (100% Solid Opaque Surface) */
    QComboBoxPrivateContainer,
    QComboBox QAbstractItemView,
    QComboBox QListView {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_focus};
        border-radius: {tokens.radius_sm}px;
        padding: 4px;
        selection-background-color: {tokens.accent_primary};
        selection-color: #FFFFFF;
        outline: none;
    }}

    QComboBox QAbstractItemView::item,
    QComboBox QListView::item {{
        background-color: transparent;
        color: {tokens.text_primary};
        min-height: 26px;
        padding: 4px 10px;
        border-radius: 4px;
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

    /* Tabs */
    QTabWidget::pane {{
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        background-color: {tokens.bg_surface};
    }}

    QTabBar::tab {{
        background-color: {tokens.bg_surface_hover};
        color: {tokens.text_secondary};
        border-top-left-radius: {tokens.radius_sm}px;
        border-top-right-radius: {tokens.radius_sm}px;
        padding: 6px 12px;
        margin-right: 2px;
        font-size: 12px;
        font-weight: 500;
    }}

    QTabBar::tab:selected {{
        background-color: {tokens.bg_surface};
        color: {tokens.accent_primary};
        font-weight: 600;
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-bottom-color: transparent;
    }}

    /* Group Box */
    QGroupBox {{
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_md}px;
        margin-top: 14px;
        padding-top: 10px;
        color: {tokens.text_primary};
        font-weight: 600;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {tokens.text_primary};
    }}

    /* Sliders */
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
        border: 2px solid {tokens.accent_primary};
        width: 16px;
        margin-top: -6px;
        margin-bottom: -6px;
        border-radius: 8px;
    }}

    /* Progress Bar */
    QProgressBar {{
        background-color: {tokens.bg_surface_hover};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_pill}px;
        height: 8px;
        text-align: center;
        font-size: 10px;
        color: {tokens.text_primary};
    }}

    QProgressBar::chunk {{
        background-color: {tokens.accent_primary};
        border-radius: {tokens.radius_pill}px;
    }}

    /* CheckBox */
    QCheckBox {{
        spacing: 6px;
        color: {tokens.text_primary};
    }}

    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: {tokens.border_width}px solid {tokens.border_subtle};
        background-color: {tokens.bg_surface_hover};
    }}

    QCheckBox::indicator:checked {{
        background-color: {tokens.accent_primary};
        border-color: {tokens.accent_primary};
    }}

    /* Tooltips */
    QToolTip {{
        background-color: {tokens.bg_surface};
        color: {tokens.text_primary};
        border: {tokens.border_width}px solid {tokens.border_subtle};
        border-radius: {tokens.radius_sm}px;
        padding: 5px 8px;
        font-size: 11px;
    }}

    /* Splitter */
    QSplitter::handle {{
        background-color: {tokens.border_subtle};
    }}

    QSplitter::handle:horizontal {{
        width: 1px;
    }}

    QSplitter::handle:vertical {{
        height: 1px;
    }}
    """
