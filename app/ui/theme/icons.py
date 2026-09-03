"""
app/ui/theme/icons.py
Resolution-independent vector SVG icon provider for Apple HIG interface.
Renders crisp icons at any high-DPI scaling factor with dynamic tinting.
"""
from typing import Dict, Optional
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import QByteArray, QSize, Qt


# Minimalist 24x24 SVG path definitions
SVG_TEMPLATES: Dict[str, str] = {
    "split": """
        <path d="M12 3v18M8 8l-4 4 4 4M16 8l4 4-4 4" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "columns": """
        <rect x="3" y="4" width="8" height="16" rx="2" stroke="{color}" stroke-width="2" fill="none"/>
        <rect x="13" y="4" width="8" height="16" rx="2" stroke="{color}" stroke-width="2" fill="none"/>
    """,
    "eye": """
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="{color}" stroke-width="2" fill="none"/>
        <circle cx="12" cy="12" r="3" stroke="{color}" stroke-width="2" fill="none"/>
    """,
    "sparkles": """
        <path d="M12 2l2.4 5.6L20 10l-5.6 2.4L12 18l-2.4-5.6L4 10l5.6-2.4L12 2zM19 16l1.2 2.8L23 20l-2.8 1.2L19 24l-1.2-2.8L15 20l2.8-1.2L19 16z" fill="{color}"/>
    """,
    "eraser": """
        <path d="M20 20H7L3 16c-1-1-1-2.5 0-3.5l10-10c1-1 2.5-1 3.5 0l4 4c1 1 1 2.5 0 3.5L13 18" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "zoom_in": """
        <circle cx="11" cy="11" r="8" stroke="{color}" stroke-width="2" fill="none"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="11" y1="8" x2="11" y2="14" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="8" y1="11" x2="14" y2="11" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
    """,
    "zoom_out": """
        <circle cx="11" cy="11" r="8" stroke="{color}" stroke-width="2" fill="none"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="8" y1="11" x2="14" y2="11" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
    """,
    "fit_window": """
        <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "actual_size": """
        <rect x="3" y="3" width="18" height="18" rx="2" stroke="{color}" stroke-width="2" fill="none"/>
        <text x="12" y="16" font-size="10" font-family="sans-serif" font-weight="bold" fill="{color}" text-anchor="middle">1:1</text>
    """,
    "play": """
        <polygon points="5 3 19 12 5 21 5 3" fill="{color}"/>
    """,
    "settings": """
        <circle cx="12" cy="12" r="3" stroke="{color}" stroke-width="2" fill="none"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="{color}" stroke-width="2" fill="none"/>
    """,
    "sun": """
        <circle cx="12" cy="12" r="5" stroke="{color}" stroke-width="2" fill="none"/>
        <line x1="12" y1="1" x2="12" y2="3" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="12" y1="21" x2="12" y2="23" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="1" y1="12" x2="3" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="21" y1="12" x2="23" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
    """,
    "moon": """
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "folder_open": """
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "trash": """
        <polyline points="3 6 5 6 21 6" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "chevron_left": """
        <polyline points="15 18 9 12 15 6" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "chevron_right": """
        <polyline points="9 18 15 12 9 6" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "layers": """
        <polygon points="12 2 2 7 12 12 22 7 12 2" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <polyline points="2 17 12 22 22 17" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <polyline points="2 12 12 17 22 12" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "check": """
        <polyline points="20 6 9 17 4 12" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    """,
    "alert": """
        <circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="2" fill="none"/>
        <line x1="12" y1="8" x2="12" y2="12" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
        <line x1="12" y1="16" x2="12.01" y2="16" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
    """,
    "download": """
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <polyline points="7 10 12 15 17 10" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <line x1="12" y1="15" x2="12" y2="3" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    """,
    "play_all": """
        <polygon points="4 4 14 12 4 20 4 4" fill="{color}"/>
        <polygon points="12 4 22 12 12 20 12 4" fill="{color}"/>
    """,
}


def render_svg_pixmap(name: str, color: str = "#F4F4F5", size: int = 18) -> QPixmap:
    """Renders a vector SVG path into a high-DPI QPixmap with given color and size."""
    body = SVG_TEMPLATES.get(name, SVG_TEMPLATES["alert"])
    filled_body = body.format(color=color)
    svg_data = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}">'
        f'{filled_body}'
        f'</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    return pixmap


def get_icon(name: str, color: str = "#F4F4F5", active_color: Optional[str] = None, size: int = 18) -> QIcon:
    """Constructs a multi-state QIcon from vector SVG."""
    icon = QIcon()
    pix_normal = render_svg_pixmap(name, color=color, size=size)
    icon.addPixmap(pix_normal, QIcon.Mode.Normal, QIcon.State.Off)

    if active_color:
        pix_active = render_svg_pixmap(name, color=active_color, size=size)
        icon.addPixmap(pix_active, QIcon.Mode.Normal, QIcon.State.On)
        icon.addPixmap(pix_active, QIcon.Mode.Active, QIcon.State.On)
    return icon
