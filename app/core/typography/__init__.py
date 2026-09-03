"""
app/core/typography - Smart Typography Engine with CJK line breaking, bisection auto-fit, and stroke rendering.
"""
from .auto_fit import AutoFitEngine, LayoutResult, AutoFitResult
from .line_breaker import LineBreaker, GYOTO_KINSOKU, GYOMATSU_KINSOKU
from .vertical_layout import VerticalLayoutEngine, VERTICAL_GLYPH_MAP
from .stroke_renderer import StrokeRenderer, StrokeStyle, DropShadowStyle
from .engine import TypographyEngine

__all__ = [
    "AutoFitEngine",
    "LayoutResult",
    "AutoFitResult",
    "LineBreaker",
    "GYOTO_KINSOKU",
    "GYOMATSU_KINSOKU",
    "VerticalLayoutEngine",
    "VERTICAL_GLYPH_MAP",
    "StrokeRenderer",
    "StrokeStyle",
    "DropShadowStyle",
    "TypographyEngine",
]
