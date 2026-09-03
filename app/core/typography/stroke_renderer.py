"""
app/core/typography/stroke_renderer.py
ITU-R BT.709 Luminance Auto-Contrast Stroke, Drop Shadow, and Dual PIL/QPainter Pipelines.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from PyQt6.QtGui import QPainter, QPainterPath, QColor, QPen, QBrush, QFont
    from PyQt6.QtCore import Qt, QPointF
    HAS_QT = True
except ImportError:
    HAS_QT = False


@dataclass(frozen=True)
class StrokeStyle:
    color_rgba: Tuple[int, int, int, int]
    width: float


@dataclass(frozen=True)
class DropShadowStyle:
    enabled: bool
    color_rgba: Tuple[int, int, int, int]
    offset_x: float
    offset_y: float
    blur_radius: float


class StrokeRenderer:
    """
    Computes ITU-R BT.709 perceived luminance for intelligent auto-contrast,
    and executes dual-pipeline rendering (PIL for headless export, QPainter for Qt canvas).
    """

    @staticmethod
    def calculate_bt709_luminance(r: int, g: int, b: int) -> float:
        """
        ITU-R BT.709 standard luminance equation.
        Y = 0.2126*R + 0.7152*G + 0.0722*B
        """
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @classmethod
    def get_auto_contrast_stroke(
        cls,
        text_rgb: Tuple[int, int, int],
        font_size: float,
        threshold: float = 140.0
    ) -> StrokeStyle:
        lum = cls.calculate_bt709_luminance(*text_rgb)
        # High luminance (light text) -> Dark stroke; Low luminance (dark text) -> Light stroke
        if lum > threshold:
            color = (0, 0, 0, 217)       # 85% black
        else:
            color = (255, 255, 255, 217) # 85% white

        # 6% of font size clamped to [0.5, 3.5]px
        stroke_w = max(0.5, min(3.5, font_size * 0.06))
        return StrokeStyle(color_rgba=color, width=stroke_w)

    @staticmethod
    def get_drop_shadow_params(font_size: float, enabled: bool = True) -> DropShadowStyle:
        if not enabled:
            return DropShadowStyle(False, (0, 0, 0, 0), 0.0, 0.0, 0.0)
        dist = max(1.0, font_size * 0.06)
        blur = max(1.0, font_size * 0.08)
        return DropShadowStyle(
            enabled=True,
            color_rgba=(0, 0, 0, 178),  # 70% black
            offset_x=dist,
            offset_y=dist,
            blur_radius=blur
        )

    # --------------------------------------------------------------------------
    # PIL / Pillow Headless Drawing Pipeline
    # --------------------------------------------------------------------------
    @classmethod
    def render_text_pil(
        cls,
        target_image: Image.Image,
        text: str,
        x: float,
        y: float,
        font: Any,
        fill_rgba: Tuple[int, int, int, int],
        stroke: Optional[StrokeStyle] = None,
        shadow: Optional[DropShadowStyle] = None
    ) -> None:
        """
        Render text glyph with stroke and drop shadow using PIL onto target_image.
        """
        # 1. Drop shadow layer
        if shadow and shadow.enabled:
            shadow_layer = Image.new("RGBA", target_image.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_draw.text(
                (x + shadow.offset_x, y + shadow.offset_y),
                text,
                font=font,
                fill=shadow.color_rgba
            )
            blurred_shadow = shadow_layer.filter(ImageFilter.GaussianBlur(shadow.blur_radius))
            target_image.alpha_composite(blurred_shadow)

        # 2. Text with stroke & fill
        draw = ImageDraw.Draw(target_image)
        stroke_w = int(round(stroke.width)) if stroke else 0
        stroke_fill = stroke.color_rgba if stroke else None

        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill_rgba,
            stroke_width=stroke_w,
            stroke_fill=stroke_fill
        )

    # --------------------------------------------------------------------------
    # PyQt6 / QPainter Vector Drawing Pipeline
    # --------------------------------------------------------------------------
    @classmethod
    def render_text_painter(
        cls,
        painter: Any,
        text: str,
        x: float,
        y: float,
        font: Any,
        fill_rgba: Tuple[int, int, int, int],
        stroke: Optional[StrokeStyle] = None,
        shadow: Optional[DropShadowStyle] = None
    ) -> None:
        """
        Render vector text with antialiased stroke and shadow via QPainterPath.
        """
        if not HAS_QT:
            raise RuntimeError("PyQt6 is required for QPainter rendering pipeline.")

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        path = QPainterPath()
        path.addText(QPointF(x, y), font, text)

        # 1. Drop shadow pass
        if shadow and shadow.enabled:
            shadow_path = QPainterPath(path)
            shadow_path.translate(shadow.offset_x, shadow.offset_y)
            sr, sg, sb, sa = shadow.color_rgba
            painter.fillPath(shadow_path, QBrush(QColor(sr, sg, sb, sa)))

        # 2. Stroke outline pass
        if stroke and stroke.width > 0:
            cr, cg, cb, ca = stroke.color_rgba
            pen = QPen(
                QColor(cr, cg, cb, ca),
                stroke.width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            )
            painter.strokePath(path, pen)

        # 3. Fill glyph body pass
        fr, fg, fb, fa = fill_rgba
        painter.fillPath(path, QBrush(QColor(fr, fg, fb, fa)))
