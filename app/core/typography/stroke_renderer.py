"""
app/core/typography/stroke_renderer.py
ITU-R BT.709 Luminance Auto-Contrast Stroke, Drop Shadow, and Dual PIL/QPainter Pipelines.
Optimized with ROI-bounded drop shadow Gaussian blur to eliminate full-page RGBA allocations.
"""
import math
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
        threshold: float = 140.0,
        bg_rgb: Optional[Tuple[int, int, int]] = None,
        base_stroke_w: Optional[float] = None,
        auto_stroke_width: bool = True
    ) -> StrokeStyle:
        lum = cls.calculate_bt709_luminance(*text_rgb)
        alpha = 230 if bg_rgb is not None else 217
        # High luminance (light text, e.g. white/cyan) -> Dark stroke; Low luminance (dark/red text) -> Light stroke
        if lum > threshold:
            color = (0, 0, 0, alpha)
        else:
            color = (255, 255, 255, alpha)

        if base_stroke_w is not None and base_stroke_w > 0:
            if auto_stroke_width:
                # Smart adaptive stroke width: scales with font_size up to base_stroke_w (default 3.5px)
                # so small 11-16px CJK dialogue stays crisp (1.8-2.2px) while larger 24px+ SFX uses full 3.5px
                max_cap = max(2.0, float(base_stroke_w))
                stroke_w = min(max_cap, max(1.6, round(font_size * 0.145, 2)))
            else:
                stroke_w = float(base_stroke_w)
        else:
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
        shadow: Optional[DropShadowStyle] = None,
        is_bold: bool = False
    ) -> None:
        """
        Render text glyph with stroke, bolding, and drop shadow using PIL onto target_image.
        Uses ROI-bounded shadow layer rendering to avoid allocating full-page RGBA buffers per line/glyph.
        """
        if not text:
            return

        font_size = getattr(font, "size", 20)
        bold_extra = int(round(font_size * 0.035)) if (is_bold and font_size >= 30) else 0

        # 1. ROI-bounded Drop Shadow Layer
        if shadow and shadow.enabled:
            shadow_stroke_w = int(round(stroke.width + bold_extra)) if (stroke and stroke.width > 0) else bold_extra
            sx = x + shadow.offset_x
            sy = y + shadow.offset_y
            blur_pad = int(math.ceil(shadow.blur_radius * 3.0)) + shadow_stroke_w + 6

            try:
                bbox = font.getbbox(text)
                tw = max(1, int(math.ceil(bbox[2] - bbox[0])))
                th = max(1, int(math.ceil(bbox[3] - bbox[1])))
                bx0 = int(math.floor(sx + min(0, bbox[0]))) - blur_pad
                by0 = int(math.floor(sy + min(0, bbox[1]))) - blur_pad
                bx1 = int(math.ceil(sx + bbox[2])) + blur_pad
                by1 = int(math.ceil(sy + bbox[3])) + blur_pad
            except Exception:
                tw = int(math.ceil(len(text) * font_size * 1.2))
                th = int(math.ceil(font_size * 1.5))
                bx0 = int(math.floor(sx)) - blur_pad
                by0 = int(math.floor(sy)) - blur_pad
                bx1 = int(math.ceil(sx + tw)) + blur_pad
                by1 = int(math.ceil(sy + th)) + blur_pad

            img_w, img_h = target_image.size
            rx0 = max(0, bx0)
            ry0 = max(0, by0)
            rx1 = min(img_w, bx1)
            ry1 = min(img_h, by1)

            if rx1 > rx0 and ry1 > ry0:
                roi_w = rx1 - rx0
                roi_h = ry1 - ry0
                shadow_roi = Image.new("RGBA", (roi_w, roi_h), (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow_roi)
                shadow_draw.text(
                    (sx - rx0, sy - ry0),
                    text,
                    font=font,
                    fill=shadow.color_rgba,
                    stroke_width=shadow_stroke_w,
                    stroke_fill=shadow.color_rgba if shadow_stroke_w > 0 else None
                )
                blurred_roi = shadow_roi.filter(ImageFilter.GaussianBlur(shadow.blur_radius))
                target_image.alpha_composite(blurred_roi, dest=(rx0, ry0))
                del shadow_roi, blurred_roi

        # 2. Text with stroke & fill
        draw = ImageDraw.Draw(target_image)
        stroke_w = int(round(stroke.width)) if stroke else 0
        if stroke_w > 0 and font_size < 16:
            stroke_w = min(stroke_w, max(1, int(round(font_size * 0.15))))
        stroke_fill = stroke.color_rgba if stroke else None

        if is_bold:
            if stroke_w > 0:
                draw.text(
                    (x, y),
                    text,
                    font=font,
                    fill=stroke_fill,
                    stroke_width=stroke_w + bold_extra,
                    stroke_fill=stroke_fill
                )
                draw.text(
                    (x, y),
                    text,
                    font=font,
                    fill=fill_rgba,
                    stroke_width=bold_extra,
                    stroke_fill=fill_rgba
                )
            else:
                draw.text(
                    (x, y),
                    text,
                    font=font,
                    fill=fill_rgba,
                    stroke_width=bold_extra,
                    stroke_fill=fill_rgba
                )
        else:
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
        shadow: Optional[DropShadowStyle] = None,
        is_bold: bool = False
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
            extra = 1.0 if is_bold else 0.0
            pen = QPen(
                QColor(cr, cg, cb, ca),
                stroke.width + extra,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            )
            painter.strokePath(path, pen)

        # 3. Fill glyph body pass
        fr, fg, fb, fa = fill_rgba
        painter.fillPath(path, QBrush(QColor(fr, fg, fb, fa)))
        if is_bold and (not stroke or stroke.width <= 0):
            bold_pen = QPen(QColor(fr, fg, fb, fa), 1.0, Qt.PenStyle.SolidLine)
            painter.strokePath(path, bold_pen)
