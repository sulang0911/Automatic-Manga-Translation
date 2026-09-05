"""
app/core/typography/engine.py
Unified TypographyEngine coordinating font auto-fitting, line breaking,
vertical RTL layout, and stroke rendering onto manga pages.
"""
import os
import sys
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw

from app.core.models import (
    TranslationBlock, StyleConfig, TextDirection,
    TextColorMode, BgColorMode, StrokeMode, OnomatopoeiaMode
)
from app.core.typography.auto_fit import AutoFitEngine, LayoutResult, AutoFitResult
from app.core.typography.line_breaker import LineBreaker, TextWidthMeasurer
from app.core.typography.vertical_layout import VerticalLayoutEngine
from app.core.typography.stroke_renderer import (
    StrokeRenderer, StrokeStyle, DropShadowStyle
)
from app.core.inpaint.color_analyzer import get_background_color_rgb

logger = logging.getLogger(__name__)


class PilTextMeasurer(TextWidthMeasurer):
    """Measures text width using PIL FreeType font instances."""
    def __init__(self, font_loader: Callable[[float], ImageFont.FreeTypeFont], is_bold: bool = False):
        self.font_loader = font_loader
        self.is_bold = is_bold

    def measure_width(self, text: str, font_size: float) -> float:
        bold_pad = (font_size * 0.09) if self.is_bold else 0.0
        try:
            font = self.font_loader(font_size)
            bbox = font.getbbox(text)
            return float(bbox[2] - bbox[0]) + (bold_pad * len(text))
        except Exception:
            # Fallback heuristic: 1.0em for CJK, 0.55em for Latin
            import re
            w = 0.0
            for ch in text:
                if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', ch):
                    w += font_size * 1.0 + bold_pad
                else:
                    w += font_size * 0.55 + bold_pad
            return w


class TypographyLayoutEvaluator:
    """Evaluates whether text at a given font size fits inside width/height bounds."""
    def __init__(self, font_loader: Callable[[float], ImageFont.FreeTypeFont], line_breaker: LineBreaker, vertical_layout: VerticalLayoutEngine, is_bold: bool = False):
        self.font_loader = font_loader
        self.line_breaker = line_breaker
        self.vertical_layout = vertical_layout
        self.is_bold = is_bold

    def evaluate(self, text: str, font_size: float, max_w: float, max_h: float, is_vertical: bool) -> LayoutResult:
        measurer = PilTextMeasurer(self.font_loader, is_bold=self.is_bold)
        if is_vertical:
            raw_cols = self.vertical_layout.wrap_vertical_columns(text, max_h, font_size)
            col_width = font_size * self.vertical_layout.column_spacing_ratio
            char_h = font_size * self.vertical_layout.char_spacing_ratio

            total_w = len(raw_cols) * col_width
            max_col_h = max((len(c) * char_h for c in raw_cols), default=0.0)

            fits = (total_w <= max_w) and (max_col_h <= max_h)
            overflow_x = max(0.0, total_w - max_w)
            overflow_y = max(0.0, max_col_h - max_h)
            lines = ["".join(c) for c in raw_cols]

            return LayoutResult(
                fits=fits,
                total_width=total_w,
                total_height=max_col_h,
                lines_or_columns=lines,
                font_size=font_size,
                overflow_x=overflow_x,
                overflow_y=overflow_y
            )
        else:
            lines = self.line_breaker.wrap_text(text, max_w, font_size, measurer)
            line_height = font_size * 1.20
            total_h = len(lines) * line_height
            max_line_w = max((measurer.measure_width(line, font_size) for line in lines), default=0.0)

            fits = (max_line_w <= max_w) and (total_h <= max_h)
            overflow_x = max(0.0, max_line_w - max_w)
            overflow_y = max(0.0, total_h - max_h)

            return LayoutResult(
                fits=fits,
                total_width=max_line_w,
                total_height=total_h,
                lines_or_columns=lines,
                font_size=font_size,
                overflow_x=overflow_x,
                overflow_y=overflow_y
            )


class TypographyEngine:
    """
    Unified high-level Typography Engine for Automatic Manga Translation.
    Handles auto-fitting, line wrapping, vertical RTL layouts, and font rendering.
    """

    SYSTEM_FONT_MAP = {
        # Standard Chinese / System fonts
        "microsoft yahei": "msyh.ttc",
        "ms yahei": "msyh.ttc",
        "微软雅黑": "msyh.ttc",
        "simhei": "simhei.ttf",
        "黑体": "simhei.ttf",
        "simsun": "simsun.ttc",
        "宋体": "simsun.ttc",
        "kaiti": "simkai.ttf",
        "楷体": "simkai.ttf",
        "ms gothic": "msgothic.ttc",
        "arial": "arial.ttf",
        "segoe ui": "segoeui.ttf",

        # Cute Manga & Anime Specialty Fonts
        "霞鹜文楷": "LXGWWenKaiLite-Regular.ttf",
        "lxgw wenkai": "LXGWWenKaiLite-Regular.ttf",
        "lxgwwenkai": "LXGWWenKaiLite-Regular.ttf",
        "可爱字体": "LXGWWenKaiLite-Regular.ttf",
        "可爱": "LXGWWenKaiLite-Regular.ttf",
        "萌系": "LXGWWenKaiLite-Regular.ttf",
        "萌系字体": "LXGWWenKaiLite-Regular.ttf",
        "日漫可爱": "LXGWWenKaiLite-Regular.ttf",
        "得意黑": "SmileySans-Oblique.ttf",
        "smileysans": "SmileySans-Oblique.ttf",
        "smiley sans": "SmileySans-Oblique.ttf",
        "幼圆": "SIMYOU.TTF",
        "youyuan": "SIMYOU.TTF",
        "comic sans ms": "comic.ttf",
        "comic sans": "comic.ttf",
        "segoe print": "segoepr.ttf",
        "segoe script": "segoesc.ttf",
        "ink free": "Inkfree.ttf",
        "华文楷体": "STKAITI.TTF",
        "stkaiti": "STKAITI.TTF",
        "华文细黑": "STXIHEI.TTF",
        "stxihei": "STXIHEI.TTF",
    }

    def __init__(self, default_font_family: str = "霞鹜文楷"):
        self.default_font_family = default_font_family
        self.auto_fit_engine = AutoFitEngine()
        self.line_breaker = LineBreaker()
        self.vertical_layout = VerticalLayoutEngine()
        self._font_cache: Dict[Tuple[str, int], Any] = {}

    def resolve_font_path(self, font_family: str) -> Optional[str]:
        """Finds true font file path on system or local assets/fonts directory."""
        # Strip descriptive suffix if present, e.g. "霞鹜文楷 (日漫萌系)" -> "霞鹜文楷"
        clean_name = font_family.split("(")[0].split("（")[0].strip()
        key = clean_name.lower()
        filename = self.SYSTEM_FONT_MAP.get(key, f"{clean_name}.ttf")

        # 1. Project local assets/fonts folder (for bundled cute fonts like LXGW WenKai / SmileySans)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        local_candidate = os.path.join(repo_root, "assets", "fonts", filename)
        if os.path.exists(local_candidate):
            return local_candidate

        cwd_local = os.path.join(os.getcwd(), "assets", "fonts", filename)
        if os.path.exists(cwd_local):
            return cwd_local

        # 2. Standard Windows fonts path
        win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        candidate = os.path.join(win_fonts, filename)
        if os.path.exists(candidate):
            return candidate

        # 3. Fallback to bundled cute font, msyh.ttc or simhei.ttf
        if os.path.exists(local_candidate):
            return local_candidate
        for fallback in ["LXGWWenKaiLite-Regular.ttf", "msyh.ttc", "simhei.ttf", "arial.ttf"]:
            p_local = os.path.join(repo_root, "assets", "fonts", fallback)
            if os.path.exists(p_local):
                return p_local
            p_win = os.path.join(win_fonts, fallback)
            if os.path.exists(p_win):
                return p_win
        return None

    def get_font(self, font_family: str, size: float) -> ImageFont.FreeTypeFont:
        int_size = max(6, int(round(size)))
        cache_key = (font_family.lower(), int_size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font_path = self.resolve_font_path(font_family)
        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, int_size)
            except Exception as e:
                logger.warning(f"Could not load font {font_path}: {e}")

        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                pass

        self._font_cache[cache_key] = font
        return font

    def render_translations(
        self,
        base_image: np.ndarray,
        blocks: List[Any],
        style_config: Any = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> np.ndarray:
        """
        Unified rendering entry point compatible with both dict and TranslationBlock blocks,
        and full app config dict, nested style dict, or StyleConfig instance.
        """
        if base_image is None or base_image.size == 0 or not blocks:
            return base_image.copy() if base_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)

        model_blocks = [
            b if isinstance(b, TranslationBlock) else TranslationBlock.from_dict(b)
            for b in blocks
        ]

        if isinstance(style_config, StyleConfig):
            cfg = style_config
        elif isinstance(style_config, dict):
            # Extract nested "style" if passed full config dict
            style_data = style_config.get("style", style_config) if "style" in style_config and isinstance(style_config["style"], dict) else style_config
            cfg = StyleConfig.from_dict(style_data)
        else:
            cfg = StyleConfig()

        return self.render_page(base_image, model_blocks, cfg, progress_callback=progress_callback)

    def render_page(
        self,
        erased_image: np.ndarray,
        blocks: List[TranslationBlock],
        style_config: Optional[StyleConfig] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> np.ndarray:
        """
        Renders localized text onto the erased background image.
        Returns the completed manga page as an OpenCV BGR image (uint8).
        """
        if erased_image is None or erased_image.size == 0 or not blocks:
            return erased_image.copy() if erased_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)

        cfg = style_config or StyleConfig()
        h_img, w_img = erased_image.shape[:2]

        # QR code detection to avoid drawing text over QR codes
        qr_regions = []
        try:
            from app.core.ocr.qr_filter import QRCodeFilter
            filt = QRCodeFilter()
            qr_regions = filt.detect_regions(erased_image)
        except Exception:
            qr_regions = []

        # Convert to PIL RGBA image for high-quality antialiased text drawing
        rgb_img = cv2.cvtColor(erased_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img).convert("RGBA")

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / total_blocks) * 100), f"渲染气泡文字 ({idx+1}/{total_blocks})...")

            text = block.translated_text or block.original_text
            if not text or not text.strip():
                continue

            if block.type == "onomatopoeia" and cfg.onomatopoeia_mode == OnomatopoeiaMode.IGNORE.value:
                continue

            # Compute pixel bounding box
            x, y, w, h = block.to_pixel_rect(w_img, h_img)
            if w <= 4 or h <= 4:
                continue

            # Prevent drawing text over detected QR code boundaries
            if qr_regions:
                cx_check = x + w / 2.0
                cy_check = y + h / 2.0
                is_over_qr = False
                for qreg in qr_regions:
                    qx1, qy1, qx2, qy2 = qreg.bbox
                    if qx1 <= cx_check <= qx2 and qy1 <= cy_check <= qy2:
                        is_over_qr = True
                        break
                    # Area overlap check
                    ix1 = max(x, qx1)
                    iy1 = max(y, qy1)
                    ix2 = min(x + w, qx2)
                    iy2 = min(y + h, qy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if (inter_area / float(w * h)) > 0.60:
                            is_over_qr = True
                            break
                if is_over_qr:
                    continue

            # Check rotation angle
            effective_angle = block.get_effective_angle() if hasattr(block, "get_effective_angle") else (
                block.angle_override if block.angle_override is not None else getattr(block, "angle", 0.0)
            )
            effective_angle = float(effective_angle or 0.0)
            is_rotated = abs(effective_angle) >= 15.0

            cx = x + w / 2.0
            cy = y + h / 2.0

            # Compute oriented baseline dimensions (L, H)
            poly = block.to_pixel_polygon(w_img, h_img) if hasattr(block, "to_pixel_polygon") else None
            if is_rotated and poly and len(poly) >= 4:
                pts = np.array(poly, dtype=np.float32)
                s1 = float(np.linalg.norm(pts[1] - pts[0]))
                s2 = float(np.linalg.norm(pts[3] - pts[0]))
                if block.direction == TextDirection.HORIZONTAL.value or (block.direction != TextDirection.VERTICAL.value and w >= h):
                    L = max(s1, s2)
                    H = min(s1, s2)
                elif block.direction == TextDirection.VERTICAL.value or (block.direction != TextDirection.HORIZONTAL.value and h > w * 1.5):
                    L = min(s1, s2)
                    H = max(s1, s2)
                else:
                    L, H = s1, s2
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                if L < 4 or H < 4:
                    L, H = float(w), float(h)
            elif is_rotated:
                rad = math.radians(abs(effective_angle))
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                denom = max(0.1, cos_a * cos_a - sin_a * sin_a)
                if abs(denom) > 0.15:
                    L = max(10.0, (w * cos_a - h * sin_a) / denom)
                    H = max(10.0, (h * cos_a - w * sin_a) / denom)
                else:
                    L = max(10.0, math.sqrt(w * w + h * h) * 0.85)
                    H = max(10.0, float(min(w, h)))
            else:
                L, H = float(w), float(h)

            # Determine text direction
            # Key fix: OCR records direction from ORIGINAL (Japanese/source) text layout.
            # When rendering translated Chinese text, we re-evaluate based on the rendered
            # box shape and target text language, NOT the OCR source-direction flag.
            # A box is only treated as vertical if it is clearly taller than wide (ratio > 1.5)
            # AND the translated text is CJK. Landscape bubbles always use horizontal layout.
            text_is_cjk = LineBreaker.is_cjk(text)
            box_is_tall = (H > L * 1.5) if is_rotated else (h > w * 1.5)
            if block.direction == TextDirection.VERTICAL.value:
                # Override: if the box is landscape or text is not CJK, switch to horizontal
                is_vertical = box_is_tall and text_is_cjk
            elif block.direction == TextDirection.HORIZONTAL.value:
                is_vertical = False
            else:
                # AUTO: conservative vertical only when box is clearly portrait + CJK text
                is_vertical = box_is_tall and text_is_cjk

            # Orientation-aware dimension for layout and auto-fit:
            box_w_eff = L if not is_vertical else H
            box_h_eff = H if not is_vertical else L

            # Reflow fix: for horizontal CJK bubbles, strip source-language newlines.
            # LLM translations often echo the original text's line-break structure
            # (e.g. English dialogue broken into short phrases). Each \n-separated
            # segment becomes its own "paragraph" in wrap_text, yielding many
            # short lines (2-5 chars each). Fix: join all segments into one
            # continuous string so line_breaker can reflow to fit the bubble width.
            render_text = text
            if not is_vertical and text_is_cjk:
                segs = [s.strip() for s in text.split("\n") if s.strip()]
                if len(segs) > 1:
                    render_text = "".join(segs)

            # Font styling resolution
            font_family = block.font_family_override or cfg.font_family
            is_bold = bool(block.font_bold_override if block.font_bold_override is not None else cfg.font_bold)
            font_loader = lambda fs: self.get_font(font_family, fs)

            evaluator = TypographyLayoutEvaluator(font_loader, self.line_breaker, self.vertical_layout, is_bold=is_bold)

            # Font size calculation along oriented baseline dimensions
            if block.font_size_override is not None:
                font_size = float(block.font_size_override)
            elif cfg.auto_fit_font_size:
                fit_res = self.auto_fit_engine.fit_text(
                    text=render_text,
                    box_w=float(box_w_eff),
                    box_h=float(box_h_eff),
                    is_vertical=is_vertical,
                    evaluator=evaluator,
                    font_scale=cfg.font_size_scale
                )
                font_size = fit_res.optimal_font_size
            else:
                base_dim = min(box_w_eff, box_h_eff)
                font_size = max(float(cfg.min_font_size), min(float(cfg.max_font_size), base_dim * 0.22 * cfg.font_size_scale))

            font = self.get_font(font_family, font_size)

            # Sample background color and luminance underneath this bubble
            bg_rgb = (255, 255, 255)
            bg_lum = 255.0
            bx1, by1 = max(0, int(x)), max(0, int(y))
            bx2, by2 = min(w_img, int(x + w)), min(h_img, int(y + h))
            if bx2 > bx1 and by2 > by1:
                crop = erased_image[by1:by2, bx1:bx2]
                bg_rgb = get_background_color_rgb(crop)
                bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]

            # Color resolution
            if block.text_color_override:
                text_color_hex = block.text_color_override
            elif cfg.text_color_mode == TextColorMode.CUSTOM.value:
                text_color_hex = cfg.custom_text_color
            else:
                # Automatic / original mode: check candidate color
                cand_hex = block.text_color or "#000000"
                cr = int(cand_hex[1:3], 16) if len(cand_hex) >= 7 else 0
                cg = int(cand_hex[3:5], 16) if len(cand_hex) >= 7 else 0
                cb = int(cand_hex[5:7], 16) if len(cand_hex) >= 7 else 0
                cand_lum = 0.299 * cr + 0.587 * cg + 0.114 * cb

                # Saturation and contrast analysis
                max_c = max(cr, cg, cb)
                min_c = min(cr, cg, cb)
                sat = (max_c - min_c) if max_c > 0 else 0
                lum_diff = abs(cand_lum - bg_lum)

                # High-contrast intelligent quantization:
                # 1. If luminance contrast against background is low (< 125)
                # 2. Or if candidate text is muddy grayscale (sat < 35 and 35 < cand_lum < 220)
                if lum_diff < 125.0 or (sat < 35 and 35 < cand_lum < 220):
                    if bg_lum < 128.0:
                        text_color_hex = "#FFFFFF"  # High contrast crisp white on dark
                    else:
                        text_color_hex = "#000000"  # High contrast crisp black on light
                else:
                    text_color_hex = cand_hex

            r = int(text_color_hex[1:3], 16) if len(text_color_hex) >= 7 else 0
            g = int(text_color_hex[3:5], 16) if len(text_color_hex) >= 7 else 0
            b = int(text_color_hex[5:7], 16) if len(text_color_hex) >= 7 else 0
            if block.type == "onomatopoeia" and cfg.onomatopoeia_mode == OnomatopoeiaMode.TRANSPARENT.value:
                fill_rgba = (r, g, b, 175)
            else:
                fill_rgba = (r, g, b, 255)

            # Stroke resolution
            stroke_mode = block.stroke_mode_override or cfg.stroke_mode
            stroke = None
            if stroke_mode == StrokeMode.AUTO.value:
                sw_param = block.stroke_width_override if block.stroke_width_override is not None else cfg.stroke_width
                stroke = StrokeRenderer.get_auto_contrast_stroke((r, g, b), font_size, bg_rgb=bg_rgb, base_stroke_w=sw_param)
            elif stroke_mode == StrokeMode.MANUAL.value:
                stroke_hex = block.stroke_color_override or cfg.stroke_color
                sr = int(stroke_hex[1:3], 16) if len(stroke_hex) >= 7 else 255
                sg = int(stroke_hex[3:5], 16) if len(stroke_hex) >= 7 else 255
                sb = int(stroke_hex[5:7], 16) if len(stroke_hex) >= 7 else 255
                sw = block.stroke_width_override if block.stroke_width_override is not None else cfg.stroke_width
                stroke = StrokeStyle(color_rgba=(sr, sg, sb, 217), width=float(sw))

            # Drop shadow
            shadow = StrokeRenderer.get_drop_shadow_params(font_size, enabled=cfg.text_shadow)

            if is_rotated:
                # Render onto local transparent canvas with padding margin
                pad_margin = int(font_size * 0.5) + 12
                canvas_w = int(math.ceil(box_w_eff)) + 2 * pad_margin
                canvas_h = int(math.ceil(box_h_eff)) + 2 * pad_margin
                draw_target = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                rel_x, rel_y = float(pad_margin), float(pad_margin)
                draw_box_w, draw_box_h = float(box_w_eff), float(box_h_eff)
            else:
                draw_target = pil_img
                rel_x, rel_y = float(x), float(y)
                draw_box_w, draw_box_h = float(w), float(h)

            # Draw onto PIL Image / draw_target
            if is_vertical:
                cols, tot_w, tot_h = self.vertical_layout.compute_layout(
                    text=render_text,
                    font_size=font_size,
                    box_x=rel_x,
                    box_y=rel_y,
                    box_w=draw_box_w,
                    box_h=draw_box_h
                )
                for col in cols:
                    for glyph in col.glyphs:
                        # Draw glyph centered at (x, y)
                        gx = glyph.x + glyph.offset_x - font_size * 0.45
                        gy = glyph.y + glyph.offset_y - font_size * 0.50
                        StrokeRenderer.render_text_pil(
                            target_image=draw_target,
                            text=glyph.char,
                            x=gx,
                            y=gy,
                            font=font,
                            fill_rgba=fill_rgba,
                            stroke=stroke,
                            shadow=shadow,
                            is_bold=is_bold
                        )
            else:
                measurer = PilTextMeasurer(font_loader, is_bold=is_bold)
                px, py = self.auto_fit_engine.calculate_padding(draw_box_w, draw_box_h)
                avail_w = max(4.0, draw_box_w - 2 * px)
                lines = self.line_breaker.wrap_text(render_text, avail_w, font_size, measurer)
                line_h = font_size * cfg.line_spacing
                tot_h = len(lines) * line_h
                start_y = rel_y + (draw_box_h - tot_h) / 2.0

                for i, line in enumerate(lines):
                    line_w = measurer.measure_width(line, font_size)
                    line_x = rel_x + (draw_box_w - line_w) / 2.0
                    line_y = start_y + i * line_h
                    StrokeRenderer.render_text_pil(
                        target_image=draw_target,
                        text=line,
                        x=line_x,
                        y=line_y,
                        font=font,
                        fill_rgba=fill_rgba,
                        stroke=stroke,
                        shadow=shadow,
                        is_bold=is_bold
                    )

            if is_rotated:
                # Rotate by -effective_angle (clockwise in image coords) and paste centered
                rotated_layer = draw_target.rotate(
                    -effective_angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=True
                )
                paste_x = int(round(cx - rotated_layer.width / 2.0))
                paste_y = int(round(cy - rotated_layer.height / 2.0))
                pil_img.paste(rotated_layer, (paste_x, paste_y), rotated_layer)

        # Convert back to OpenCV BGR
        res_rgb = np.array(pil_img.convert("RGB"))
        res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

        if progress_callback:
            progress_callback(100, "文字渲染完成")

        return res_bgr
