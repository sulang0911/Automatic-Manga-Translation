"""
app/core/typography/engine.py
Unified TypographyEngine coordinating font auto-fitting, line breaking,
vertical RTL layout, and stroke rendering onto manga pages.
"""
import os
import sys
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

logger = logging.getLogger(__name__)


class PilTextMeasurer(TextWidthMeasurer):
    """Measures text width using PIL FreeType font instances."""
    def __init__(self, font_loader: Callable[[float], ImageFont.FreeTypeFont]):
        self.font_loader = font_loader

    def measure_width(self, text: str, font_size: float) -> float:
        try:
            font = self.font_loader(font_size)
            bbox = font.getbbox(text)
            return float(bbox[2] - bbox[0])
        except Exception:
            # Fallback heuristic: 1.0em for CJK, 0.55em for Latin
            import re
            w = 0.0
            for ch in text:
                if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', ch):
                    w += font_size * 1.0
                else:
                    w += font_size * 0.55
            return w


class TypographyLayoutEvaluator:
    """Evaluates whether text at a given font size fits inside width/height bounds."""
    def __init__(self, font_loader: Callable[[float], ImageFont.FreeTypeFont], line_breaker: LineBreaker, vertical_layout: VerticalLayoutEngine):
        self.font_loader = font_loader
        self.line_breaker = line_breaker
        self.vertical_layout = vertical_layout

    def evaluate(self, text: str, font_size: float, max_w: float, max_h: float, is_vertical: bool) -> LayoutResult:
        measurer = PilTextMeasurer(self.font_loader)
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

    def __init__(self, default_font_family: str = "Microsoft YaHei"):
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

            # Determine text direction
            if block.direction == TextDirection.VERTICAL.value:
                is_vertical = True
            elif block.direction == TextDirection.HORIZONTAL.value:
                is_vertical = False
            else:
                # Auto direction heuristic: vertical if h > w * 1.15 and text is CJK
                is_vertical = (h > w * 1.15) and LineBreaker.is_cjk(text)

            # Font styling resolution
            font_family = block.font_family_override or cfg.font_family
            font_loader = lambda fs: self.get_font(font_family, fs)

            evaluator = TypographyLayoutEvaluator(font_loader, self.line_breaker, self.vertical_layout)

            # Font size calculation
            if block.font_size_override is not None:
                font_size = float(block.font_size_override)
            elif cfg.auto_fit_font_size:
                fit_res = self.auto_fit_engine.fit_text(
                    text=text,
                    box_w=float(w),
                    box_h=float(h),
                    is_vertical=is_vertical,
                    evaluator=evaluator,
                    font_scale=cfg.font_size_scale
                )
                font_size = fit_res.optimal_font_size
            else:
                base_dim = min(w, h)
                font_size = max(float(cfg.min_font_size), min(float(cfg.max_font_size), base_dim * 0.22 * cfg.font_size_scale))

            font = self.get_font(font_family, font_size)

            # Color resolution
            text_color_hex = block.text_color_override or (
                cfg.custom_text_color if cfg.text_color_mode == TextColorMode.CUSTOM.value else block.text_color
            )
            r = int(text_color_hex[1:3], 16) if len(text_color_hex) >= 7 else 0
            g = int(text_color_hex[3:5], 16) if len(text_color_hex) >= 7 else 0
            b = int(text_color_hex[5:7], 16) if len(text_color_hex) >= 7 else 0
            fill_rgba = (r, g, b, 255)

            # Stroke resolution
            stroke_mode = block.stroke_mode_override or cfg.stroke_mode
            stroke = None
            if stroke_mode == StrokeMode.AUTO.value:
                stroke = StrokeRenderer.get_auto_contrast_stroke((r, g, b), font_size)
            elif stroke_mode == StrokeMode.MANUAL.value:
                stroke_hex = block.stroke_color_override or cfg.stroke_color
                sr = int(stroke_hex[1:3], 16) if len(stroke_hex) >= 7 else 255
                sg = int(stroke_hex[3:5], 16) if len(stroke_hex) >= 7 else 255
                sb = int(stroke_hex[5:7], 16) if len(stroke_hex) >= 7 else 255
                sw = block.stroke_width_override if block.stroke_width_override is not None else cfg.stroke_width
                stroke = StrokeStyle(color_rgba=(sr, sg, sb, 217), width=float(sw))

            # Drop shadow
            shadow = StrokeRenderer.get_drop_shadow_params(font_size, enabled=cfg.text_shadow)

            # Draw onto PIL Image
            if is_vertical:
                cols, tot_w, tot_h = self.vertical_layout.compute_layout(
                    text=text,
                    font_size=font_size,
                    box_x=float(x),
                    box_y=float(y),
                    box_w=float(w),
                    box_h=float(h)
                )
                for col in cols:
                    for glyph in col.glyphs:
                        # Draw glyph centered at (x, y)
                        gx = glyph.x + glyph.offset_x - font_size * 0.45
                        gy = glyph.y + glyph.offset_y - font_size * 0.50
                        StrokeRenderer.render_text_pil(
                            target_image=pil_img,
                            text=glyph.char,
                            x=gx,
                            y=gy,
                            font=font,
                            fill_rgba=fill_rgba,
                            stroke=stroke,
                            shadow=shadow
                        )
            else:
                measurer = PilTextMeasurer(font_loader)
                px, py = self.auto_fit_engine.calculate_padding(w, h)
                avail_w = max(4.0, w - 2 * px)
                lines = self.line_breaker.wrap_text(text, avail_w, font_size, measurer)
                line_h = font_size * cfg.line_spacing
                tot_h = len(lines) * line_h
                start_y = y + (h - tot_h) / 2.0

                for i, line in enumerate(lines):
                    line_w = measurer.measure_width(line, font_size)
                    line_x = x + (w - line_w) / 2.0
                    line_y = start_y + i * line_h
                    StrokeRenderer.render_text_pil(
                        target_image=pil_img,
                        text=line,
                        x=line_x,
                        y=line_y,
                        font=font,
                        fill_rgba=fill_rgba,
                        stroke=stroke,
                        shadow=shadow
                    )

        # Convert back to OpenCV BGR
        res_rgb = np.array(pil_img.convert("RGB"))
        res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

        if progress_callback:
            progress_callback(100, "文字渲染完成")

        return res_bgr
