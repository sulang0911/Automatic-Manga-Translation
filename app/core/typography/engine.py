"""
app/core/typography/engine.py
Unified TypographyEngine coordinating font auto-fitting, line breaking,
vertical RTL layout, and stroke rendering onto manga pages.
"""
import os
import sys
import re
import math
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import Counter
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw

from app.core.models import (
    TranslationBlock, StyleConfig, TextDirection, BlockType,
    TextColorMode, BgColorMode, StrokeMode, OnomatopoeiaMode
)
from app.core.typography.auto_fit import AutoFitEngine, LayoutResult, AutoFitResult
from app.core.typography.line_breaker import LineBreaker, TextWidthMeasurer
from app.core.typography.vertical_layout import VerticalLayoutEngine
from app.core.typography.stroke_renderer import (
    StrokeRenderer, StrokeStyle, DropShadowStyle
)
from app.core.inpaint.color_analyzer import (
    get_background_color_rgb,
    collapse_color_turns,
    _classify_color_family,
)

logger = logging.getLogger(__name__)

SPEAKER_LINE_RE = re.compile(
    r'^\s*[\(\[\{（【〔]([A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af_.\- ]{1,14})[\)\]\}）】〕]'
)
STAGE_DIRECTION_ONLY_RE = re.compile(
    r'^\s*[\(\[\{（【〔][^)\]\}）】〕]{1,24}[\)\]\}）】〕]\s*$'
)
SPEAKER_SPLIT_RE = re.compile(
    r'(?:^|(?<=[。！？!?.”"」』）】〕\s]))[ \t]*(?=[\(\[\{（【〔][^)\]\}）】〕\n]{1,14}[\)\]\}）】〕]\s*[“"「『（\(]?)'
)


def _extract_speaker_tag(line: str) -> Optional[str]:
    """Extracts normalized speaker name from a line starting with (Speaker) followed by dialogue."""
    if not line or STAGE_DIRECTION_ONLY_RE.match(line):
        return None
    m = SPEAKER_LINE_RE.match(line)
    if not m:
        return None
    return m.group(1).strip().lower()


def _is_stage_direction_para(line: str) -> bool:
    """Returns True if the paragraph is purely a parenthetical stage note like '(略显失望)' with no dialogue."""
    return bool(line and STAGE_DIRECTION_ONLY_RE.match(line.strip()))


def format_speaker_dialogue(text: str, original_text: str = "") -> Tuple[str, int]:
    """
    Detects multi-speaker dialogue lines (e.g. '(锡)「...」(俊)「...」' or '(Suk) "..." (Jun) "..."')
    and splits each speaker turn onto its own line with '\\n'.
    Returns (formatted_text, speaker_turn_count).
    """
    if not text:
        return text, 0

    norm = SPEAKER_SPLIT_RE.sub("\n", text.strip())
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]
    speaker_cnt = sum(1 for ln in lines if SPEAKER_LINE_RE.match(ln))
    if speaker_cnt >= 2:
        return "\n".join(lines), speaker_cnt

    if original_text:
        orig_norm = SPEAKER_SPLIT_RE.sub("\n", original_text.strip())
        orig_lines = [ln.strip() for ln in orig_norm.split("\n") if ln.strip()]
        orig_spk = sum(1 for ln in orig_lines if SPEAKER_LINE_RE.match(ln))
        t_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if orig_spk >= 2 and len(t_lines) >= 2:
            return "\n".join(t_lines), len(t_lines)

    return text, speaker_cnt


def compute_smart_line_spacing(
    lines_count: int,
    font_size: float,
    box_h: float,
    base_spacing: float = 1.30,
    auto_enabled: bool = True
) -> float:
    """
    Computes optimal line spacing multiplier:
    - When auto_enabled is False, returns base_spacing (default 1.30x).
    - When auto_enabled is True, dynamically adapts line spacing around base_spacing (1.30x)
      based on line count and box vertical headroom so multi-line paragraphs breathe naturally
      without overflowing tight bounding boxes.
    """
    if not auto_enabled:
        return float(base_spacing)
    if lines_count <= 1:
        return 1.15
    raw_text_h = max(1.0, lines_count * font_size)
    headroom_ratio = box_h / raw_text_h
    if lines_count == 2:
        return max(1.20, min(float(base_spacing), headroom_ratio * 0.96))
    elif lines_count == 3:
        return max(1.18, min(float(base_spacing), headroom_ratio * 0.95))
    else:
        return max(1.16, min(float(base_spacing), headroom_ratio * 0.94))


class PilTextMeasurer(TextWidthMeasurer):
    """Measures text width using PIL FreeType font instances."""
    def __init__(self, font_loader: Callable[[float], ImageFont.FreeTypeFont], is_bold: bool = False):
        self.font_loader = font_loader
        self.is_bold = is_bold

    def measure_width(self, text: str, font_size: float) -> float:
        bold_pad = 0.0
        try:
            font = self.font_loader(font_size)
            bbox = font.getbbox(text)
            return float(bbox[2] - bbox[0]) + (bold_pad * len(text))
        except Exception:
            w = 0.0
            for ch in text:
                if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', ch):
                    w += font_size * 1.0 + bold_pad
                else:
                    w += font_size * 0.55 + bold_pad
            return w


class TypographyLayoutEvaluator:
    """Evaluates whether text at a given font size fits inside width/height bounds."""
    def __init__(
        self,
        font_loader: Callable[[float], ImageFont.FreeTypeFont],
        line_breaker: LineBreaker,
        vertical_layout: VerticalLayoutEngine,
        is_bold: bool = False,
        line_spacing: float = 1.30,
        auto_line_spacing: bool = True,
        speaker_turn_count: int = 0
    ):
        self.font_loader = font_loader
        self.line_breaker = line_breaker
        self.vertical_layout = vertical_layout
        self.is_bold = is_bold
        self.line_spacing = line_spacing
        self.auto_line_spacing = auto_line_spacing
        self.speaker_turn_count = speaker_turn_count

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
            eff_spacing = compute_smart_line_spacing(
                len(lines), font_size, max_h,
                base_spacing=self.line_spacing,
                auto_enabled=self.auto_line_spacing
            )
            line_height = font_size * eff_spacing
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


def _compute_source_dialogue_cap(
    block: TranslationBlock,
    w_img: int,
    h_img: int,
    box_w: float,
    box_h: float,
    is_vertical: bool,
    orig_is_horizontal_script: bool,
    hard_max: float,
) -> float:
    """
    Estimates a natural font-size upper cap (at font_size_scale=1.0) so concise
    translations in oversized or sparse boxes stay proportional to page resolution
    and original dialogue density instead of inflating to the global ceiling.
    """
    if getattr(block, "type", BlockType.BUBBLE.value) == BlockType.ONOMATOPOEIA.value:
        return hard_max

    min_dim = float(max(1, min(w_img, h_img)))
    # Resolution-proportional ceiling: ~32px@720p, ~45px@1000p, ~54px@1200p, 72px@4K
    cap = min(hard_max, max(24.0, min_dim * 0.045))

    orig_txt = (getattr(block, "original_text", "") or "").strip()
    orig_lines = [ln.strip() for ln in orig_txt.splitlines() if ln.strip()]
    n_lines = max(1, int(getattr(block, "line_count", 1) or 1), len(orig_lines))

    if not is_vertical and orig_is_horizontal_script and len(orig_txt) >= 24:
        # Estimate Latin/Hangul line pitch from box area vs average glyph footprint
        char_count = sum(len(ln) for ln in orig_lines) if orig_lines else len(orig_txt)
        est_pitch = math.sqrt((max(1.0, box_w) * max(1.0, box_h)) / max(1.0, char_count * 0.625))
        cap = min(cap, max(12.0, est_pitch * 1.15))
    elif is_vertical and n_lines >= 2:
        cap = min(cap, max(12.0, (max(1.0, box_w) / float(n_lines)) * 1.15))
    elif not is_vertical and n_lines >= 2 and len(orig_txt) >= 8:
        cap = min(cap, max(12.0, (max(1.0, box_h) / float(n_lines)) * 1.05))

    return max(8.0, cap)


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

    _GLOBAL_FONT_CACHE: Dict[Tuple[Any, ...], Any] = {}
    _GLOBAL_FONT_PATH_CACHE: Dict[str, Optional[str]] = {}
    _FONT_CACHE_LOCK = threading.Lock()
    _LAST_QR_SIG: Optional[Tuple[Any, ...]] = None
    _LAST_QR_REGIONS: List[Any] = []
    _QR_CACHE_LOCK = threading.Lock()

    def __init__(self, default_font_family: str = "霞鹜文楷"):
        self.default_font_family = default_font_family
        self.auto_fit_engine = AutoFitEngine()
        self.line_breaker = LineBreaker()
        self.vertical_layout = VerticalLayoutEngine()
        self._font_cache: Dict[Tuple[Any, ...], Any] = TypographyEngine._GLOBAL_FONT_CACHE

    def resolve_font_path(self, font_family: str) -> Optional[str]:
        """Finds true font file path on system or local assets/fonts directory."""
        cache_key = font_family.strip().lower()
        with TypographyEngine._FONT_CACHE_LOCK:
            if cache_key in TypographyEngine._GLOBAL_FONT_PATH_CACHE:
                cached_p = TypographyEngine._GLOBAL_FONT_PATH_CACHE[cache_key]
                if cached_p is None or os.path.exists(cached_p):
                    return cached_p

        clean_name = font_family.split("(")[0].split("（")[0].strip()
        key = clean_name.lower()
        filename = self.SYSTEM_FONT_MAP.get(key, f"{clean_name}.ttf")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")

        resolved: Optional[str] = None
        candidates = [
            os.path.join(repo_root, "assets", "fonts", filename),
            os.path.join(os.getcwd(), "assets", "fonts", filename),
            os.path.join(win_fonts, filename),
        ]
        for fallback in ["LXGWWenKaiLite-Regular.ttf", "msyh.ttc", "simhei.ttf", "arial.ttf"]:
            candidates.append(os.path.join(repo_root, "assets", "fonts", fallback))
            candidates.append(os.path.join(win_fonts, fallback))
        candidates.extend([
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ])

        for cand in candidates:
            if os.path.exists(cand):
                resolved = cand
                break

        with TypographyEngine._FONT_CACHE_LOCK:
            TypographyEngine._GLOBAL_FONT_PATH_CACHE[cache_key] = resolved
        return resolved

    def get_font(self, font_family: str, size: float) -> ImageFont.FreeTypeFont:
        int_size = max(6, int(round(size)))
        tid = threading.get_ident()
        cache_key = (tid, font_family.lower(), int_size)
        with TypographyEngine._FONT_CACHE_LOCK:
            cached = self._font_cache.get(cache_key)
        if cached is not None:
            return cached

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

        with TypographyEngine._FONT_CACHE_LOCK:
            existing = self._font_cache.get(cache_key)
            if existing is not None:
                return existing
            if len(self._font_cache) >= 256:
                for k in list(self._font_cache.keys())[:48]:
                    self._font_cache.pop(k, None)
            self._font_cache[cache_key] = font
        return font

    @staticmethod
    def _resolve_color_hex(
        cand_hex: str,
        bg_lum: float,
        is_dark_bg_block: bool,
        cfg: StyleConfig,
        text_color_override: Optional[str] = None
    ) -> str:
        if text_color_override:
            return text_color_override
        if cfg.text_color_mode == TextColorMode.CUSTOM.value:
            return cfg.custom_text_color

        cand_hex = cand_hex or "#000000"
        cr = int(cand_hex[1:3], 16) if len(cand_hex) >= 7 else 0
        cg = int(cand_hex[3:5], 16) if len(cand_hex) >= 7 else 0
        cb = int(cand_hex[5:7], 16) if len(cand_hex) >= 7 else 0
        cand_lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
        max_c = max(cr, cg, cb)
        min_c = min(cr, cg, cb)
        sat = (max_c - min_c) if max_c > 0 else 0
        lum_diff = abs(cand_lum - bg_lum)

        if sat >= 50 and max_c >= 120:
            return cand_hex
        if is_dark_bg_block:
            return "#FFFFFF"
        if lum_diff < 125.0 or (sat < 35 and 35 < cand_lum < 220):
            return "#FFFFFF" if bg_lum < 128.0 else "#000000"
        return cand_hex

    @staticmethod
    def _build_page_speaker_color_map(blocks: List[TranslationBlock]) -> Dict[str, str]:
        """
        Learns a consistent mapping from speaker tag (e.g. '悠奈', '锡', '俊', 'yuna', 'suk', 'jun')
        to its extracted original hex color across all blocks on the page.
        """
        family_votes: Dict[str, Counter] = {}
        family_hexes: Dict[str, Dict[str, Counter]] = {}
        for b in blocks:
            raw_text = b.translated_text or b.original_text or ""
            if not raw_text.strip():
                continue
            formatted, _ = format_speaker_dialogue(raw_text, b.original_text or "")
            paras = [p.strip() for p in formatted.split("\n") if p.strip()]
            dialogue_paras = [p for p in paras if not _is_stage_direction_para(p)]

            line_cols = getattr(b, "line_colors", None) or []
            turns = collapse_color_turns(line_cols) if line_cols else ([b.text_color] if b.text_color else [])
            if not turns:
                continue

            if len(turns) == len(paras):
                paired = list(zip(paras, turns))
            elif len(turns) == len(dialogue_paras):
                paired = list(zip(dialogue_paras, turns))
            elif len(dialogue_paras) == 1 and len(turns) == 1:
                paired = [(dialogue_paras[0], turns[0])]
            else:
                paired = []
                for idx, p in enumerate(dialogue_paras):
                    t_idx = min(len(turns) - 1, int(round(idx * (len(turns) - 1) / max(1, len(dialogue_paras) - 1))))
                    paired.append((p, turns[t_idx]))

            for para_str, col_hex in paired:
                spk = _extract_speaker_tag(para_str)
                fam = _classify_color_family(col_hex) if col_hex else "achromatic"
                if spk and col_hex and fam not in ("light", "dark", "achromatic"):
                    family_votes.setdefault(spk, Counter())[fam] += 1
                    family_hexes.setdefault(spk, {}).setdefault(fam, Counter())[col_hex] += 1

        result: Dict[str, str] = {}
        for spk, fam_cnt in family_votes.items():
            if fam_cnt:
                top_fam = fam_cnt.most_common(1)[0][0]
                result[spk] = family_hexes[spk][top_fam].most_common(1)[0][0]
        return result

    def render_translations(
        self,
        base_image: np.ndarray,
        blocks: List[Any],
        style_config: Any = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> np.ndarray:
        if base_image is None or base_image.size == 0 or not blocks:
            return base_image.copy() if base_image is not None else np.zeros((1, 1, 3), dtype=np.uint8)

        model_blocks = [
            b if isinstance(b, TranslationBlock) else TranslationBlock.from_dict(b)
            for b in blocks
        ]

        if isinstance(style_config, StyleConfig):
            cfg = style_config
        elif isinstance(style_config, dict):
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

        qr_regions = []
        try:
            # Only scan for QR regions if at least one block has a square-ish badge aspect ratio
            has_qr_candidate = False
            for b_cand in blocks:
                bx, by, bw, bh = b_cand.to_pixel_rect(w_img, h_img)
                if bw >= 16 and bh >= 16 and bw <= int(w_img * 0.45) and bh <= int(h_img * 0.45):
                    asp = bw / float(max(1, bh))
                    if 0.65 <= asp <= 1.50 and not getattr(b_cand, "line_colors", None):
                        has_qr_candidate = True
                        break
            if has_qr_candidate:
                mid_y, mid_x = h_img // 2, w_img // 2
                img_sig = (
                    h_img, w_img,
                    int(erased_image.ctypes.data),
                    int(erased_image[0, 0, 0]),
                    int(erased_image[mid_y, mid_x, 0]),
                    int(erased_image[-1, -1, -1]),
                )
                with TypographyEngine._QR_CACHE_LOCK:
                    cached_qr = list(TypographyEngine._LAST_QR_REGIONS) if TypographyEngine._LAST_QR_SIG == img_sig else None
                if cached_qr is not None:
                    qr_regions = cached_qr
                else:
                    det = cv2.QRCodeDetector()
                    ok, decoded_info, points, _ = det.detectAndDecodeMulti(erased_image)
                    if ok and points is not None and len(points) > 0:
                        from app.core.ocr.qr_filter import QRCodeFilter
                        filt = QRCodeFilter()
                        for idx_p, pts in enumerate(points):
                            dec = decoded_info[idx_p] if idx_p < len(decoded_info) else ""
                            reg = filt._build_region("qrcode", pts, w_img, h_img, dec)
                            if reg:
                                qr_regions.append(reg)
                    with TypographyEngine._QR_CACHE_LOCK:
                        TypographyEngine._LAST_QR_SIG = img_sig
                        TypographyEngine._LAST_QR_REGIONS = list(qr_regions)
        except Exception:
            qr_regions = []

        rgb_img = cv2.cvtColor(erased_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img).convert("RGBA")

        speaker_color_map = self._build_page_speaker_color_map(blocks)

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / total_blocks) * 100), f"渲染气泡文字 ({idx+1}/{total_blocks})...")

            text = block.translated_text or block.original_text
            if not text or not text.strip():
                continue

            if block.type == "onomatopoeia" and cfg.onomatopoeia_mode == OnomatopoeiaMode.IGNORE.value:
                continue

            x, y, w, h = block.to_pixel_rect(w_img, h_img)
            if w <= 4 or h <= 4:
                continue

            if qr_regions:
                cx_check = x + w / 2.0
                cy_check = y + h / 2.0
                is_over_qr = False
                for qreg in qr_regions:
                    qx1, qy1, qx2, qy2 = qreg.bbox
                    if qx1 <= cx_check <= qx2 and qy1 <= cy_check <= qy2:
                        is_over_qr = True
                        break
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

            effective_angle = block.get_effective_angle() if hasattr(block, "get_effective_angle") else (
                block.angle_override if block.angle_override is not None else getattr(block, "angle", 0.0)
            )
            effective_angle = float(effective_angle or 0.0)
            is_rotated = abs(effective_angle) >= 15.0

            cx = x + w / 2.0
            cy = y + h / 2.0

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

            text_is_cjk = LineBreaker.is_cjk(text)
            orig_txt = getattr(block, "original_text", "") or ""
            orig_hanzi_kana = sum(1 for c in orig_txt if ('\u4e00' <= c <= '\u9fff') or ('\u3040' <= c <= '\u30ff'))
            orig_horiz_script = sum(1 for c in orig_txt if ('a' <= c <= 'z' or 'A' <= c <= 'Z' or '\uac00' <= c <= '\ud7af'))
            orig_is_horizontal_script = (orig_horiz_script >= 2 and orig_hanzi_kana <= max(1, int(orig_horiz_script * 0.1)))
            box_is_tall = (H > L * 1.5) if is_rotated else (h > w * 1.5)
            if block.direction == TextDirection.VERTICAL.value:
                is_vertical = box_is_tall and text_is_cjk
            elif block.direction == TextDirection.HORIZONTAL.value:
                is_vertical = False
            elif orig_is_horizontal_script:
                is_vertical = False
            else:
                is_vertical = box_is_tall and text_is_cjk

            box_w_eff = L
            box_h_eff = H

            # Detect multi-speaker dialogue turns (e.g. (Suk) "..." / (Jun) "..." or (锡)「...」 / (俊)「...」)
            # If the block contains >= 2 speaker turns, preserve '\n' between every speaker turn!
            # Only join '\n' into a single paragraph when it is a single-speaker speech bubble.
            formatted_text, speaker_turn_count = format_speaker_dialogue(text, block.original_text or "")
            render_text = formatted_text
            if not is_vertical and text_is_cjk and speaker_turn_count < 2:
                segs = [s.strip() for s in render_text.split("\n") if s.strip()]
                if len(segs) > 1:
                    render_text = "".join(segs)

            font_family = block.font_family_override or cfg.font_family
            is_bold = bool(block.font_bold_override if block.font_bold_override is not None else cfg.font_bold)
            font_loader = lambda fs: self.get_font(font_family, fs)
            auto_ls = getattr(cfg, "auto_line_spacing", True)

            evaluator = TypographyLayoutEvaluator(
                font_loader,
                self.line_breaker,
                self.vertical_layout,
                is_bold=is_bold,
                line_spacing=cfg.line_spacing,
                auto_line_spacing=auto_ls,
                speaker_turn_count=speaker_turn_count
            )

            is_speaker_dialogue = (speaker_turn_count >= 2) or bool(SPEAKER_LINE_RE.match(render_text))
            eff_bubble_shape = "caption" if is_speaker_dialogue else "rect"

            if block.font_size_override is not None:
                font_size = float(block.font_size_override)
            elif cfg.auto_fit_font_size:
                hard_max = float(getattr(cfg, "max_font_size", 72) or 72)
                base_cap = _compute_source_dialogue_cap(
                    block=block,
                    w_img=w_img,
                    h_img=h_img,
                    box_w=float(box_w_eff),
                    box_h=float(box_h_eff),
                    is_vertical=is_vertical,
                    orig_is_horizontal_script=orig_is_horizontal_script,
                    hard_max=hard_max,
                )
                scale_val = max(0.1, float(cfg.font_size_scale))
                eff_max = min(hard_max, base_cap * max(1.0, scale_val))
                fit_res = self.auto_fit_engine.fit_text(
                    text=render_text,
                    box_w=float(box_w_eff),
                    box_h=float(box_h_eff),
                    is_vertical=is_vertical,
                    evaluator=evaluator,
                    font_scale=scale_val,
                    bubble_shape=eff_bubble_shape,
                    min_font_size=float(cfg.min_font_size),
                    max_font_size=eff_max,
                )
                font_size = fit_res.optimal_font_size
            else:
                base_dim = min(box_w_eff, box_h_eff)
                font_size = max(float(cfg.min_font_size), min(float(cfg.max_font_size), base_dim * 0.22 * cfg.font_size_scale))

            font = self.get_font(font_family, font_size)

            stored_bg_hex = getattr(block, "bg_color_override", None) or getattr(block, "bg_color", None)
            if isinstance(block, dict):
                stored_bg_hex = block.get("bg_color_override") or block.get("bg_color")

            stored_bg_rgb = None
            stored_bg_lum = None
            if stored_bg_hex and isinstance(stored_bg_hex, str):
                s_hex = stored_bg_hex.strip()
                if s_hex.startswith("#") and len(s_hex) >= 7:
                    try:
                        s_r = int(s_hex[1:3], 16)
                        s_g = int(s_hex[3:5], 16)
                        s_b = int(s_hex[5:7], 16)
                        stored_bg_rgb = (s_r, s_g, s_b)
                        stored_bg_lum = 0.299 * s_r + 0.587 * s_g + 0.114 * s_b
                    except ValueError:
                        pass

            sampled_rgb = (255, 255, 255)
            sampled_lum = 255.0
            bx1, by1 = max(0, int(x)), max(0, int(y))
            bx2, by2 = min(w_img, int(x + w)), min(h_img, int(y + h))
            if bx2 > bx1 and by2 > by1:
                crop = erased_image[by1:by2, bx1:bx2]
                sampled_rgb = get_background_color_rgb(crop)
                sampled_lum = 0.299 * sampled_rgb[0] + 0.587 * sampled_rgb[1] + 0.114 * sampled_rgb[2]

            is_stored_dark = (
                (stored_bg_hex and str(stored_bg_hex).strip().lower() == "#000000") or
                (stored_bg_lum is not None and stored_bg_lum < 60.0)
            )
            is_sampled_dark = sampled_lum < 60.0

            if is_stored_dark:
                bg_rgb = stored_bg_rgb if stored_bg_rgb is not None else (0, 0, 0)
                bg_lum = stored_bg_lum if stored_bg_lum is not None else 0.0
            elif is_sampled_dark:
                bg_rgb = sampled_rgb
                bg_lum = sampled_lum
            elif stored_bg_lum is not None and str(stored_bg_hex).strip().upper() != "#FFFFFF":
                bg_rgb = stored_bg_rgb
                bg_lum = stored_bg_lum
            else:
                bg_rgb = sampled_rgb
                bg_lum = sampled_lum

            is_dark_bg_block = is_stored_dark or (bg_lum < 60.0)

            text_color_hex = self._resolve_color_hex(
                block.text_color or "#000000",
                bg_lum,
                is_dark_bg_block,
                cfg,
                text_color_override=block.text_color_override
            )

            r = int(text_color_hex[1:3], 16) if len(text_color_hex) >= 7 else 0
            g = int(text_color_hex[3:5], 16) if len(text_color_hex) >= 7 else 0
            b = int(text_color_hex[5:7], 16) if len(text_color_hex) >= 7 else 0
            fill_alpha = 175 if (block.type == "onomatopoeia" and cfg.onomatopoeia_mode == OnomatopoeiaMode.TRANSPARENT.value) else 255
            fill_rgba = (r, g, b, fill_alpha)

            # Stroke resolution helper
            stroke_mode = block.stroke_mode_override or cfg.stroke_mode
            has_block_sw = block.stroke_width_override is not None
            auto_sw = False if has_block_sw else getattr(cfg, "auto_stroke_width", True)
            sw_param = block.stroke_width_override if has_block_sw else cfg.stroke_width

            def _make_stroke(rgb_tuple: Tuple[int, int, int]) -> Optional[StrokeStyle]:
                if stroke_mode == StrokeMode.AUTO.value:
                    return StrokeRenderer.get_auto_contrast_stroke(
                        rgb_tuple,
                        font_size,
                        bg_rgb=bg_rgb,
                        base_stroke_w=sw_param,
                        auto_stroke_width=auto_sw
                    )
                elif stroke_mode == StrokeMode.MANUAL.value:
                    stroke_hex = block.stroke_color_override or cfg.stroke_color
                    sr = int(stroke_hex[1:3], 16) if len(stroke_hex) >= 7 else 255
                    sg = int(stroke_hex[3:5], 16) if len(stroke_hex) >= 7 else 255
                    sb = int(stroke_hex[5:7], 16) if len(stroke_hex) >= 7 else 255
                    eff_sw = min(float(sw_param), max(1.6, round(font_size * 0.145, 2))) if auto_sw else float(sw_param)
                    return StrokeStyle(color_rgba=(sr, sg, sb, 217), width=eff_sw)
                return None

            stroke = _make_stroke((r, g, b))

            # Drop shadow
            shadow = StrokeRenderer.get_drop_shadow_params(font_size, enabled=cfg.text_shadow)

            if is_rotated:
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
                px, py = self.auto_fit_engine.calculate_padding(draw_box_w, draw_box_h, bubble_shape=eff_bubble_shape)
                avail_w = max(4.0, draw_box_w - 2 * px)
                avail_h = max(4.0, draw_box_h - 2 * py)

                # Resolve per-paragraph / per-line colors when in ORIGINAL color mode without block override
                raw_line_colors = getattr(block, "line_colors", None) or []
                turns = collapse_color_turns(raw_line_colors) if raw_line_colors else []
                use_multiline_colors = (
                    not block.text_color_override
                    and cfg.text_color_mode != TextColorMode.CUSTOM.value
                    and (len(turns) > 1 or bool(speaker_color_map))
                )

                paragraphs = [p for p in render_text.split("\n") if p]
                if not paragraphs:
                    paragraphs = [render_text]
                dialogue_indices = [pi for pi, p in enumerate(paragraphs) if not _is_stage_direction_para(p)]
                diag_idx_to_turn: Dict[int, str] = {}
                if len(turns) == len(paragraphs):
                    for pi in range(len(paragraphs)):
                        diag_idx_to_turn[pi] = turns[pi]
                elif len(turns) == len(dialogue_indices) and dialogue_indices:
                    for ti, pi in enumerate(dialogue_indices):
                        diag_idx_to_turn[pi] = turns[ti]

                lines: List[str] = []
                line_hex_list: List[str] = []

                for p_idx, para in enumerate(paragraphs):
                    wrapped_p = self.line_breaker.wrap_text(para, avail_w, font_size, measurer)
                    para_hex = text_color_hex
                    if use_multiline_colors:
                        if _is_stage_direction_para(para):
                            para_hex = "#FFFFFF" if bg_lum < 170.0 else "#000000"
                        else:
                            spk = _extract_speaker_tag(para)
                            cand_p_hex = None
                            if spk and spk in speaker_color_map:
                                cand_p_hex = speaker_color_map[spk]
                            elif p_idx in diag_idx_to_turn:
                                cand_p_hex = diag_idx_to_turn[p_idx]
                            elif raw_line_colors:
                                r_idx = min(
                                    len(raw_line_colors) - 1,
                                    int(round(p_idx * (len(raw_line_colors) - 1) / max(1, len(paragraphs) - 1)))
                                )
                                cand_p_hex = raw_line_colors[r_idx]
                            if cand_p_hex:
                                para_hex = self._resolve_color_hex(
                                    cand_p_hex, bg_lum, is_dark_bg_block, cfg, text_color_override=None
                                )

                    for wl in wrapped_p:
                        lines.append(wl)
                        line_hex_list.append(para_hex)

                # If there was only 1 paragraph wrapped into multiple lines and raw_line_colors has multiple turns
                if use_multiline_colors and len(paragraphs) == 1 and len(lines) > 1 and len(turns) > 1:
                    for li in range(len(lines)):
                        r_idx = min(
                            len(raw_line_colors) - 1,
                            int(round(li * (len(raw_line_colors) - 1) / max(1, len(lines) - 1)))
                        )
                        line_hex_list[li] = self._resolve_color_hex(
                            raw_line_colors[r_idx], bg_lum, is_dark_bg_block, cfg, text_color_override=None
                        )

                eff_line_spacing = compute_smart_line_spacing(
                    len(lines),
                    font_size,
                    avail_h,
                    base_spacing=cfg.line_spacing,
                    auto_enabled=auto_ls
                )
                line_h = font_size * eff_line_spacing
                tot_h = len(lines) * line_h
                start_y = rel_y + (draw_box_h - tot_h) / 2.0

                # Multi-speaker dialogue blocks (e.g. (Suk)... / (Jun)...) or [Yuna] narration blocks are left-aligned
                is_left_aligned_dialogue = (speaker_turn_count >= 2) or bool(SPEAKER_LINE_RE.match(render_text))

                for i, line in enumerate(lines):
                    line_w = measurer.measure_width(line, font_size)
                    if is_left_aligned_dialogue:
                        line_x = rel_x + px
                    else:
                        line_x = rel_x + (draw_box_w - line_w) / 2.0
                    line_y = start_y + i * line_h

                    l_hex = line_hex_list[i] if i < len(line_hex_list) else text_color_hex
                    lr = int(l_hex[1:3], 16) if len(l_hex) >= 7 else r
                    lg = int(l_hex[3:5], 16) if len(l_hex) >= 7 else g
                    lb = int(l_hex[5:7], 16) if len(l_hex) >= 7 else b
                    l_fill_rgba = (lr, lg, lb, fill_alpha)
                    l_stroke = _make_stroke((lr, lg, lb))

                    StrokeRenderer.render_text_pil(
                        target_image=draw_target,
                        text=line,
                        x=line_x,
                        y=line_y,
                        font=font,
                        fill_rgba=l_fill_rgba,
                        stroke=l_stroke,
                        shadow=shadow,
                        is_bold=is_bold
                    )

            if is_rotated:
                rotated_layer = draw_target.rotate(
                    -effective_angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=True
                )
                paste_x = int(round(cx - rotated_layer.width / 2.0))
                paste_y = int(round(cy - rotated_layer.height / 2.0))
                pil_img.paste(rotated_layer, (paste_x, paste_y), rotated_layer)

        res_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGBA2BGR)

        if progress_callback:
            progress_callback(100, "文字渲染完成")

        return res_bgr
