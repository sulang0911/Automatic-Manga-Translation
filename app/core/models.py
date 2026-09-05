"""
app/core/models.py
Domain models for Manga/Webtoon Translation System.
"""
from __future__ import annotations
import uuid
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class BlockType(str, Enum):
    BUBBLE = "bubble"
    ONOMATOPOEIA = "onomatopoeia"
    OTHER = "other"


class TextDirection(str, Enum):
    AUTO = "auto"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class ReadingOrderMode(str, Enum):
    MANGA_RTL = "manga_rtl"
    WEBTOON_TTB = "webtoon_ttb"
    WESTERN_LTR = "western_ltr"


class TextColorMode(str, Enum):
    ORIGINAL = "original"
    CUSTOM = "custom"


class BgColorMode(str, Enum):
    ORIGINAL = "original"
    CUSTOM = "custom"
    NONE = "none"


class StrokeMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    OFF = "off"


class OnomatopoeiaMode(str, Enum):
    IGNORE = "ignore"
    TRANSPARENT = "transparent"
    NORMAL = "normal"


class PageStatus(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    OCR = "ocr"
    INPAINTING = "inpainting"
    TRANSLATING = "translating"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranslationBlock:
    """
    Represents a recognized and localized speech bubble or text region.
    All coordinate values (ymin, xmin, ymax, xmax) are stored as normalized percentages in [0.0, 100.0].
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    original_text: str = ""
    translated_text: str = ""
    ymin: float = 0.0
    xmin: float = 0.0
    ymax: float = 0.0
    xmax: float = 0.0
    polygon: Optional[List[List[int]]] = None  # [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    text_color: str = "#000000"
    bg_color: str = "#FFFFFF"
    font_size_pct: float = 2.0
    type: str = BlockType.BUBBLE.value
    direction: str = TextDirection.AUTO.value
    confidence: float = 1.0
    reading_order: int = 0
    line_count: int = 1

    # Per-block typographic overrides
    font_family_override: Optional[str] = None
    font_size_override: Optional[float] = None
    text_color_override: Optional[str] = None
    bg_color_override: Optional[str] = None
    stroke_mode_override: Optional[str] = None
    stroke_color_override: Optional[str] = None
    stroke_width_override: Optional[float] = None
    font_bold_override: Optional[bool] = None
    angle: float = 0.0  # Rotation angle in degrees (-180.0 to 180.0)
    angle_override: Optional[float] = None

    def __post_init__(self):
        # Validate and clamp coordinates to [0.0, 100.0]
        self.xmin = max(0.0, min(float(self.xmin), 100.0))
        self.ymin = max(0.0, min(float(self.ymin), 100.0))
        self.xmax = max(self.xmin, min(float(self.xmax), 100.0))
        self.ymax = max(self.ymin, min(float(self.ymax), 100.0))
        if isinstance(self.type, Enum):
            self.type = self.type.value
        if isinstance(self.direction, Enum):
            self.direction = self.direction.value
        if self.angle is None:
            self.angle = 0.0
        else:
            try:
                self.angle = float(self.angle)
            except (ValueError, TypeError):
                self.angle = 0.0
        if self.angle_override is not None:
            try:
                self.angle_override = float(self.angle_override)
            except (ValueError, TypeError):
                self.angle_override = None
        if self.polygon is not None:
            try:
                self.polygon = [[int(round(p[0])), int(round(p[1]))] for p in self.polygon]
            except Exception:
                pass

    def get_effective_angle(self) -> float:
        """Returns user override angle if present, otherwise auto-detected angle."""
        return float(self.angle_override if self.angle_override is not None else (self.angle or 0.0))

    def to_pixel_rect(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Returns integer (x, y, width, height) in pixel coordinates."""
        if img_width <= 0 or img_height <= 0:
            return (0, 0, 1, 1)
        x = int((self.xmin / 100.0) * img_width)
        y = int((self.ymin / 100.0) * img_height)
        w = max(1, int(((self.xmax - self.xmin) / 100.0) * img_width))
        h = max(1, int(((self.ymax - self.ymin) / 100.0) * img_height))
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        return (x, y, max(1, w), max(1, h))

    def to_pixel_box(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Returns integer (xmin, ymin, xmax, ymax) in pixel coordinates."""
        x, y, w, h = self.to_pixel_rect(img_width, img_height)
        return (x, y, x + w, y + h)

    def to_pixel_polygon(self, img_w: int, img_h: int) -> Optional[List[List[int]]]:
        """
        Returns polygon vertices in pixel coordinates.
        If self.polygon is already defined and valid, returns it.
        Otherwise synthesizes an oriented or axis-aligned 4-point rectangle from normalized coordinates.
        Returns None if image dimensions are invalid.
        """
        if self.polygon and len(self.polygon) >= 3:
            return [[int(round(p[0])), int(round(p[1]))] for p in self.polygon]

        if img_w <= 0 or img_h <= 0:
            return None

        xmin, ymin, xmax, ymax = self.to_pixel_box(img_w, img_h)
        eff_angle = self.get_effective_angle()
        if abs(eff_angle) < 2.5:
            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

        # Synthesize rotated rectangle around center with canonical [TL, TR, BR, BL] vertex ordering
        try:
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            w, h = float(max(1, xmax - xmin)), float(max(1, ymax - ymin))
            rad = math.radians(eff_angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            # Unit vector along text baseline (left to right)
            u_x, u_y = cos_a, sin_a
            # Unit vector along line height (top to bottom, orthogonal to baseline)
            v_x, v_y = -sin_a, cos_a

            hw, hh = w / 2.0, h / 2.0
            tl = [cx - hw * u_x - hh * v_x, cy - hw * u_y - hh * v_y]
            tr = [cx + hw * u_x - hh * v_x, cy + hw * u_y - hh * v_y]
            br = [cx + hw * u_x + hh * v_x, cy + hw * u_y + hh * v_y]
            bl = [cx - hw * u_x + hh * v_x, cy - hw * u_y + hh * v_y]

            return [[int(round(p[0])), int(round(p[1]))] for p in [tl, tr, br, bl]]
        except Exception:
            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    @classmethod
    def from_pixel_box(
        cls,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        img_width: int,
        img_height: int,
        polygon: Optional[List[List[int]]] = None,
        angle: float = 0.0,
        **kwargs
    ) -> TranslationBlock:
        """Factory creating a TranslationBlock from pixel coordinates."""
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive.")
        norm_xmin = round((float(xmin) / img_width) * 100.0, 2)
        norm_ymin = round((float(ymin) / img_height) * 100.0, 2)
        norm_xmax = round((float(xmax) / img_width) * 100.0, 2)
        norm_ymax = round((float(ymax) / img_height) * 100.0, 2)
        return cls(
            xmin=norm_xmin,
            ymin=norm_ymin,
            xmax=norm_xmax,
            ymax=norm_ymax,
            polygon=polygon,
            angle=angle,
            **kwargs
        )

    def center_normalized(self) -> Tuple[float, float]:
        """Returns (center_x, center_y) in normalized [0, 100] coordinates."""
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def aspect_ratio(self) -> float:
        """Returns width / height ratio."""
        h = max(0.01, self.ymax - self.ymin)
        w = self.xmax - self.xmin
        return w / h

    def is_vertical_candidate(self) -> bool:
        """Heuristic check: returns True if height > width * 1.15."""
        return (self.ymax - self.ymin) > (self.xmax - self.xmin) * 1.15

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TranslationBlock:
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class StyleConfig:
    """
    Global typography and rendering configuration.
    """
    font_family: str = "霞鹜文楷"
    font_size_scale: float = 1.0
    auto_fit_font_size: bool = True
    min_font_size: int = 8
    max_font_size: int = 72
    line_spacing: float = 1.15
    text_color_mode: str = TextColorMode.ORIGINAL.value
    custom_text_color: str = "#000000"
    bg_color_mode: str = BgColorMode.ORIGINAL.value
    custom_bg_color: str = "#FFFFFF"
    bg_opacity: float = 0.95
    text_shadow: bool = False
    shadow_color: str = "rgba(0,0,0,0.5)"
    shadow_blur: float = 3.0
    text_stroke: bool = True
    stroke_mode: str = StrokeMode.AUTO.value
    stroke_color: str = "#FFFFFF"
    stroke_width: float = 2.0
    font_bold: bool = True
    font_italic: bool = False
    onomatopoeia_mode: str = OnomatopoeiaMode.NORMAL.value
    reading_direction: str = ReadingOrderMode.MANGA_RTL.value
    export_compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StyleConfig:
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class MangaPage:
    """
    Represents a manga page in a batch or active workspace.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    file_path: str = ""
    file_name: str = ""
    width: int = 0
    height: int = 0
    status: str = PageStatus.IDLE.value
    progress: int = 0
    error_message: Optional[str] = None
    blocks: List[TranslationBlock] = field(default_factory=list)

    # Disk cache paths
    has_ocr_cache: bool = False
    has_erased_cache: bool = False
    has_translated_cache: bool = False
    erased_image_path: Optional[str] = None
    translated_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["blocks"] = [b.to_dict() for b in self.blocks]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MangaPage:
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields and k != "blocks"}
        blocks_data = data.get("blocks", [])
        page = cls(**filtered)
        page.blocks = [TranslationBlock.from_dict(b) for b in blocks_data]
        return page
