"""
app/core/typography/vertical_layout.py
CJK Vertical Text Rendering with RTL Column Progression and Punctuation Offset.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict
import re


VERTICAL_GLYPH_MAP: Dict[str, str] = {
    '「': '﹁', '」': '﹂',
    '『': '﹃', '』': '﹄',
    '（': '︵', '）': '︶',
    '【': '︻', '】': '︼',
    '《': '︽', '》': '︾',
    '〔': '︹', '〕': '︺',
    'ー': '丨', '―': '︱', '—': '︱',
    '…': '︙', '～': '︴',
    '!?': '⁉', '?!': '⁈', '!!': '‼', '??': '⁇'
}

CORNER_PUNCTUATION = set("。、，．")


@dataclass(frozen=True)
class VerticalGlyph:
    char: str
    x: float
    y: float
    offset_x: float
    offset_y: float


@dataclass(frozen=True)
class VerticalColumn:
    col_index: int
    x_center: float
    glyphs: List[VerticalGlyph]
    height: float


class VerticalLayoutEngine:
    """
    RTL column progression, top-to-bottom character stack,
    with glyph rotation and punctuation corner shifts.
    """
    def __init__(
        self,
        column_spacing_ratio: float = 1.25,
        char_spacing_ratio: float = 1.10
    ):
        self.column_spacing_ratio = column_spacing_ratio
        self.char_spacing_ratio = char_spacing_ratio

    def wrap_vertical_columns(
        self,
        text: str,
        max_height: float,
        font_size: float
    ) -> List[List[str]]:
        """
        Split text into vertical columns that fit within max_height.
        Applies vertical punctuation replacements during segmentation.
        """
        processed = text
        for pair, repl in [('!?', '⁉'), ('?!', '⁈'), ('!!', '‼'), ('??', '⁇')]:
            processed = processed.replace(pair, repl)

        char_step = font_size * self.char_spacing_ratio
        max_chars_per_col = max(1, int(max_height / char_step))

        columns: List[List[str]] = []
        current_col: List[str] = []

        paragraphs = processed.split("\n")
        for p in paragraphs:
            trimmed = p.strip()
            if not trimmed:
                continue
            for char in trimmed:
                vert_char = VERTICAL_GLYPH_MAP.get(char, char)
                if len(current_col) >= max_chars_per_col and current_col:
                    columns.append(current_col)
                    current_col = [vert_char]
                else:
                    current_col.append(vert_char)
            if current_col:
                columns.append(current_col)
                current_col = []

        return columns if columns else [[VERTICAL_GLYPH_MAP.get(c, c) for c in text]]

    def compute_layout(
        self,
        text: str,
        font_size: float,
        box_x: float,
        box_y: float,
        box_w: float,
        box_h: float
    ) -> Tuple[List[VerticalColumn], float, float]:
        """
        Compute exact 2D coordinates for vertical RTL columns and stacked glyphs.
        Returns (columns, total_width, total_height).
        """
        col_width = font_size * self.column_spacing_ratio
        char_height = font_size * self.char_spacing_ratio

        raw_cols = self.wrap_vertical_columns(text, box_h, font_size)
        num_cols = len(raw_cols)
        total_width = num_cols * col_width

        # Right-to-Left (RTL) progression:
        # Col 0 sits at the far right of the centered text block
        center_x = box_x + box_w / 2.0
        start_x = center_x + (total_width / 2.0) - (col_width / 2.0)

        columns: List[VerticalColumn] = []
        max_col_h = 0.0

        for col_idx, col_chars in enumerate(raw_cols):
            # RTL X position: subtract column offset
            col_x = start_x - col_idx * col_width
            col_h = len(col_chars) * char_height
            max_col_h = max(max_col_h, col_h)

            # Center column vertically within box_h
            start_y = box_y + (box_h - col_h) / 2.0 + (char_height / 2.0)

            glyphs: List[VerticalGlyph] = []
            for j, ch in enumerate(col_chars):
                glyph_y = start_y + j * char_height
                # Punctuation shift: shift periods/commas to upper-right quadrant
                ox, oy = (0.0, 0.0)
                if ch in CORNER_PUNCTUATION:
                    ox = font_size * 0.28
                    oy = -font_size * 0.28

                glyphs.append(VerticalGlyph(char=ch, x=col_x, y=glyph_y, offset_x=ox, offset_y=oy))

            columns.append(VerticalColumn(
                col_index=col_idx,
                x_center=col_x,
                glyphs=glyphs,
                height=col_h
            ))

        return columns, total_width, max_col_h
