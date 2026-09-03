"""
app/core/ocr - Text detection, recognition, and spatial reading order sorting.
"""
from .base import BaseOCREngine, is_solid_color_page, merge_adjacent_boxes
from .reading_order import sort_reading_order
from .easyocr_engine import EasyOCREngine
from .paddle_engine import PaddleOCREngine

__all__ = [
    "BaseOCREngine",
    "is_solid_color_page",
    "merge_adjacent_boxes",
    "sort_reading_order",
    "EasyOCREngine",
    "PaddleOCREngine",
]
