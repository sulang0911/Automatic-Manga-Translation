"""
app/core/ocr - Text detection, recognition, and spatial reading order sorting.
Lazy-loads heavy OCR engines (EasyOCR / PaddleOCR / CTD / MangaOCR) on first use.
"""
import importlib
from typing import Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BaseOCREngine": ("app.core.ocr.base", "BaseOCREngine"),
    "is_solid_color_page": ("app.core.ocr.base", "is_solid_color_page"),
    "merge_adjacent_boxes": ("app.core.ocr.base", "merge_adjacent_boxes"),
    "sort_reading_order": ("app.core.ocr.reading_order", "sort_reading_order"),
    "EasyOCREngine": ("app.core.ocr.easyocr_engine", "EasyOCREngine"),
    "PaddleOCREngine": ("app.core.ocr.paddle_engine", "PaddleOCREngine"),
    "MangaOCRRecognizer": ("app.core.ocr.manga_ocr_wrapper", "MangaOCRRecognizer"),
    "get_manga_ocr": ("app.core.ocr.manga_ocr_wrapper", "get_manga_ocr"),
    "ComicTextDetectorEngine": ("app.core.ocr.ctd_engine", "ComicTextDetectorEngine"),
    "TransformersOCRRecognizer": ("app.core.ocr.transformers_rec_wrapper", "TransformersOCRRecognizer"),
    "get_transformers_ocr": ("app.core.ocr.transformers_rec_wrapper", "get_transformers_ocr"),
    "recognize_crop_with_vlm": ("app.core.ocr.vlm_ocr", "recognize_crop_with_vlm"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(spec[0])
    val = getattr(mod, spec[1])
    globals()[name] = val
    return val


def __dir__():
    return sorted(set(globals()) | set(__all__))
