"""
app.core - Core Engine Subsystems for Automatic Manga Translation.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .models import (
    BlockType,
    TextDirection,
    ReadingOrderMode,
    TextColorMode,
    BgColorMode,
    StrokeMode,
    OnomatopoeiaMode,
    PageStatus,
    TranslationBlock,
    StyleConfig,
    MangaPage,
)
from .config import AppConfig

_LAZY_HARDWARE = ("get_gpu_info", "is_legacy_pascal_or_maxwell_gpu", "is_vram_constrained")

__all__ = [
    "BlockType",
    "TextDirection",
    "ReadingOrderMode",
    "TextColorMode",
    "BgColorMode",
    "StrokeMode",
    "OnomatopoeiaMode",
    "PageStatus",
    "TranslationBlock",
    "StyleConfig",
    "MangaPage",
    "AppConfig",
    *_LAZY_HARDWARE,
]


def __getattr__(name: str):
    if name in _LAZY_HARDWARE:
        from . import hardware
        val = getattr(hardware, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
