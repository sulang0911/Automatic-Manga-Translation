"""
app.core - Core Engine Subsystems for Automatic Manga Translation.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except ImportError:
    pass

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
from .hardware import get_gpu_info, is_legacy_pascal_or_maxwell_gpu, is_vram_constrained

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
    "get_gpu_info",
    "is_legacy_pascal_or_maxwell_gpu",
    "is_vram_constrained",
]
