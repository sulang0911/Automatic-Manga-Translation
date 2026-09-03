"""
app/core/inpaint - Image inpainting, color analysis, and background reconstruction.
"""
from .base import BaseInpainter, blend_inpainted_image
from .color_analyzer import (
    get_background_color_rgb,
    get_background_color_hex,
    is_background_uniform,
    get_text_mask,
    analyze_text_color,
    dilate_mask,
)
from .opencv_engine import OpenCVInpainter
from .lama_engine import LaMaInpainter

__all__ = [
    "BaseInpainter",
    "blend_inpainted_image",
    "get_background_color_rgb",
    "get_background_color_hex",
    "is_background_uniform",
    "get_text_mask",
    "analyze_text_color",
    "dilate_mask",
    "OpenCVInpainter",
    "LaMaInpainter",
]
