"""
app/core/inpaint - Image inpainting, color analysis, and background reconstruction.
Lazy-loads heavy ML backends (e.g., LaMa / PyTorch) on first attribute access so
lightweight callers such as UI startup and color analysis stay fast.
"""
import importlib
from typing import Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "BaseInpainter": ("app.core.inpaint.base", "BaseInpainter"),
    "blend_inpainted_image": ("app.core.inpaint.base", "blend_inpainted_image"),
    "get_background_color_rgb": ("app.core.inpaint.color_analyzer", "get_background_color_rgb"),
    "get_background_color_hex": ("app.core.inpaint.color_analyzer", "get_background_color_hex"),
    "is_background_uniform": ("app.core.inpaint.color_analyzer", "is_background_uniform"),
    "get_text_mask": ("app.core.inpaint.color_analyzer", "get_text_mask"),
    "analyze_text_color": ("app.core.inpaint.color_analyzer", "analyze_text_color"),
    "extract_line_text_colors": ("app.core.inpaint.color_analyzer", "extract_line_text_colors"),
    "collapse_color_turns": ("app.core.inpaint.color_analyzer", "collapse_color_turns"),
    "refresh_blocks_text_colors": ("app.core.inpaint.color_analyzer", "refresh_blocks_text_colors"),
    "dilate_mask": ("app.core.inpaint.color_analyzer", "dilate_mask"),
    "OpenCVInpainter": ("app.core.inpaint.opencv_engine", "OpenCVInpainter"),
    "LaMaInpainter": ("app.core.inpaint.lama_engine", "LaMaInpainter"),
    "get_block_pixel_mask": ("app.core.inpaint.restore_helper", "get_block_pixel_mask"),
    "restore_block_pixels": ("app.core.inpaint.restore_helper", "restore_block_pixels"),
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
