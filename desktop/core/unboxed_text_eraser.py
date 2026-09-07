"""
desktop/core/unboxed_text_eraser.py
Re-export for desktop core compatibility.
"""
from app.core.inpaint.unboxed_text_eraser import (
    is_unboxed_text_block,
    erase_unboxed_text_block,
)

__all__ = ["is_unboxed_text_block", "erase_unboxed_text_block"]
