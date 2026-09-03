"""
app/core/inpaint/base.py
Abstract base class and Gaussian alpha feather blending functions for inpainting engines.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Callable
import numpy as np
import cv2
from app.core.models import TranslationBlock, StyleConfig


class BaseInpainter(ABC):
    @abstractmethod
    def inpaint(
        self,
        image: np.ndarray,
        blocks: List[TranslationBlock],
        style_config: Optional[StyleConfig] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Optional[np.ndarray]:
        """
        Inpaints and erases recognized text blocks from the input BGR image.
        Returns the erased background as an OpenCV BGR image (uint8), or None if image is None.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the inpainter dependencies and weights are loaded."""
        pass


def blend_inpainted_image(
    original_img: Optional[np.ndarray],
    inpainted_img: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    feather_radius: int = 4
) -> Optional[np.ndarray]:
    """
    Blends inpainted texture into the original artwork using Gaussian blurred alpha feathering.
    Eliminates visible seams along the boundary of the inpainting mask.
    """
    if original_img is None:
        return None
    if inpainted_img is None or mask is None or np.sum(mask) == 0:
        return original_img.copy()

    if feather_radius <= 0:
        result = original_img.copy()
        result[mask > 0] = inpainted_img[mask > 0]
        return result

    ksize = feather_radius * 2 + 1
    # Normalized Gaussian alpha mask
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    alpha = np.expand_dims(alpha, axis=2)

    blended = inpainted_img.astype(np.float32) * alpha + original_img.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)
