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
    Bounded to the active mask ROI to avoid full-page float32 RAM spikes.
    """
    if original_img is None:
        return None
    if inpainted_img is None or mask is None or not np.any(mask):
        return original_img.copy()

    if feather_radius <= 0:
        result = original_img.copy()
        result[mask > 0] = inpainted_img[mask > 0]
        return result

    h_img, w_img = original_img.shape[:2]
    nz_y, nz_x = np.nonzero(mask)
    if len(nz_y) == 0:
        return original_img.copy()

    pad = feather_radius * 3 + 2
    y1 = max(0, int(nz_y.min()) - pad)
    y2 = min(h_img, int(nz_y.max()) + pad + 1)
    x1 = max(0, int(nz_x.min()) - pad)
    x2 = min(w_img, int(nz_x.max()) + pad + 1)

    orig_roi = original_img[y1:y2, x1:x2]
    inp_roi = inpainted_img[y1:y2, x1:x2]
    mask_roi = mask[y1:y2, x1:x2]

    ksize = feather_radius * 2 + 1
    alpha = cv2.GaussianBlur(mask_roi.astype(np.float32) * (1.0 / 255.0), (ksize, ksize), 0)
    alpha = np.expand_dims(alpha, axis=2)

    blended_roi = inp_roi.astype(np.float32) * alpha + orig_roi.astype(np.float32) * (1.0 - alpha)
    blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)
    del alpha

    result = original_img.copy()
    result[y1:y2, x1:x2] = blended_roi
    return result
