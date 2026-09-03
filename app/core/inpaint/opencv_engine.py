"""
app/core/inpaint/opencv_engine.py
Fast OpenCV-based inpainter using median flat-fill for uniform bubbles,
OpenCV Telea diffusion for textured regions, bilateral filtering, and dynamic feathering.
"""
import logging
from typing import List, Optional, Callable
import cv2
import numpy as np
from app.core.models import TranslationBlock, StyleConfig, OnomatopoeiaMode
from app.core.inpaint.base import BaseInpainter, blend_inpainted_image
from app.core.inpaint.color_analyzer import (
    get_background_color_rgb,
    is_background_uniform,
    get_text_mask,
    dilate_mask
)

logger = logging.getLogger(__name__)


class OpenCVInpainter(BaseInpainter):
    def __init__(self, method: str = "telea"):
        self.method = method  # "telea" or "ns"

    def is_available(self) -> bool:
        return True

    def inpaint(
        self,
        image: np.ndarray,
        blocks: List[TranslationBlock],
        style_config: Optional[StyleConfig] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Optional[np.ndarray]:
        if image is None:
            return None
        if image.size == 0 or not blocks:
            return image.copy()

        h_img, w_img = image.shape[:2]
        erased_img = image.copy()
        inpaint_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        base_dim = max(h_img, w_img)
        dyn_bubble_dil = max(1, int(base_dim * 0.002))
        dyn_onoma_dil = max(2, int(base_dim * 0.004))
        feather_radius = max(2, int(base_dim * 0.003))

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / total_blocks) * 45), f"分析背景区域 ({idx+1}/{total_blocks})...")

            if block.type == "onomatopoeia" and style_config and style_config.onomatopoeia_mode == OnomatopoeiaMode.IGNORE.value:
                continue

            x, y, w, h = block.to_pixel_rect(w_img, h_img)
            if w <= 0 or h <= 0:
                continue

            crop = erased_img[y:y+h, x:x+w]
            r, g, b = get_background_color_rgb(crop)
            bg_bgr = (b, g, r)

            text_mask = get_text_mask(crop, bg_bgr)
            uniform = is_background_uniform(crop, std_thresh=15.0)

            if block.type == "bubble" and uniform:
                # Instant flat-fill for solid speech bubbles
                dilated = dilate_mask(text_mask, dyn_bubble_dil)
                crop[dilated == 255] = bg_bgr
                erased_img[y:y+h, x:x+w] = crop
            else:
                # Accumulate into inpaint mask for Telea diffusion
                dilated = dilate_mask(text_mask, dyn_onoma_dil)
                inpaint_mask[y:y+h, x:x+w] = cv2.bitwise_or(inpaint_mask[y:y+h, x:x+w], dilated)

        if np.sum(inpaint_mask) > 0:
            if progress_callback:
                progress_callback(55, "正在执行 OpenCV 纹理补全与双边滤波...")

            inpaint_flag = cv2.INPAINT_NS if self.method == "ns" else cv2.INPAINT_TELEA
            inpainted = cv2.inpaint(erased_img, inpaint_mask, 3, inpaint_flag)
            filtered = cv2.bilateralFilter(inpainted, 5, 50, 50)
            erased_img = blend_inpainted_image(erased_img, filtered, inpaint_mask, feather_radius)

        if progress_callback:
            progress_callback(100, "背景擦除完成")

        return erased_img
