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
        progress_callback: Optional[Callable[[int, str], None]] = None,
        qr_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if image is None:
            return None
        if image.size == 0 or not blocks:
            return image.copy()

        h_img, w_img = image.shape[:2]
        erased_img = image.copy()
        inpaint_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # Detect or use QR protection mask
        if qr_mask is None:
            try:
                from app.core.ocr.qr_filter import QRCodeFilter
                filt = QRCodeFilter()
                qr_regs = filt.detect_regions(image)
                qr_mask = filt.get_protection_mask((h_img, w_img), qr_regs)
            except Exception:
                qr_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        base_dim = max(h_img, w_img)
        feather_radius = max(2, int(base_dim * 0.003))

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / total_blocks) * 45), f"分析背景区域 ({idx+1}/{total_blocks})...")

            if block.type == "onomatopoeia" and style_config and style_config.onomatopoeia_mode == OnomatopoeiaMode.IGNORE.value:
                continue

            poly = block.to_pixel_polygon(w_img, h_img) if hasattr(block, "to_pixel_polygon") else None
            if poly is None:
                x, y, w, h = block.to_pixel_rect(w_img, h_img)
                if w <= 0 or h <= 0:
                    continue
                poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            poly_pts = np.array(poly, dtype=np.int32)
            px_min = max(0, int(np.min(poly_pts[:, 0])))
            py_min = max(0, int(np.min(poly_pts[:, 1])))
            px_max = min(w_img, int(np.max(poly_pts[:, 0])))
            py_max = min(h_img, int(np.max(poly_pts[:, 1])))

            if px_max <= px_min or py_max <= py_min:
                continue

            w = px_max - px_min
            h = py_max - py_min

            crop = erased_img[py_min:py_max, px_min:px_max]
            r, g, b = get_background_color_rgb(crop)
            bg_bgr = (b, g, r)

            # Local polygon mask
            local_poly = poly_pts - np.array([px_min, py_min], dtype=np.int32)
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [local_poly], 255)

            # Adaptive morphological dilation based on polygon minor axis (min_dim * 0.08, clamped [2, 8]px)
            if len(poly_pts) >= 4:
                side1 = float(np.linalg.norm(poly_pts[1] - poly_pts[0]))
                side2 = float(np.linalg.norm(poly_pts[3] - poly_pts[0]))
                min_dim = min(side1, side2)
                if min_dim < 1.0:
                    min_dim = min(w, h)
            else:
                min_dim = min(w, h)
            adaptive_dil = max(2, min(8, int(round(min_dim * 0.08))))

            text_mask = get_text_mask(crop, bg_bgr)
            text_mask = cv2.bitwise_and(text_mask, poly_mask)
            uniform = is_background_uniform(crop, std_thresh=15.0)

            qr_mask_crop = qr_mask[py_min:py_max, px_min:px_max]

            if block.type == "bubble" and uniform:
                # Instant flat-fill for solid speech bubbles
                dilated = dilate_mask(text_mask, adaptive_dil)
                # Ensure QR regions are NOT overwritten
                crop[(dilated == 255) & (qr_mask_crop == 0)] = bg_bgr
                erased_img[py_min:py_max, px_min:px_max] = crop
            else:
                # Accumulate into inpaint mask
                dilated = dilate_mask(text_mask, adaptive_dil)
                if np.sum(text_mask) == 0:
                    dilated = dilate_mask(poly_mask, adaptive_dil)
                inpaint_mask[py_min:py_max, px_min:px_max] = cv2.bitwise_or(
                    inpaint_mask[py_min:py_max, px_min:px_max], dilated
                )

        # Explicitly zero out QR code regions from inpaint_mask
        inpaint_mask[qr_mask > 0] = 0
        # Re-ensure any flat-fill never touched QR regions
        erased_img[qr_mask > 0] = image[qr_mask > 0]

        if np.sum(inpaint_mask) > 0:
            if progress_callback:
                progress_callback(55, "正在执行 OpenCV 纹理补全与双边滤波...")

            inpaint_flag = cv2.INPAINT_NS if self.method == "ns" else cv2.INPAINT_TELEA
            inpainted = cv2.inpaint(erased_img, inpaint_mask, 3, inpaint_flag)
            filtered = cv2.bilateralFilter(inpainted, 5, 50, 50)
            erased_img = blend_inpainted_image(erased_img, filtered, inpaint_mask, feather_radius)
            # Guarantee zero-pixel modification on QR code areas
            erased_img[qr_mask > 0] = image[qr_mask > 0]

        if progress_callback:
            progress_callback(100, "背景擦除完成")

        return erased_img
