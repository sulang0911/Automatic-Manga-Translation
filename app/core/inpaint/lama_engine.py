"""
app/core/inpaint/lama_engine.py
Deep learning neural inpainter using SimpleLama with:
- Dimension padding compensation
- VRAM safety downsampling for ultra-high-resolution images (> 2048px)
- Automatic fallback to OpenCVInpainter on CUDA OOM or load failure
- PyTorch GPU cache cleanup
"""
import gc
import logging
from typing import List, Optional, Callable
import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from app.core.models import TranslationBlock, StyleConfig, OnomatopoeiaMode
from app.core.inpaint.base import BaseInpainter, blend_inpainted_image
from app.core.inpaint.color_analyzer import (
    get_background_color_rgb,
    is_background_uniform,
    get_text_mask,
    dilate_mask
)
from app.core.inpaint.opencv_engine import OpenCVInpainter

logger = logging.getLogger(__name__)


class LaMaInpainter(BaseInpainter):
    def __init__(self, fallback_to_opencv: bool = True):
        self.fallback_to_opencv = fallback_to_opencv
        self._lama = None
        self._lama_available = False
        self._opencv_fallback = OpenCVInpainter(method="telea")
        self._init_lama()

    def _init_lama(self):
        try:
            from simple_lama_inpainting import SimpleLama
            logger.info("Initializing Simple-LaMa neural inpainting model...")
            self._lama = SimpleLama()
            self._lama_available = True
            logger.info("Simple-LaMa neural inpainting model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Simple-LaMa: {e}. OpenCV fallback will be used.")
            self._lama_available = False

    def is_available(self) -> bool:
        return self._lama_available

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

        if not self._lama_available and self.fallback_to_opencv:
            return self._opencv_fallback.inpaint(image, blocks, style_config, progress_callback)

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
                progress_callback(int((idx / total_blocks) * 35), f"分析修复掩码 ({idx+1}/{total_blocks})...")

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
                # Instant flat-fill for solid dialogue bubbles
                dilated = dilate_mask(text_mask, dyn_bubble_dil)
                crop[dilated == 255] = bg_bgr
                erased_img[y:y+h, x:x+w] = crop
            else:
                # Accumulate for neural inpainting
                dilated = dilate_mask(text_mask, dyn_onoma_dil)
                inpaint_mask[y:y+h, x:x+w] = cv2.bitwise_or(inpaint_mask[y:y+h, x:x+w], dilated)

        if np.sum(inpaint_mask) > 0:
            if progress_callback:
                progress_callback(45, "正在执行 Simple-LaMa 深度图像修复...")

            inpainted = None
            try:
                # VRAM Safety Check: if image dimension > 2048px, downscale for LaMa inference
                scale_factor = 1.0
                if base_dim > 2048:
                    scale_factor = 2048.0 / base_dim
                    infer_w = int(w_img * scale_factor)
                    infer_h = int(h_img * scale_factor)
                    infer_img = cv2.resize(erased_img, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                    infer_mask = cv2.resize(inpaint_mask, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
                else:
                    infer_img = erased_img
                    infer_mask = inpaint_mask

                img_rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
                img_pil = PILImage.fromarray(img_rgb)
                mask_pil = PILImage.fromarray(infer_mask)

                # Execute LaMa model
                res_pil = self._lama(img_pil, mask_pil)
                res_rgb = np.array(res_pil)
                res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

                # Compensate for LaMa multiple-of-8 padding
                infer_h_actual, infer_w_actual = infer_img.shape[:2]
                res_bgr = res_bgr[:infer_h_actual, :infer_w_actual]

                # Rescale back if downsampled
                if scale_factor < 1.0:
                    inpainted = cv2.resize(res_bgr, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
                else:
                    inpainted = res_bgr

                logger.info("LaMa neural inpainting completed successfully.")
            except Exception as e:
                logger.error(f"LaMa execution encountered error: {e}. Falling back to OpenCV Telea.", exc_info=True)
                inpainted = None
            finally:
                # Explicit VRAM garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Automatic OpenCV Telea Fallback
            if inpainted is None:
                if progress_callback:
                    progress_callback(60, "LaMa 资源受限，正在回退使用 OpenCV Telea 算法...")
                inpainted = cv2.inpaint(erased_img, inpaint_mask, 3, cv2.INPAINT_TELEA)

            # Bilateral filter and feathered blend
            filtered = cv2.bilateralFilter(inpainted, 5, 50, 50)
            erased_img = blend_inpainted_image(erased_img, filtered, inpaint_mask, feather_radius)

        if progress_callback:
            progress_callback(100, "图像修复与去字完成")

        return erased_img
