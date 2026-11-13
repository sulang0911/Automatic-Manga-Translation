"""
app/core/inpaint/lama_engine.py
Deep learning neural inpainter using SimpleLama with:
- Dimension padding compensation
- ROI-bounded inference & VRAM safety downsampling for high-resolution images
- Automatic fallback to OpenCVInpainter on CUDA OOM or load failure
- PyTorch GPU cache cleanup & torch.no_grad() guard
"""
import gc
import logging
from contextlib import nullcontext
from typing import List, Optional, Callable
import cv2
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
from PIL import Image as PILImage

from app.core.models import TranslationBlock, StyleConfig, OnomatopoeiaMode
from app.core.inpaint.base import BaseInpainter
from app.core.inpaint.color_analyzer import (
    get_background_color_rgb,
    is_background_uniform,
    get_text_mask,
    dilate_mask
)
from app.core.inpaint.opencv_engine import OpenCVInpainter

logger = logging.getLogger(__name__)


def blend_inpainted_image(
    original_img: Optional[np.ndarray],
    inpainted_img: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    feather_radius: int = 4
) -> Optional[np.ndarray]:
    """
    Blends inpainted texture into the original artwork using Gaussian blurred alpha feathering.
    Ensures core mask retains full opacity and prevents bleeding original white text edges
    into dark/black background boxes. Bounded to the active mask ROI for minimal RAM usage.
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
    mask_f = mask_roi.astype(np.float32) * (1.0 / 255.0)
    alpha = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
    # Ensure core mask regions retain full inpainting opacity so thin text strokes don't bleed original text
    np.maximum(alpha, mask_f, out=alpha)
    del mask_f

    alpha_3d = np.expand_dims(alpha, axis=2)
    blended_roi = inp_roi.astype(np.float32) * alpha_3d + orig_roi.astype(np.float32) * (1.0 - alpha_3d)
    blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8)
    del alpha_3d

    # Inverted dark/mid-tone background protection:
    orig_gray = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2GRAY)
    inpaint_gray = cv2.cvtColor(inp_roi, cv2.COLOR_BGR2GRAY)
    dark_inpaint = inpaint_gray < 140
    bright_orig = orig_gray > (inpaint_gray + 15)
    bleed_mask = (alpha > 0.05) & dark_inpaint & bright_orig
    blended_roi[bleed_mask] = inp_roi[bleed_mask]
    del alpha, orig_gray, inpaint_gray, dark_inpaint, bright_orig, bleed_mask

    result = original_img.copy()
    result[y1:y2, x1:x2] = blended_roi
    return result


class LaMaInpainter(BaseInpainter):
    def __init__(self, fallback_to_opencv: bool = True):
        self.fallback_to_opencv = fallback_to_opencv
        self._lama = None
        self._lama_available = False
        self._opencv_fallback = OpenCVInpainter(method="telea")
        self._init_lama()

    def _init_lama(self):
        try:
            import desktop.core.inpaint_engine as _dp_inp
            if getattr(_dp_inp, "_GLOBAL_SIMPLE_LAMA", None) is not None:
                self._lama = _dp_inp._GLOBAL_SIMPLE_LAMA
                self._lama_available = True
                return
            from simple_lama_inpainting import SimpleLama
            logger.info("Initializing Simple-LaMa neural inpainting model...")
            self._lama = SimpleLama()
            _dp_inp._GLOBAL_SIMPLE_LAMA = self._lama
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
        progress_callback: Optional[Callable[[int, str], None]] = None,
        qr_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        if image is None:
            return None
        if image.size == 0 or not blocks:
            return image.copy()

        # Detect or use QR protection mask
        h_img, w_img = image.shape[:2]
        if qr_mask is None:
            try:
                from app.core.ocr.qr_filter import QRCodeFilter
                filt = QRCodeFilter()
                qr_regs = filt.detect_regions(image)
                qr_mask = filt.get_protection_mask((h_img, w_img), qr_regs)
            except Exception:
                qr_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        if not self._lama_available and self.fallback_to_opencv:
            return self._opencv_fallback.inpaint(image, blocks, style_config, progress_callback, qr_mask=qr_mask)

        erased_img = image.copy()
        inpaint_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        base_dim = max(h_img, w_img)
        feather_radius = max(2, int(base_dim * 0.003))

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / total_blocks) * 35), f"分析修复掩码 ({idx+1}/{total_blocks})...")

            b_type = getattr(block, "type", "") if not isinstance(block, dict) else block.get("type", "")
            b_force = getattr(block, "force_erase", False) if not isinstance(block, dict) else block.get("force_erase", False)
            has_trans = bool(getattr(block, "translated_text", "") if not isinstance(block, dict) else block.get("translated_text", ""))
            if isinstance(style_config, dict):
                onoma_mode = style_config.get("onomatopoeia_mode", "normal")
            elif style_config:
                onoma_mode = getattr(style_config, "onomatopoeia_mode", "normal")
            else:
                onoma_mode = "normal"
            if b_type == "onomatopoeia" and onoma_mode == "ignore" and not b_force and not has_trans:
                continue

            from app.core.inpaint.unboxed_text_eraser import is_unboxed_text_block, erase_unboxed_text_block
            if is_unboxed_text_block(block, erased_img):
                erased_img, unboxed_mask = erase_unboxed_text_block(
                    erased_img, block, qr_mask=qr_mask, min_dilation=3
                )
                if unboxed_mask is not None and np.any(unboxed_mask):
                    inpaint_mask = cv2.bitwise_or(inpaint_mask, unboxed_mask)
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

            local_poly = poly_pts - np.array([px_min, py_min], dtype=np.int32)
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [local_poly], 255)

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
            uniform = is_background_uniform(crop, std_thresh=16.0)

            qr_mask_crop = qr_mask[py_min:py_max, px_min:px_max]

            block_bg = getattr(block, "bg_color_override", None) or getattr(block, "bg_color", None)
            if isinstance(block, dict):
                block_bg = block.get("bg_color_override") or block.get("bg_color")

            bg_hex_lum = None
            hex_bgr = None
            if block_bg and isinstance(block_bg, str):
                s_clean = block_bg.strip()
                if s_clean.startswith("#") and len(s_clean) >= 7:
                    try:
                        hr = int(s_clean[1:3], 16)
                        hg = int(s_clean[3:5], 16)
                        hb = int(s_clean[5:7], 16)
                        hex_bgr = (hb, hg, hr)
                        bg_hex_lum = 0.299 * hr + 0.587 * hg + 0.114 * hb
                    except ValueError:
                        pass

            crop_bg_lum = 0.299 * r + 0.587 * g + 0.114 * b

            is_dark_bg = False
            if block_bg and isinstance(block_bg, str) and block_bg.strip().lower() == "#000000":
                is_dark_bg = True
            elif bg_hex_lum is not None and bg_hex_lum < 50.0:
                is_dark_bg = True
            elif crop_bg_lum < 50.0:
                is_dark_bg = True

            if is_dark_bg:
                if block_bg and isinstance(block_bg, str) and block_bg.strip().lower() == "#000000":
                    target_bg_bgr = (0, 0, 0)
                elif bg_hex_lum is not None and bg_hex_lum < 50.0 and (crop_bg_lum >= 50.0 or crop_bg_lum > bg_hex_lum + 30):
                    target_bg_bgr = hex_bgr
                elif crop_bg_lum < 50.0:
                    target_bg_bgr = bg_bgr
                else:
                    target_bg_bgr = hex_bgr if hex_bgr is not None else bg_bgr

                text_mask = get_text_mask(crop, target_bg_bgr)
                text_mask = cv2.bitwise_and(text_mask, poly_mask)

            block_type = getattr(block, "type", "bubble")
            if isinstance(block, dict):
                block_type = block.get("type", "bubble")

            explicit_black_box = bool(block_bg and isinstance(block_bg, str) and block_bg.strip().lower() == "#000000")
            if explicit_black_box or (is_dark_bg and uniform) or (block_type == "bubble" and uniform):
                fill_color = target_bg_bgr if is_dark_bg else bg_bgr
                dilated = dilate_mask(text_mask, adaptive_dil)
                if not np.any(text_mask):
                    dilated = dilate_mask(poly_mask, adaptive_dil)
                crop[(dilated == 255) & (qr_mask_crop == 0)] = fill_color
                erased_img[py_min:py_max, px_min:px_max] = crop
            else:
                from app.core.inpaint.color_analyzer import get_stroked_text_mask
                pad_s = 10
                epx_min = max(0, px_min - pad_s)
                epy_min = max(0, py_min - pad_s)
                epx_max = min(w_img, px_max + pad_s)
                epy_max = min(h_img, py_max + pad_s)
                exp_crop = erased_img[epy_min:epy_max, epx_min:epx_max]
                stroked_mask = get_stroked_text_mask(exp_crop, img_max_dim=base_dim)
                if stroked_mask is not None:
                    qr_exp = qr_mask[epy_min:epy_max, epx_min:epx_max]
                    stroked_mask[qr_exp > 0] = 0
                    inpaint_mask[epy_min:epy_max, epx_min:epx_max] = cv2.bitwise_or(
                        inpaint_mask[epy_min:epy_max, epx_min:epx_max], stroked_mask
                    )
                else:
                    dilated = dilate_mask(text_mask, adaptive_dil)
                    if not np.any(text_mask):
                        dilated = dilate_mask(poly_mask, adaptive_dil)
                    inpaint_mask[py_min:py_max, px_min:px_max] = cv2.bitwise_or(
                        inpaint_mask[py_min:py_max, px_min:px_max], dilated
                    )

        # Explicitly zero out QR code regions from inpaint_mask
        inpaint_mask[qr_mask > 0] = 0
        erased_img[qr_mask > 0] = image[qr_mask > 0]

        if np.any(inpaint_mask):
            if progress_callback:
                progress_callback(45, "正在执行 Simple-LaMa 深度图像修复...")

            nz_y, nz_x = np.nonzero(inpaint_mask)
            pad_roi = max(48, feather_radius * 4 + 16)
            ry1 = max(0, int(nz_y.min()) - pad_roi)
            ry2 = min(h_img, int(nz_y.max()) + pad_roi + 1)
            rx1 = max(0, int(nz_x.min()) - pad_roi)
            rx2 = min(w_img, int(nz_x.max()) + pad_roi + 1)
            roi_h = ry2 - ry1
            roi_w = rx2 - rx1

            erased_roi = erased_img[ry1:ry2, rx1:rx2]
            mask_roi = inpaint_mask[ry1:ry2, rx1:rx2]
            qr_roi = qr_mask[ry1:ry2, rx1:rx2]
            orig_roi = image[ry1:ry2, rx1:rx2]

            telea_roi = cv2.inpaint(erased_roi, mask_roi, 4, cv2.INPAINT_TELEA)
            telea_roi[qr_roi > 0] = orig_roi[qr_roi > 0]
            telea_gray = cv2.cvtColor(telea_roi, cv2.COLOR_BGR2GRAY)

            inpainted_roi = None
            try:
                scale_factor = 1.0
                roi_max_dim = max(roi_h, roi_w)
                if roi_max_dim > 1536:
                    scale_factor = 1536.0 / roi_max_dim
                    infer_w = max(8, int(roi_w * scale_factor))
                    infer_h = max(8, int(roi_h * scale_factor))
                    infer_img = cv2.resize(telea_roi, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                    infer_mask = cv2.resize(mask_roi, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
                else:
                    infer_img = telea_roi
                    infer_mask = mask_roi

                img_rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
                img_pil = PILImage.fromarray(img_rgb)
                mask_pil = PILImage.fromarray(infer_mask)
                del img_rgb

                grad_ctx = torch.no_grad() if HAS_TORCH else nullcontext()
                with grad_ctx:
                    res_pil = self._lama(img_pil, mask_pil)

                res_rgb = np.array(res_pil)
                del res_pil, img_pil, mask_pil
                res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
                del res_rgb

                infer_h_actual, infer_w_actual = infer_img.shape[:2]
                res_bgr = res_bgr[:infer_h_actual, :infer_w_actual]

                if scale_factor < 1.0:
                    inpainted_roi = cv2.resize(res_bgr, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC)
                else:
                    inpainted_roi = res_bgr
                del res_bgr

                dark_or_mid = np.expand_dims(telea_gray < 185, axis=2)
                inpainted_roi = np.where(dark_or_mid, np.minimum(inpainted_roi, telea_roi), inpainted_roi)
                del dark_or_mid

                logger.info("LaMa neural inpainting completed successfully.")
            except Exception as e:
                logger.error(f"LaMa execution encountered error: {e}. Falling back to OpenCV Telea.", exc_info=True)
                inpainted_roi = None
            finally:
                try:
                    from app.core.hardware import cleanup_gpu_memory
                    cleanup_gpu_memory(trim_ram=False)
                except Exception:
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            if inpainted_roi is None:
                if progress_callback:
                    progress_callback(60, "LaMa 资源受限，正在回退使用 OpenCV Telea 算法...")
                inpainted_roi = telea_roi

            filtered = cv2.bilateralFilter(inpainted_roi, 5, 50, 50)
            blended_roi = blend_inpainted_image(telea_roi, filtered, mask_roi, feather_radius)
            blended_roi[qr_roi > 0] = orig_roi[qr_roi > 0]
            erased_img[ry1:ry2, rx1:rx2] = blended_roi
            del telea_roi, telea_gray, inpainted_roi, filtered, blended_roi

        if progress_callback:
            progress_callback(100, "图像修复与去字完成")

        return erased_img
