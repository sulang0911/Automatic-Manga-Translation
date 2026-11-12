import cv2
import numpy as np
from typing import List, Dict, Any, Optional

def get_background_color_rgb(crop: np.ndarray) -> List[int]:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return [255, 255, 255]
    border_pixels = []
    border_pixels.extend(crop[0, :])
    border_pixels.extend(crop[h - 1, :])
    if h > 2:
        border_pixels.extend(crop[1:h - 1, 0])
        border_pixels.extend(crop[1:h - 1, w - 1])
    border_pixels = np.array(border_pixels)
    if len(border_pixels) == 0:
        return [255, 255, 255]
    median_color = np.median(border_pixels, axis=0).astype(int)
    return [int(c) for c in median_color]

def get_text_mask(crop: np.ndarray, bg_color: List[int], threshold: int = 20) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    diff = np.abs(crop.astype(np.int32) - np.array(bg_color, dtype=np.int32))
    max_diff = np.max(diff, axis=2).astype(np.uint8)
    _, thresh = cv2.threshold(max_diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def dilate_mask(mask: np.ndarray, dilation_pixels: int = 4) -> np.ndarray:
    if dilation_pixels <= 0:
        return mask
    kernel_size = dilation_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)

def blend_inpainted_image(original_img: np.ndarray, inpainted_img: np.ndarray, mask: np.ndarray, feather_radius: int = 4) -> np.ndarray:
    if original_img is None:
        return None
    if inpainted_img is None or mask is None or not np.any(mask):
        return original_img.copy() if original_img is not None else None
    if feather_radius <= 0:
        result = original_img.copy()
        result[mask > 0] = inpainted_img[mask > 0]
        return result

    h_img, w_img = original_img.shape[:2]
    # Bound float32 blending strictly to the active mask bounding box (+ feather margin)
    # Outside this padded box, GaussianBlur of mask is 0.0 and output is 100% original_img.
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
    # Ensure Gaussian alpha feathering does NOT bleed original bright text edge pixels into dark/mid-tone regions
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

_GLOBAL_SIMPLE_LAMA = None

class InpaintEngine:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._lama = None
        self._lama_available = False
        self._init_lama()

    def _init_lama(self):
        global _GLOBAL_SIMPLE_LAMA
        if _GLOBAL_SIMPLE_LAMA is not None:
            self._lama = _GLOBAL_SIMPLE_LAMA
            self._lama_available = True
            return
        try:
            from simple_lama_inpainting import SimpleLama
            import torch
            has_cuda = torch.cuda.is_available()
            print(f"[*] Loading Simple-LaMa inpainter (CUDA: {has_cuda})...")
            _GLOBAL_SIMPLE_LAMA = SimpleLama()
            self._lama = _GLOBAL_SIMPLE_LAMA
            self._lama_available = True
            print("[+] Simple-LaMa inpainter loaded successfully.")
        except Exception as e:
            print(f"[*] Info: Simple-LaMa not active: {e}. Using high-fidelity OpenCV inpainting.")
            self._lama_available = False

    def inpaint(self, image: np.ndarray, blocks: List[Dict[str, Any]], 
                bubble_dilation: int = 3, onomatopoeia_dilation: int = 6, 
                feather_radius: int = 4, progress_callback=None,
                qr_mask: Optional[np.ndarray] = None,
                manual_mask: Optional[np.ndarray] = None,
                onomatopoeia_mode: str = "normal", **kwargs) -> np.ndarray:
        if image is None or image.size == 0 or not blocks:
            return image.copy() if image is not None else None

        h_img, w_img = image.shape[:2]

        # Detect or use QR protection mask
        if qr_mask is None:
            try:
                from app.core.ocr.qr_filter import QRCodeFilter
                filt = QRCodeFilter()
                qr_regs = filt.detect_regions(image)
                qr_mask = filt.get_protection_mask((h_img, w_img), qr_regs)
            except Exception:
                qr_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        if manual_mask is not None:
            qr_mask = cv2.bitwise_or(qr_mask, manual_mask)

        erased_img = image.copy()
        inpaint_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        base_dim = max(h_img, w_img)

        from app.core.inpaint.unboxed_text_eraser import is_unboxed_text_block, erase_unboxed_text_block

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / max(1, total_blocks)) * 40), f"正在分析第 {idx+1}/{total_blocks} 个气泡区域背景...")

            block_type = block.get("type", "bubble") if isinstance(block, dict) else getattr(block, "type", "bubble")
            erase_override = block.get("force_erase", False) if isinstance(block, dict) else getattr(block, "force_erase", False)
            has_trans = bool(block.get("translated_text", "") if isinstance(block, dict) else getattr(block, "translated_text", ""))

            # Only skip if explicitly configured to ignore onomatopoeia AND not forced to erase AND has no translation
            if block_type == "onomatopoeia" and onomatopoeia_mode == "ignore" and not erase_override and not has_trans:
                continue

            if is_unboxed_text_block(block, erased_img):
                erased_img, unboxed_mask = erase_unboxed_text_block(
                    erased_img, block, qr_mask=qr_mask, min_dilation=max(3, onomatopoeia_dilation)
                )
                if unboxed_mask is not None and np.any(unboxed_mask):
                    inpaint_mask = cv2.bitwise_or(inpaint_mask, unboxed_mask)
                continue

            poly = None
            if hasattr(block, "to_pixel_polygon"):
                poly = block.to_pixel_polygon(w_img, h_img)
            elif isinstance(block, dict):
                if block.get("polygon") and len(block["polygon"]) >= 3:
                    poly = [[int(round(p[0])), int(round(p[1]))] for p in block["polygon"]]
                else:
                    bx1 = int((block.get("xmin", 0) / 100.0) * w_img)
                    by1 = int((block.get("ymin", 0) / 100.0) * h_img)
                    bx2 = int((block.get("xmax", 0) / 100.0) * w_img)
                    by2 = int((block.get("ymax", 0) / 100.0) * h_img)
                    eff_angle = float(block.get("angle_override") if block.get("angle_override") is not None else block.get("angle", 0.0) or 0.0)
                    if abs(eff_angle) < 2.5:
                        poly = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
                    else:
                        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
                        rect = ((cx, cy), (bw, bh), eff_angle)
                        box = cv2.boxPoints(rect)
                        poly = [[int(round(p[0])), int(round(p[1]))] for p in box]

            if poly is None:
                continue

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
            bg_color = get_background_color_rgb(crop)
            text_mask = get_text_mask(crop, bg_color)

            # Local polygon mask
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

            text_mask = cv2.bitwise_and(text_mask, poly_mask)

            from app.core.inpaint.color_analyzer import is_background_uniform, get_stroked_text_mask
            is_uniform = is_background_uniform(crop, std_thresh=16.0)

            qr_mask_crop = qr_mask[py_min:py_max, px_min:px_max]

            block_bg = block.get("bg_color_override") or block.get("bg_color") if isinstance(block, dict) else (getattr(block, "bg_color_override", None) or getattr(block, "bg_color", None))
            bg_hex_lum = None
            hex_bgr = None
            if block_bg and isinstance(block_bg, str):
                s_clean = block_bg.strip()
                if s_clean.startswith("#") and len(s_clean) >= 7:
                    try:
                        hr = int(s_clean[1:3], 16)
                        hg = int(s_clean[3:5], 16)
                        hb = int(s_clean[5:7], 16)
                        hex_bgr = [hb, hg, hr]
                        bg_hex_lum = 0.299 * hr + 0.587 * hg + 0.114 * hb
                    except ValueError:
                        pass

            crop_bg_lum = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]

            is_dark_bg = False
            if block_bg and isinstance(block_bg, str) and block_bg.strip().lower() == "#000000":
                is_dark_bg = True
            elif bg_hex_lum is not None and bg_hex_lum < 50.0:
                is_dark_bg = True
            elif crop_bg_lum < 50.0:
                is_dark_bg = True

            if is_dark_bg:
                if block_bg and isinstance(block_bg, str) and block_bg.strip().lower() == "#000000":
                    target_bg_color = [0, 0, 0]
                elif bg_hex_lum is not None and bg_hex_lum < 50.0 and (crop_bg_lum >= 50.0 or crop_bg_lum > bg_hex_lum + 30):
                    target_bg_color = hex_bgr
                elif crop_bg_lum < 50.0:
                    target_bg_color = bg_color
                else:
                    target_bg_color = hex_bgr if hex_bgr is not None else bg_color

                text_mask = get_text_mask(crop, target_bg_color)
                text_mask = cv2.bitwise_and(text_mask, poly_mask)

            block_type = block.get("type", "bubble") if isinstance(block, dict) else getattr(block, "type", "bubble")
            explicit_black_box = bool(block_bg and isinstance(block_bg, str) and block_bg.strip().lower() == "#000000")
            if explicit_black_box or is_uniform or (block_type == "bubble" and is_uniform):
                fill_color = target_bg_color if is_dark_bg else bg_color
                dilated = dilate_mask(text_mask, adaptive_dil)
                if not np.any(text_mask):
                    dilated = dilate_mask(poly_mask, adaptive_dil)
                crop[(dilated == 255) & (qr_mask_crop == 0)] = fill_color
                erased_img[py_min:py_max, px_min:px_max] = crop
            else:
                pad_s = 10
                epx_min = max(0, px_min - pad_s)
                epy_min = max(0, py_min - pad_s)
                epx_max = min(w_img, px_max + pad_s)
                epy_max = min(h_img, py_max + pad_s)
                exp_crop = erased_img[epy_min:epy_max, epx_min:epx_max]
                stroked_mask = get_stroked_text_mask(exp_crop)
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
                progress_callback(50, "正在执行深度智能背景修复与羽化消除...")

            dyn_feather = max(2, int(base_dim * 0.003)) if feather_radius <= 0 else feather_radius

            # Compute active ROI around inpaint_mask so OpenCV Telea, Simple-LaMa, and bilateralFilter
            # only process the active masked region rather than allocating massive full-page tensors!
            nz_y, nz_x = np.nonzero(inpaint_mask)
            pad_roi = max(48, dyn_feather * 4 + 16)
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

            inpaint_flag = cv2.INPAINT_NS if self.mode == "opencv_ns" else cv2.INPAINT_TELEA
            telea_roi = cv2.inpaint(erased_roi, mask_roi, 4, inpaint_flag)
            telea_roi[qr_roi > 0] = orig_roi[qr_roi > 0]
            telea_gray = cv2.cvtColor(telea_roi, cv2.COLOR_BGR2GRAY)

            inpainted_roi = None
            if self.mode != "opencv_telea" and self.mode != "opencv_ns" and self._lama_available and self._lama:
                try:
                    from PIL import Image as PILImage
                    try:
                        import torch
                        grad_ctx = torch.no_grad()
                    except Exception:
                        from contextlib import nullcontext
                        grad_ctx = nullcontext()

                    # VRAM-safe downscale cap (1536px max dimension) to prevent 1080 Ti VRAM spill into 8GB System RAM
                    max_lama_dim = 1536
                    roi_max_dim = max(roi_h, roi_w)
                    scale_factor = 1.0
                    if roi_max_dim > max_lama_dim:
                        scale_factor = float(max_lama_dim) / float(roi_max_dim)
                        infer_w = max(8, int(round(roi_w * scale_factor)))
                        infer_h = max(8, int(round(roi_h * scale_factor)))
                        infer_img = cv2.resize(telea_roi, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                        infer_mask = cv2.resize(mask_roi, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        infer_img = telea_roi
                        infer_mask = mask_roi

                    img_rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
                    img_pil = PILImage.fromarray(img_rgb)
                    mask_pil = PILImage.fromarray(infer_mask)
                    del img_rgb

                    with grad_ctx:
                        res_pil = self._lama(img_pil, mask_pil)

                    inpainted_rgb = np.array(res_pil)
                    del res_pil, img_pil, mask_pil
                    res_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
                    del inpainted_rgb

                    infer_h_actual, infer_w_actual = infer_img.shape[:2]
                    if res_bgr.shape[:2] != (infer_h_actual, infer_w_actual):
                        res_bgr = res_bgr[:infer_h_actual, :infer_w_actual]

                    if scale_factor < 1.0:
                        inpainted_roi = cv2.resize(res_bgr, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC)
                    else:
                        inpainted_roi = res_bgr
                    del res_bgr

                    # Prevent Simple-LaMa Fourier convolution from hallucinating bright/white ghost smudges
                    # over dark or mid-tone backgrounds where Telea diffusion is already clean
                    dark_or_mid = np.expand_dims(telea_gray < 185, axis=2)
                    inpainted_roi = np.where(dark_or_mid, np.minimum(inpainted_roi, telea_roi), inpainted_roi)
                    del dark_or_mid
                except Exception as e:
                    print(f"[-] LaMa execution error: {e}. Fallback to OpenCV.")
                    inpainted_roi = None
                finally:
                    try:
                        from app.core.hardware import cleanup_gpu_memory
                        cleanup_gpu_memory(trim_ram=False)
                    except Exception:
                        pass

            if inpainted_roi is None:
                inpainted_roi = telea_roi

            inpainted_filtered = cv2.bilateralFilter(inpainted_roi, 5, 50, 50)
            blended_roi = blend_inpainted_image(telea_roi, inpainted_filtered, mask_roi, dyn_feather)
            blended_roi[qr_roi > 0] = orig_roi[qr_roi > 0]
            erased_img[ry1:ry2, rx1:rx2] = blended_roi
            del telea_roi, telea_gray, inpainted_roi, inpainted_filtered, blended_roi

        if progress_callback:
            progress_callback(100, "图像背景修复完成")

        return erased_img
