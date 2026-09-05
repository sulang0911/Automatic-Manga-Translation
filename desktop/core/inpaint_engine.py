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

def get_text_mask(crop: np.ndarray, bg_color: List[int]) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg_gray = int(0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0])
    diff = cv2.absdiff(gray, bg_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
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
    if inpainted_img is None or mask is None or np.sum(mask) == 0:
        return original_img.copy() if original_img is not None else None
    if feather_radius <= 0:
        result = original_img.copy()
        result[mask > 0] = inpainted_img[mask > 0]
        return result
    ksize = feather_radius * 2 + 1
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    # Ensure core mask regions retain full inpainting opacity so thin text strokes don't bleed original text
    alpha = np.maximum(alpha, (mask.astype(np.float32) / 255.0))
    alpha_3d = np.expand_dims(alpha, axis=2)
    blended = inpainted_img.astype(np.float32) * alpha_3d + original_img.astype(np.float32) * (1.0 - alpha_3d)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Inverted dark background protection:
    # Ensure Gaussian alpha feathering does NOT bleed original white text edge pixels into dark/black regions
    # (preventing gray halos/smudges in dark boxes)
    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    inpaint_gray = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2GRAY)
    dark_inpaint = inpaint_gray < 60
    bright_orig = orig_gray > (inpaint_gray + 20)
    bleed_mask = (alpha > 0.05) & dark_inpaint & bright_orig
    blended[bleed_mask] = inpainted_img[bleed_mask]

    return blended

class InpaintEngine:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self._lama = None
        self._lama_available = False
        self._init_lama()

    def _init_lama(self):
        try:
            from simple_lama_inpainting import SimpleLama
            import torch
            has_cuda = torch.cuda.is_available()
            print(f"[*] Loading Simple-LaMa inpainter (CUDA: {has_cuda})...")
            self._lama = SimpleLama()
            self._lama_available = True
            print("[+] Simple-LaMa inpainter loaded successfully.")
        except Exception as e:
            print(f"[*] Info: Simple-LaMa not active: {e}. Using high-fidelity OpenCV inpainting.")
            self._lama_available = False

    def inpaint(self, image: np.ndarray, blocks: List[Dict[str, Any]], 
                bubble_dilation: int = 3, onomatopoeia_dilation: int = 6, 
                feather_radius: int = 4, progress_callback=None,
                qr_mask: Optional[np.ndarray] = None) -> np.ndarray:
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

        erased_img = image.copy()
        inpaint_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        base_dim = max(h_img, w_img)

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / max(1, total_blocks)) * 40), f"正在分析第 {idx+1}/{total_blocks} 个气泡区域背景...")

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

            text_mask = cv2.bitwise_and(text_mask, poly_mask)

            # Check background uniformity
            h_c, w_c = crop.shape[:2]
            border_pixels = []
            if h_c > 0 and w_c > 0:
                border_pixels.extend(crop[0, :])
                border_pixels.extend(crop[h_c - 1, :])
                if h_c > 2:
                    border_pixels.extend(crop[1:h_c - 1, 0])
                    border_pixels.extend(crop[1:h_c - 1, w_c - 1])
            border_pixels = np.array(border_pixels)
            is_uniform = True
            if len(border_pixels) > 0:
                border_gray = 0.299 * border_pixels[:, 2] + 0.587 * border_pixels[:, 1] + 0.114 * border_pixels[:, 0]
                is_uniform = np.std(border_gray) < 16.0

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
            if is_dark_bg or (block_type == "bubble" and is_uniform):
                fill_color = target_bg_color if is_dark_bg else bg_color
                dilated = dilate_mask(text_mask, adaptive_dil)
                if np.sum(text_mask) == 0:
                    dilated = dilate_mask(poly_mask, adaptive_dil)
                crop[(dilated == 255) & (qr_mask_crop == 0)] = fill_color
                erased_img[py_min:py_max, px_min:px_max] = crop
            else:
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
                progress_callback(50, "正在执行深度智能背景修复与羽化消除...")

            inpainted = None
            if self.mode != "opencv_telea" and self.mode != "opencv_ns" and self._lama_available and self._lama:
                try:
                    from PIL import Image as PILImage
                    img_rgb = cv2.cvtColor(erased_img, cv2.COLOR_BGR2RGB)
                    img_pil = PILImage.fromarray(img_rgb)
                    mask_pil = PILImage.fromarray(inpaint_mask)
                    res_pil = self._lama(img_pil, mask_pil)
                    inpainted_rgb = np.array(res_pil)
                    inpainted = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
                    if inpainted.shape[:2] != (h_img, w_img):
                        inpainted = inpainted[:h_img, :w_img]
                except Exception as e:
                    print(f"[-] LaMa execution error: {e}. Fallback to OpenCV.")
                    inpainted = None

            if inpainted is None:
                inpaint_flag = cv2.INPAINT_NS if self.mode == "opencv_ns" else cv2.INPAINT_TELEA
                inpainted = cv2.inpaint(erased_img, inpaint_mask, 3, inpaint_flag)

            inpainted_filtered = cv2.bilateralFilter(inpainted, 5, 50, 50)
            dyn_feather = max(2, int(base_dim * 0.003)) if feather_radius <= 0 else feather_radius
            erased_img = blend_inpainted_image(erased_img, inpainted_filtered, inpaint_mask, dyn_feather)
            # Guarantee zero-pixel modification on QR code areas
            erased_img[qr_mask > 0] = image[qr_mask > 0]

        if progress_callback:
            progress_callback(100, "图像背景修复完成")

        return erased_img
