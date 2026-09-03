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
    if feather_radius <= 0:
        result = original_img.copy()
        result[mask > 0] = inpainted_img[mask > 0]
        return result
    ksize = feather_radius * 2 + 1
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    alpha = np.expand_dims(alpha, axis=2)
    blended = inpainted_img.astype(np.float32) * alpha + original_img.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)

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
                feather_radius: int = 4, progress_callback=None) -> np.ndarray:
        if image is None or image.size == 0 or not blocks:
            return image.copy() if image is not None else None

        h_img, w_img = image.shape[:2]
        erased_img = image.copy()
        inpaint_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        base_dim = max(h_img, w_img)
        dyn_bubble_dil = max(1, int(base_dim * 0.002)) if bubble_dilation <= 0 else bubble_dilation
        dyn_onoma_dil = max(2, int(base_dim * 0.004)) if onomatopoeia_dilation <= 0 else onomatopoeia_dilation

        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if progress_callback:
                progress_callback(int((idx / max(1, total_blocks)) * 40), f"正在分析第 {idx+1}/{total_blocks} 个气泡区域背景...")

            xmin = int((block.get("xmin", 0) / 100.0) * w_img)
            ymin = int((block.get("ymin", 0) / 100.0) * h_img)
            xmax = int((block.get("xmax", 0) / 100.0) * w_img)
            ymax = int((block.get("ymax", 0) / 100.0) * h_img)

            xmin = max(0, min(xmin, w_img - 1))
            ymin = max(0, min(ymin, h_img - 1))
            xmax = max(0, min(xmax, w_img))
            ymax = max(0, min(ymax, h_img))

            if xmax <= xmin or ymax <= ymin:
                continue

            block_type = block.get("type", "bubble")
            crop = erased_img[ymin:ymax, xmin:xmax]
            bg_color = get_background_color_rgb(crop)
            text_mask = get_text_mask(crop, bg_color)

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

            if block_type == "bubble" and is_uniform:
                dilated = dilate_mask(text_mask, dyn_bubble_dil)
                crop[dilated == 255] = bg_color
                erased_img[ymin:ymax, xmin:xmax] = crop
            else:
                dilated = dilate_mask(text_mask, dyn_onoma_dil)
                inpaint_mask[ymin:ymax, xmin:xmax] = cv2.bitwise_or(inpaint_mask[ymin:ymax, xmin:xmax], dilated)

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

        if progress_callback:
            progress_callback(100, "图像背景修复完成")

        return erased_img
