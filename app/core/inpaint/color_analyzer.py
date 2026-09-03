"""
app/core/inpaint/color_analyzer.py
Perimeter median background sampling, text stroke mask generation, and background uniformity analysis.
"""
from typing import Tuple, List
import cv2
import numpy as np


def get_background_color_rgb(crop: np.ndarray) -> Tuple[int, int, int]:
    """
    Extracts the median color of the perimeter border pixels of a crop.
    Input: BGR image crop.
    Returns: (R, G, B) integer tuple in [0, 255].
    """
    if crop is None or crop.size == 0:
        return (255, 255, 255)
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return (255, 255, 255)

    border_pixels: List[np.ndarray] = []
    border_pixels.extend(crop[0, :])
    border_pixels.extend(crop[h - 1, :])
    if h > 2:
        border_pixels.extend(crop[1:h - 1, 0])
        border_pixels.extend(crop[1:h - 1, w - 1])

    border_arr = np.array(border_pixels)
    if len(border_arr) == 0:
        return (255, 255, 255)

    median_bgr = np.median(border_arr, axis=0).astype(int)
    b, g, r = median_bgr[0], median_bgr[1], median_bgr[2]
    return (int(r), int(g), int(b))


def get_background_color_hex(crop: np.ndarray) -> str:
    """Returns background color as '#RRGGBB' string."""
    r, g, b = get_background_color_rgb(crop)
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def is_background_uniform(crop: np.ndarray, std_thresh: float = 15.0) -> bool:
    """
    Evaluates whether perimeter pixels have uniform luminance (speech bubble)
    or high variance (textured screentone/background art).
    """
    if crop is None or crop.size == 0:
        return True
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return True

    border_pixels: List[np.ndarray] = []
    border_pixels.extend(crop[0, :])
    border_pixels.extend(crop[h - 1, :])
    if h > 2:
        border_pixels.extend(crop[1:h - 1, 0])
        border_pixels.extend(crop[1:h - 1, w - 1])

    border_arr = np.array(border_pixels)
    if len(border_arr) == 0:
        return True

    # ITU-R BT.601 luminance on BGR
    border_gray = 0.299 * border_arr[:, 2] + 0.587 * border_arr[:, 1] + 0.114 * border_arr[:, 0]
    return float(np.std(border_gray)) < std_thresh


def get_text_mask(crop: np.ndarray, bg_bgr: Tuple[int, int, int], threshold: int = 25) -> np.ndarray:
    """
    Generates a binary text mask (255 = text stroke, 0 = background)
    by computing absolute grayscale difference against background luminance.
    """
    if crop is None or crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg_gray = int(0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2])
    diff = cv2.absdiff(gray, bg_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def analyze_text_color(crop: np.ndarray, bg_bgr: Tuple[int, int, int], dist_thresh: int = 60) -> str:
    """
    Extracts text stroke color by computing the average RGB of pixels that deviate
    from the background color by more than dist_thresh Manhattan distance.
    Falls back to luminance extrema if text stroke pixel count is small.
    """
    if crop is None or crop.size == 0:
        return "#000000"

    h, w = crop.shape[:2]
    # Downscale for performance if crop is large
    sample = crop
    if h > 80 or w > 150:
        sample = cv2.resize(crop, (min(w, 150), min(h, 80)), interpolation=cv2.INTER_AREA)

    pixels = sample.reshape(-1, 3).astype(np.int32)
    bg_arr = np.array(bg_bgr, dtype=np.int32)
    # Manhattan distance from bg_bgr
    dists = np.sum(np.abs(pixels - bg_arr), axis=1)
    text_mask = dists > dist_thresh

    if np.sum(text_mask) > 8:
        avg_bgr = np.mean(pixels[text_mask], axis=0).astype(int)
        b, g, r = avg_bgr[0], avg_bgr[1], avg_bgr[2]
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
    else:
        # Fallback: compute bg luminance
        bg_lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
        return "#000000" if bg_lum > 127 else "#FFFFFF"


def dilate_mask(mask: np.ndarray, dilation_pixels: int = 3) -> np.ndarray:
    """Expands a binary mask by dilation_pixels using a rectangular kernel."""
    if dilation_pixels <= 0 or mask is None:
        return mask
    kernel_size = dilation_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)
