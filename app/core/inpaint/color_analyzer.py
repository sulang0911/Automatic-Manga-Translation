"""
app/core/inpaint/color_analyzer.py
Analyzes speech bubble perimeter pixels, background luminance uniformity,
and extracts binary text masks and stroke colors.
"""
from typing import Tuple, List, Optional
import cv2
import numpy as np


def get_background_color_rgb(crop: np.ndarray) -> Tuple[int, int, int]:
    """
    Samples perimeter border pixels of a cropped image (BGR)
    and returns the median background color as an (R, G, B) tuple.
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
    b, g, r = int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2])
    return (r, g, b)


def get_background_color_hex(crop: np.ndarray) -> str:
    """Returns the median perimeter background color as an uppercase #RRGGBB hex string."""
    r, g, b = get_background_color_rgb(crop)
    return f"#{r:02X}{g:02X}{b:02X}"


def is_background_uniform(crop: np.ndarray, std_thresh: float = 16.0) -> bool:
    """
    Determines whether the background around a text block is solid/uniform
    by checking the standard deviation of perimeter luminance AND verifying
    that the interior background is not a multi-tone 3D/textured scene.
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
    if float(np.std(border_gray)) >= std_thresh:
        return False

    # Check if the crop contains coupled stroked text (both dark < 55 and bright > 180)
    # floating over a non-uniform 3D background
    if crop.ndim == 3 and h >= 10 and w >= 10:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        med_lum = float(np.median(border_gray))
        # Exclude pixels that deviate strongly from median background (i.e. text strokes)
        bg_candidates = gray[np.abs(gray.astype(np.float32) - med_lum) < 35.0]
        if len(bg_candidates) > 20 and float(np.std(bg_candidates)) > 9.5:
            return False

    return True


def get_stroked_text_mask(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Extracts a precise binary mask of stroked/outlined text (e.g. black text with white outline,
    or red/cyan text with white/dark outline) on non-uniform 3D/CG backgrounds.
    Ignores background objects (metal pipes, glowing lights, SFX burst lines) that lack coupled glyph cores.
    Returns None if the crop does not exhibit coupled stroked text.
    """
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < 68).astype(np.uint8) * 255
    bright = (gray > 165).astype(np.uint8) * 255

    # Detect high-saturation chromatic glyph cores (e.g. red SFX letters or cyan dialogue)
    c_int = crop.astype(np.int32)
    max_c = np.max(c_int, axis=2)
    min_c = np.min(c_int, axis=2)
    sat = max_c - min_c
    chroma_core = ((sat > 75) & (max_c > 125)).astype(np.uint8) * 255

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_dil = cv2.dilate(bright, k5)
    dark_dil = cv2.dilate(dark, k5)

    # Coupled dark+bright or chroma+bright strokes
    coupled_dark = cv2.bitwise_and(dark, bright_dil)
    coupled_chroma = cv2.bitwise_and(chroma_core, bright_dil)

    n_dark = int(np.count_nonzero(coupled_dark))
    n_chroma = int(np.count_nonzero(coupled_chroma))

    if n_dark + n_chroma < 12:
        return None

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    if n_chroma > n_dark and n_chroma >= 12:
        # Chromatic-core stroked text (e.g. red SFX letters with white outline on pink glow):
        # Do NOT include dark background pixels outside the glow so inpainting fills with pure glow color
        coupled_bright = cv2.bitwise_and(bright, cv2.dilate(chroma_core, k5))
        core_seeds = cv2.bitwise_or(coupled_chroma, coupled_bright)
        seed_zone = cv2.dilate(core_seeds, k9)
        aa_bright = (gray > 140).astype(np.uint8) * 255
        stroke_mask = cv2.bitwise_or(chroma_core, cv2.bitwise_and(aa_bright, seed_zone))
        k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.dilate(stroke_mask, k_halo, iterations=1)
    else:
        # Dark-core with white-outline (or white-core with dark-outline) text:
        coupled_bright = cv2.bitwise_and(bright, cv2.dilate(cv2.bitwise_or(dark, chroma_core), k5))
        core_seeds = cv2.bitwise_or(coupled_dark, cv2.bitwise_or(coupled_bright, coupled_chroma))
        seed_zone = cv2.dilate(core_seeds, k9)
        aa_bright = (gray > 110).astype(np.uint8) * 255
        aa_dark = (gray < 88).astype(np.uint8) * 255
        all_candidates = cv2.bitwise_or(aa_dark, cv2.bitwise_or(aa_bright, chroma_core))
        stroke_mask = cv2.bitwise_and(all_candidates, seed_zone)
        k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        return cv2.dilate(stroke_mask, k_halo, iterations=1)


def get_text_mask(crop: np.ndarray, bg_bgr: Tuple[int, int, int], threshold: int = 20) -> np.ndarray:
    """
    Generates a binary text mask (255 = text stroke, 0 = background)
    by computing maximum channel color deviation against background.
    Preserves chromatic edges (e.g. red/pink text on white backgrounds)
    that are lost in simple grayscale conversion.
    """
    if crop is None or crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    diff = np.abs(crop.astype(np.int32) - np.array(bg_bgr, dtype=np.int32))
    max_diff = np.max(diff, axis=2).astype(np.uint8)
    _, thresh = cv2.threshold(max_diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def analyze_text_color(crop: np.ndarray, bg_bgr: Tuple[int, int, int], dist_thresh: int = 60) -> str:
    if crop is None or crop.size == 0:
        return "#000000"

    h, w = crop.shape[:2]
    sample = crop
    if h > 160 and w > 320:
        sample = cv2.resize(crop, (min(w, 320), min(h, 160)), interpolation=cv2.INTER_AREA)

    if sample.ndim == 3:
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        bright_stroke = (gray > 180).astype(np.uint8)
        dark_stroke = (gray < 50).astype(np.uint8)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        near_stroke = (cv2.dilate(bright_stroke, k3) > 0) | (cv2.dilate(dark_stroke, k3) > 0)

        s_int = sample.astype(np.int32)
        b_ch, g_ch, r_ch = s_int[:, :, 0], s_int[:, :, 1], s_int[:, :, 2]
        max_c = np.max(s_int, axis=2)
        min_c = np.min(s_int, axis=2)
        sat = max_c - min_c

        cyan_glyph = (g_ch - r_ch > 30) & (b_ch - r_ch > 30) & (max_c > 140) & (sat > 45) & (cv2.dilate(bright_stroke, k3) > 0)
        if int(np.sum(cyan_glyph)) >= max(20, int(sample.shape[0] * sample.shape[1] * 0.015)):
            return "#3CE8F5"

        chroma_glyph = (sat > 100) & (max_c > 135) & near_stroke
        black_glyph = (gray < 45) & (cv2.dilate(bright_stroke, k3) > 0)
        n_chroma = int(np.sum(chroma_glyph))
        n_black = int(np.sum(black_glyph))

        if n_chroma >= max(15, int(sample.shape[0] * sample.shape[1] * 0.03)) and n_chroma > n_black * 1.5:
            avg_bgr = np.mean(sample[chroma_glyph], axis=0).astype(int)
            return f"#{int(avg_bgr[2]):02X}{int(avg_bgr[1]):02X}{int(avg_bgr[0]):02X}"
        if n_black >= max(15, int(sample.shape[0] * sample.shape[1] * 0.02)) and n_black >= n_chroma:
            bg_lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
            return "#000000" if bg_lum > 127 else "#FFFFFF"

    pixels = sample.reshape(-1, 3).astype(np.int32)
    bg_arr = np.array(bg_bgr, dtype=np.int32)
    dists = np.sum(np.abs(pixels - bg_arr), axis=1)
    text_mask = dists > dist_thresh

    if np.sum(text_mask) > 8:
        avg_bgr = np.mean(pixels[text_mask], axis=0).astype(int)
        b, g, r = avg_bgr[0], avg_bgr[1], avg_bgr[2]
        return f"#{int(r):02X}{int(g):02X}{int(b):02X}"
    else:
        bg_lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
        return "#000000" if bg_lum > 127 else "#FFFFFF"


def dilate_mask(mask: np.ndarray, dilation_pixels: int = 3) -> np.ndarray:
    if dilation_pixels <= 0 or mask is None:
        return mask
    kernel_size = dilation_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)
