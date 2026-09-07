"""
app/core/inpaint/unboxed_text_eraser.py
High-precision erasure algorithm for unboxed / free text (chapter titles, page margin notes,
sound effects / onomatopoeia, and floating text without explicit speech bubbles).

Core capabilities:
1. Intelligent Unboxed Text Detection: Distinguishes free-standing chapter titles, margin notes,
   and SFX from enclosed dialogue bubbles, even when default OCR categorizes them as bubbles.
2. Orientation-Aware Adaptive Expansion: Expands tight OCR bounding boxes according to character
   scale and reading orientation to capture sub-pixel anti-aliasing edges, ascenders, descenders, and accents.
3. Contrast-Aware Boundary Barrier Protection: Ray-casts outward in 4 directions to detect and halt before
   crossing high-contrast comic panel frames or artwork borders (adapting to both light margins and dark banners).
4. Text-Insulated Peripheral Uniformity Analysis: Accurately isolates background variance from text stroke
   edge artifacts to distinguish solid/flat margins from textured artwork.
5. Dual-Mode Erasure:
   - Uniform backgrounds: Instant pristine flat-fill using peripheral background color (eliminates
     blurry gray halos, smudges, and partial-erasure artifacts).
   - Textured artwork backgrounds: Delivers an expanded, fully dilated stroke mask for seamless
     diffusion / neural inpainting without boundary stroke bleeding.
"""
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


def _parse_block_coordinates(block: Any, w_img: int, h_img: int) -> Optional[np.ndarray]:
    """Extracts integer pixel polygon points (N, 2) from a block or dict."""
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
            poly = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
    else:
        return None

    if poly is None or len(poly) < 3:
        return None
    return np.array(poly, dtype=np.int32)


def is_unboxed_text_block(block: Any, image: Optional[np.ndarray] = None) -> bool:
    """
    Determines whether a block represents text without an explicit speech bubble or text box.
    
    Checks:
    1. Explicit type designation (!= 'bubble') or 'is_unboxed' flag.
    2. If image context is supplied, checks for margin placement, banner aspect ratio,
       or open unboxed background even if OCR engine defaulted the type to 'bubble'.
    """
    b_type = block.get("type", "bubble") if isinstance(block, dict) else getattr(block, "type", "bubble")
    if b_type != "bubble":
        return True
    if isinstance(block, dict) and block.get("is_unboxed"):
        return True
    if hasattr(block, "is_unboxed") and getattr(block, "is_unboxed"):
        return True

    # Image-aware heuristic for blocks that lack an explicit speech bubble
    if image is not None and image.size > 0:
        h_img, w_img = image.shape[:2]
        poly_pts = _parse_block_coordinates(block, w_img, h_img)
        if poly_pts is not None and len(poly_pts) >= 3:
            px_min = max(0, int(np.min(poly_pts[:, 0])))
            py_min = max(0, int(np.min(poly_pts[:, 1])))
            px_max = min(w_img, int(np.max(poly_pts[:, 0])))
            py_max = min(h_img, int(np.max(poly_pts[:, 1])))
            bw = max(1, px_max - px_min)
            bh = max(1, py_max - py_min)
            aspect = bw / bh

            # Check if block is positioned in outer page margins (typical for chapter titles & notes)
            is_top_margin = py_max <= 0.16 * h_img
            is_bottom_margin = py_min >= 0.88 * h_img
            is_side_margin = (px_max <= 0.08 * w_img) or (px_min >= 0.92 * w_img)
            is_extreme_aspect = (aspect >= 4.0) or (aspect <= 0.15)

            if is_top_margin or is_bottom_margin or is_side_margin or is_extreme_aspect:
                # Sample local background around the block
                pad = 4
                s_ymin, s_ymax = max(0, py_min - pad), min(h_img, py_max + pad)
                s_xmin, s_xmax = max(0, px_min - pad), min(w_img, px_max + pad)
                crop = image[s_ymin:s_ymax, s_xmin:s_xmax]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                    perim = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
                    if len(perim) > 0:
                        med = float(np.median(perim))
                        clean_perim = perim[np.abs(perim - med) < 35]
                        std = float(np.std(clean_perim)) if len(clean_perim) > 4 else float(np.std(perim))
                        if std < 25.0 or med > 225 or med < 40:
                            return True

    return False


def erase_unboxed_text_block(
    image: np.ndarray,
    block: Any,
    qr_mask: Optional[np.ndarray] = None,
    border_stop_threshold: int = 80,
    min_dilation: int = 3
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Erases text without an explicit text box from the given image.

    Args:
        image: BGR numpy image array (H, W, 3).
        block: TranslationBlock instance or dict containing polygon/box coordinates.
        qr_mask: Optional binary mask (H, W) where >0 pixels must never be altered.
        border_stop_threshold: Grayscale pixel intensity reference for barrier detection.
        min_dilation: Minimum morphological dilation radius in pixels.

    Returns:
        (modified_image, inpaint_mask_contribution)
        If the local background is uniform, the text is flat-filled directly in modified_image,
        and inpaint_mask_contribution is None.
        If the background is textured, inpaint_mask_contribution is a binary mask (H, W) containing
        the dilated text stroke region to be inpainted.
    """
    if image is None or image.size == 0:
        return image, None

    h_img, w_img = image.shape[:2]
    poly_pts = _parse_block_coordinates(block, w_img, h_img)
    if poly_pts is None or len(poly_pts) < 3:
        return image, None

    px_min = max(0, int(np.min(poly_pts[:, 0])))
    py_min = max(0, int(np.min(poly_pts[:, 1])))
    px_max = min(w_img, int(np.max(poly_pts[:, 0])))
    py_max = min(h_img, int(np.max(poly_pts[:, 1])))

    if px_max <= px_min or py_max <= py_min:
        return image, None

    bw = px_max - px_min
    bh = py_max - py_min
    char_dim = max(10, min(bw, bh))

    # Orientation-aware adaptive expansion margins
    if bw >= bh:
        pad_x = max(8, int(round(bw * 0.05)), int(round(bh * 0.30)))
        pad_y = max(6, int(round(bh * 0.35)))
    else:
        pad_y = max(8, int(round(bh * 0.05)), int(round(bw * 0.30)))
        pad_x = max(6, int(round(bw * 0.35)))

    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Initial background luminance estimate from block properties or surrounding corners
    bg_lum_est = 255.0
    block_bg = block.get("bg_color_override") or block.get("bg_color") if isinstance(block, dict) else (
        getattr(block, "bg_color_override", None) or getattr(block, "bg_color", None)
    )
    if block_bg and isinstance(block_bg, str) and block_bg.strip().startswith("#") and len(block_bg.strip()) >= 7:
        try:
            hr = int(block_bg.strip()[1:3], 16)
            hg = int(block_bg.strip()[3:5], 16)
            hb = int(block_bg.strip()[5:7], 16)
            bg_lum_est = 0.299 * hr + 0.587 * hg + 0.114 * hb
        except ValueError:
            pass
    else:
        # Sample outer corners
        corner_pts = [
            (max(0, px_min - 2), max(0, py_min - 2)),
            (min(w_img - 1, px_max + 2), max(0, py_min - 2)),
            (max(0, px_min - 2), min(h_img - 1, py_max + 2)),
            (min(w_img - 1, px_max + 2), min(h_img - 1, py_max + 2)),
        ]
        corner_vals = [float(gray_full[cy, cx]) for cx, cy in corner_pts]
        bg_lum_est = float(np.median(corner_vals))

    is_dark_bg = (bg_lum_est < 90.0)

    # Contrast-aware barrier values
    # In manga, comic panel borders are solid lines dividing panels or page margins.
    if is_dark_bg:
        barrier_val = max(150, min(235, int(bg_lum_est + 60)))
        def is_barrier(slice_arr):
            return len(slice_arr) > 0 and np.mean(slice_arr > barrier_val) > 0.60
    else:
        barrier_val = min(100, max(20, int(bg_lum_est - 60)))
        def is_barrier(slice_arr):
            return len(slice_arr) > 0 and np.mean(slice_arr < barrier_val) > 0.60

    # 1. Search upwards: halt if a strong frame border barrier is encountered
    exp_y_min = py_min
    for y in range(py_min - 1, max(0, py_min - pad_y) - 1, -1):
        row_slice = gray_full[y, max(0, px_min - pad_x):min(w_img, px_max + pad_x)]
        if is_barrier(row_slice):
            break
        exp_y_min = y

    # 2. Search downwards: halt if a strong frame border barrier is encountered
    exp_y_max = py_max
    for y in range(py_max, min(h_img, py_max + pad_y)):
        row_slice = gray_full[y, max(0, px_min - pad_x):min(w_img, px_max + pad_x)]
        if is_barrier(row_slice):
            break
        exp_y_max = y + 1

    # 3. Search leftwards: halt if a strong border barrier is encountered
    exp_x_min = px_min
    for x in range(px_min - 1, max(0, px_min - pad_x) - 1, -1):
        col_slice = gray_full[exp_y_min:exp_y_max, x]
        if is_barrier(col_slice):
            break
        exp_x_min = x

    # 4. Search rightwards: halt if a strong border barrier is encountered
    exp_x_max = px_max
    for x in range(px_max, min(w_img, px_max + pad_x)):
        col_slice = gray_full[exp_y_min:exp_y_max, x]
        if is_barrier(col_slice):
            break
        exp_x_max = x + 1

    crop = image[exp_y_min:exp_y_max, exp_x_min:exp_x_max].copy()
    crop_gray = gray_full[exp_y_min:exp_y_max, exp_x_min:exp_x_max]
    ch, cw = crop.shape[:2]

    if ch < 2 or cw < 2:
        return image, None

    # Sample peripheral perimeter pixels (avoiding inner text glyphs)
    border_pixels = []
    border_pixels.extend(crop[0, :])
    border_pixels.extend(crop[-1, :])
    if ch > 2:
        border_pixels.extend(crop[1:-1, 0])
        border_pixels.extend(crop[1:-1, -1])
    border_pixels = np.array(border_pixels)

    if len(border_pixels) == 0:
        med_bgr = [255, 255, 255] if not is_dark_bg else [20, 20, 20]
    else:
        med_bgr = np.median(border_pixels, axis=0).astype(int)

    bg_color = [int(c) for c in med_bgr]
    border_gray = 0.299 * border_pixels[:, 2] + 0.587 * border_pixels[:, 1] + 0.114 * border_pixels[:, 0]
    bg_lum = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]

    # Explicit block background override check
    if block_bg and isinstance(block_bg, str) and block_bg.strip().startswith("#") and len(block_bg.strip()) >= 7:
        try:
            hr = int(block_bg.strip()[1:3], 16)
            hg = int(block_bg.strip()[3:5], 16)
            hb = int(block_bg.strip()[5:7], 16)
            override_lum = 0.299 * hr + 0.587 * hg + 0.114 * hb
            if block_bg.strip().lower() == "#000000" or override_lum < 30.0:
                bg_color = [hb, hg, hr]
                bg_lum = override_lum
        except ValueError:
            pass

    # Filter out text stroke pixels touching the edge when measuring background std
    clean_border = border_gray[np.abs(border_gray - bg_lum) < 35.0]
    border_std = float(np.std(clean_border)) if len(clean_border) > 4 else float(np.std(border_gray))

    # Uniformity decision: Flat margin, pure white page gutter, or solid dark background
    is_uniform = (border_std < 18.0) or (bg_lum > 225 and border_std < 25.0) or (bg_lum < 40 and border_std < 25.0)

    # Text stroke segmentation: Difference against background luminance
    bg_gray_val = int(round(bg_lum))
    diff = cv2.absdiff(crop_gray, bg_gray_val)
    thresh_val = 18 if is_uniform else 25
    _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological dilation tailored to character scale to ensure full coverage
    dil_size = max(min_dilation, int(round(char_dim * 0.12)))
    dil_size = max(2, min(dil_size, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_size * 2 + 1, dil_size * 2 + 1))
    dilated_mask = cv2.dilate(thresh, kernel)

    # Respect QR code protection mask
    qr_crop = None
    if qr_mask is not None:
        qr_crop = qr_mask[exp_y_min:exp_y_max, exp_x_min:exp_x_max]

    if is_uniform:
        out_img = image.copy()
        fill_mask = (dilated_mask > 0)
        if qr_crop is not None:
            fill_mask = fill_mask & (qr_crop == 0)
        crop[fill_mask] = bg_color
        out_img[exp_y_min:exp_y_max, exp_x_min:exp_x_max] = crop
        return out_img, None
    else:
        contrib = np.zeros((h_img, w_img), dtype=np.uint8)
        if qr_crop is not None:
            dilated_mask[qr_crop > 0] = 0
        contrib[exp_y_min:exp_y_max, exp_x_min:exp_x_max] = dilated_mask
        return image, contrib
