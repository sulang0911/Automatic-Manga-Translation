"""
app/core/inpaint/color_analyzer.py
Analyzes speech bubble perimeter pixels, background luminance uniformity,
and extracts binary text masks, per-block text colors, and per-line multi-speaker colors.
"""
from typing import Tuple, List, Optional, Any
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


def get_stroked_text_mask(crop: np.ndarray, img_max_dim: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Extracts a precise binary mask of stroked/outlined text (e.g. black text with white outline,
    or red/cyan text with white/dark outline) on non-uniform 3D/CG backgrounds.
    Scales morphological coupling, outline enclosure, and outer-glow dilation proportionally
    with page resolution (img_max_dim) and measured halo falloff width.
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

    # Estimate resolution / soft-glow scale; TOPHAT strips large white walls so only compact halos measure width.
    if img_max_dim is not None and img_max_dim > 0:
        s_dim = max(1.0, float(img_max_dim) / 1024.0)
    else:
        s_dim = max(1.0, float(max(crop.shape[:2])) / 650.0)
    white_aa = cv2.bitwise_and((gray > 105).astype(np.uint8) * 255, cv2.bitwise_not(chroma_core))
    k25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    halo_only = cv2.morphologyEx(white_aa, cv2.MORPH_TOPHAT, k25)
    dt_w = cv2.distanceTransform(halo_only, cv2.DIST_L2, 3)
    near_d = cv2.dilate(cv2.bitwise_or(dark, chroma_core), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    vals = dt_w[(halo_only > 0) & (near_d > 0)]
    med_hw = float(np.percentile(vals, 85)) if len(vals) > 20 else 2.0
    s = max(1.0, min(4.0, max(s_dim, min(3.2, med_hw / 2.2))))

    couple_k = max(5, min(13, int(round(2.0 * s)) * 2 + 1))
    k_couple = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (couple_k, couple_k))
    bright_dil = cv2.dilate(bright, k_couple)

    # Filter out large/thick chromatic background regions (e.g. yellow floor, warm wood, colored walls)
    # so colored scenery behind black+white-outline text does not masquerade as chromatic text strokes.
    text_chroma = chroma_core
    if np.any(chroma_core):
        h_c, w_c = chroma_core.shape
        crop_area = float(h_c * w_c)
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(chroma_core)
        erode_k = max(7, min(25, int(round(3.0 * s)) * 2 + 1))
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
        wo_dil = cv2.dilate(cv2.bitwise_and(bright, cv2.bitwise_not(chroma_core)), k_couple)
        text_chroma = np.zeros_like(chroma_core)
        for cid in range(1, n_cc):
            x, y, cw, ch, area = stats[cid]
            touches_border = (x == 0 or y == 0 or x + cw >= w_c or y + ch >= h_c)
            if touches_border and area > crop_area * 0.04:
                continue
            comp_mask = (labels == cid).astype(np.uint8) * 255
            eroded_px = np.count_nonzero(cv2.erode(comp_mask, k_erode))
            if touches_border and eroded_px > area * 0.25:
                continue
            if not touches_border and area > crop_area * 0.08 and eroded_px > area * 0.35:
                if np.count_nonzero(cv2.bitwise_and(comp_mask, wo_dil)) < area * 0.45:
                    continue
            text_chroma[labels == cid] = 255

    # Coupled dark+bright or chromatic-glyph+bright strokes
    coupled_dark = cv2.bitwise_and(dark, bright_dil)
    coupled_chroma = cv2.bitwise_and(text_chroma, bright_dil)

    n_dark = int(np.count_nonzero(coupled_dark))
    n_chroma = int(np.count_nonzero(coupled_chroma))

    if n_dark + n_chroma < 12:
        return None

    # Dark glyph strokes enclosed by bright outlines (black fill + white outline) must not be
    # discarded even when chromatic elements also appear in the crop.
    seed_r = max(4, min(16, int(round(4.0 * s))))
    k_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_r * 2 + 1, seed_r * 2 + 1))
    bright_closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_seed)
    n_dark_enclosed = int(np.count_nonzero(cv2.bitwise_and(dark, bright_closed)))
    n_chroma_enclosed = int(np.count_nonzero(cv2.bitwise_and(text_chroma, bright_closed)))
    has_strong_dark_text = (n_dark >= 40 and n_dark_enclosed >= max(30, int(n_chroma_enclosed * 0.45)))

    if n_chroma > n_dark and n_chroma >= 12 and not has_strong_dark_text:
        # Chromatic-core stroked text (e.g. red SFX letters with white outline on pink glow):
        # Do NOT include dark background pixels outside the glow so inpainting fills with pure glow color
        coupled_bright = cv2.bitwise_and(bright, cv2.dilate(text_chroma, k_couple))
        core_seeds = cv2.bitwise_or(coupled_chroma, coupled_bright)
        seed_zone = cv2.dilate(core_seeds, k_seed)
        aa_bright = (gray > 115).astype(np.uint8) * 255
        # Catch thin anti-aliased chromatic punctuation (e.g. quotes) near text lines
        line_r = max(12, min(48, int(round(12.0 * s))))
        k_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_r * 2 + 1, line_r * 2 + 1))
        aa_chroma = (
            (sat > 42) & (max_c > 125) & (bright_dil > 0) & (cv2.dilate(core_seeds, k_line) > 0)
        ).astype(np.uint8) * 255
        stroke_mask = cv2.bitwise_or(text_chroma, cv2.bitwise_or(aa_chroma, cv2.bitwise_and(aa_bright, seed_zone)))
        halo_r = max(3, min(18, int(round(4.0 * s))))
        k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_r * 2 + 1, halo_r * 2 + 1))
        return cv2.dilate(stroke_mask, k_halo, iterations=1)
    else:
        # Dark-core with white-outline (or white-core with dark-outline, or mixed) text:
        coupled_bright = cv2.bitwise_and(bright, cv2.dilate(cv2.bitwise_or(dark, text_chroma), k_couple))
        core_seeds = cv2.bitwise_or(coupled_dark, cv2.bitwise_or(coupled_bright, coupled_chroma))
        seed_zone = cv2.dilate(core_seeds, k_seed)
        aa_bright = (gray > 95).astype(np.uint8) * 255
        aa_dark = (gray < 88).astype(np.uint8) * 255
        all_candidates = cv2.bitwise_or(aa_dark, cv2.bitwise_or(aa_bright, text_chroma))
        stroke_mask = cv2.bitwise_and(all_candidates, seed_zone)
        halo_r = max(4, min(18, int(round(4.5 * s))))
        k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_r * 2 + 1, halo_r * 2 + 1))
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


def _classify_color_family(hex_color: str) -> str:
    """Classifies a #RRGGBB hex string into a perceptual color family for turn grouping."""
    if not hex_color or len(hex_color) < 7 or not hex_color.startswith("#"):
        return "achromatic"
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except ValueError:
        return "achromatic"

    max_c = max(r, g, b)
    min_c = min(r, g, b)
    sat = max_c - min_c
    if sat < 40:
        return "light" if max_c > 128 else "dark"

    if r - max(g, b) >= 25:
        return "red"
    if r > 140 and g > 110 and min(r, g) - b >= 35:
        return "yellow"
    if b - g >= 25 and r - g >= 20:
        return "purple"
    if b - r >= 25 and b - g >= 10:
        return "blue"
    if min(g, b) - r >= 25 and (g - b) < 35:
        return "cyan"
    if g - r >= 20 and g >= b - 9:
        return "green"
    return "other"


def _extract_dominant_chroma_hex(
    sample: np.ndarray,
    chroma_mask: np.ndarray,
    sat: np.ndarray,
    has_bright_outline: bool = True
) -> str:
    """
    Extracts the dominant chromatic glyph color from chroma_mask by clustering pixels into
    hue families so multi-colored crops never average complementary hues into gray,
    and de-bleeds JPEG white-outline anti-aliasing when present.
    """
    b_vals = sample[:, :, 0][chroma_mask].astype(np.int32)
    g_vals = sample[:, :, 1][chroma_mask].astype(np.int32)
    r_vals = sample[:, :, 2][chroma_mask].astype(np.int32)
    s_vals = sat[chroma_mask].astype(np.int32)

    red_m = (r_vals - np.maximum(g_vals, b_vals)) >= 25
    yellow_m = (r_vals > 140) & (g_vals > 110) & (np.minimum(r_vals, g_vals) - b_vals >= 35) & (~red_m)
    purple_m = (b_vals - g_vals >= 25) & (r_vals - g_vals >= 20) & (~red_m)
    blue_m = (b_vals - r_vals >= 25) & (b_vals - g_vals >= 10) & (~purple_m)
    cyan_m = (np.minimum(g_vals, b_vals) - r_vals >= 25) & ((g_vals - b_vals) < 35) & (b_vals - g_vals < 10) & (~yellow_m) & (~purple_m)
    green_m = (g_vals - r_vals >= 20) & (g_vals >= b_vals - 9) & (~yellow_m) & (~cyan_m)

    families = [
        ("red", red_m),
        ("cyan", cyan_m),
        ("green", green_m),
        ("blue", blue_m),
        ("yellow", yellow_m),
        ("purple", purple_m),
    ]

    best_name = "other"
    best_mask = np.ones_like(s_vals, dtype=bool)
    best_weight = -1
    for name, fmask in families:
        if np.any(fmask):
            weight = int(np.sum(s_vals[fmask]))
            if weight > best_weight:
                best_weight = weight
                best_name = name
                best_mask = fmask

    fam_r = r_vals[best_mask]
    fam_g = g_vals[best_mask]
    fam_b = b_vals[best_mask]
    fam_s = s_vals[best_mask]

    if len(fam_s) > 4:
        thr = np.percentile(fam_s, 70)
        vivid = fam_s >= thr
        r_med = int(np.median(fam_r[vivid]))
        g_med = int(np.median(fam_g[vivid]))
        b_med = int(np.median(fam_b[vivid]))
    else:
        r_med = int(np.median(fam_r))
        g_med = int(np.median(fam_g))
        b_med = int(np.median(fam_b))

    min_med = min(r_med, g_med, b_med)
    if has_bright_outline and min_med >= 35:
        if best_name == "red":
            r_med = min(242, max(205, int(r_med * 1.06)))
            g_med = max(28, min(72, int(g_med * 0.72)))
            b_med = max(28, min(72, int(b_med * 0.72)))
        elif best_name == "cyan":
            r_med = max(14, min(52, int(r_med * 0.65)))
            g_med = min(215, max(172, int(g_med * 1.04)))
            b_med = min(215, max(168, int(b_med * 1.04)))
        elif best_name == "green":
            r_med = max(24, min(68, int(r_med * 0.62)))
            g_med = min(228, max(182, int(g_med * 1.06)))
            b_med = max(96, min(142, int(b_med * 0.72)))
        elif best_name == "blue":
            r_med = max(28, min(72, int(r_med * 0.68)))
            g_med = max(150, min(196, int(g_med * 0.96)))
            b_med = min(246, max(215, int(b_med * 1.08)))

    return f"#{r_med:02X}{g_med:02X}{b_med:02X}"


def analyze_text_color(crop: np.ndarray, bg_bgr: Tuple[int, int, int], dist_thresh: int = 60) -> str:
    if crop is None or crop.size == 0:
        return "#000000"

    h, w = crop.shape[:2]
    sample = crop
    if h > 360 and w > 540:
        sample = cv2.resize(crop, (min(w, 540), min(h, 360)), interpolation=cv2.INTER_AREA)

    if sample.ndim == 3:
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        bright_stroke = (gray > 175).astype(np.uint8)
        dark_stroke = (gray < 52).astype(np.uint8)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        near_bright = cv2.dilate(bright_stroke, k3) > 0
        near_dark = cv2.dilate(dark_stroke, k3) > 0
        near_stroke = near_bright | near_dark

        s_int = sample.astype(np.int32)
        max_c = np.max(s_int, axis=2)
        min_c = np.min(s_int, axis=2)
        sat = max_c - min_c

        bg_arr_3d = np.array(bg_bgr, dtype=np.int32).reshape(1, 1, 3)
        bg_diff = np.max(np.abs(s_int - bg_arr_3d), axis=2)

        chroma_glyph = (sat >= 50) & (max_c >= 120) & near_stroke & (bg_diff > 35)
        black_glyph = (gray < 45) & near_bright
        n_chroma = int(np.sum(chroma_glyph))
        n_black = int(np.sum(black_glyph))

        sample_area = sample.shape[0] * sample.shape[1]
        min_chroma_px = max(12, int(sample_area * 0.005))
        if n_chroma >= min_chroma_px and n_chroma > n_black * 0.65:
            has_bright = int(np.sum(bright_stroke)) >= 8
            return _extract_dominant_chroma_hex(sample, chroma_glyph, sat, has_bright_outline=has_bright)

        if n_black >= max(15, int(sample_area * 0.015)) and n_black >= n_chroma:
            bg_lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
            return "#000000" if bg_lum > 127 else "#FFFFFF"

    pixels = sample.reshape(-1, 3).astype(np.int32)
    bg_arr = np.array(bg_bgr, dtype=np.int32)
    dists = np.sum(np.abs(pixels - bg_arr), axis=1)
    text_mask = dists > dist_thresh

    if np.sum(text_mask) > 8:
        fg_pixels = pixels[text_mask]
        fg_max = np.max(fg_pixels, axis=1)
        fg_min = np.min(fg_pixels, axis=1)
        fg_sat = fg_max - fg_min
        chroma_fg = (fg_sat >= 65) & (fg_max >= 125)
        if np.sum(chroma_fg) >= max(8, int(len(fg_pixels) * 0.25)):
            fake_sample = fg_pixels[chroma_fg].reshape(-1, 1, 3)
            fake_mask = np.ones((fake_sample.shape[0], 1), dtype=bool)
            fake_sat = fg_sat[chroma_fg].reshape(-1, 1)
            return _extract_dominant_chroma_hex(fake_sample, fake_mask, fake_sat, has_bright_outline=False)

        avg_bgr = np.mean(fg_pixels, axis=0).astype(int)
        b, g, r = int(avg_bgr[0]), int(avg_bgr[1]), int(avg_bgr[2])
        if max(r, g, b) - min(r, g, b) < 35:
            bg_lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
            return "#000000" if bg_lum > 127 else "#FFFFFF"
        return f"#{r:02X}{g:02X}{b:02X}"
    else:
        bg_lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
        return "#000000" if bg_lum > 127 else "#FFFFFF"


def extract_line_text_colors(crop: np.ndarray, bg_bgr: Tuple[int, int, int]) -> List[str]:
    """
    Detects horizontal visual text rows inside a block crop and extracts the text color
    for each individual row from top to bottom.
    """
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return []

    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return [analyze_text_color(crop, bg_bgr)]

    if w > 720 or h > 1080:
        scale = min(720.0 / w, 1080.0 / h)
        crop = cv2.resize(
            crop,
            (max(12, int(round(w * scale))), max(12, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA
        )
        h, w = crop.shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright = (gray > 175).astype(np.uint8)
    dark = (gray < 52).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near_bright = cv2.dilate(bright, k3) > 0
    near_dark = cv2.dilate(dark, k3) > 0

    c_int = crop.astype(np.int32)
    sat = np.max(c_int, axis=2) - np.min(c_int, axis=2)
    max_c = np.max(c_int, axis=2)

    bg_arr_3d = np.array(bg_bgr, dtype=np.int32).reshape(1, 1, 3)
    bg_diff = np.max(np.abs(c_int - bg_arr_3d), axis=2)

    chroma_px = (sat >= 50) & (max_c >= 120) & (near_bright | near_dark) & (bg_diff > 35)
    stroke_px = chroma_px | ((gray < 45) & near_bright)

    row_counts = np.sum(chroma_px, axis=1)
    min_row_px = max(4, int(w * 0.015))
    using_chroma = int(np.sum(row_counts >= min_row_px)) >= 6
    if not using_chroma:
        row_counts = np.sum(stroke_px, axis=1)

    rows: List[Tuple[int, int]] = []
    in_row = False
    start_y = 0
    for y_i, cnt in enumerate(row_counts):
        if cnt >= min_row_px and not in_row:
            in_row = True
            start_y = y_i
        elif cnt < min_row_px and in_row:
            in_row = False
            if y_i - start_y >= 5:
                rows.append((max(0, start_y - 1), min(h, y_i + 1)))
    if in_row and h - start_y >= 5:
        rows.append((max(0, start_y - 1), h))

    # Reject isolated background specks (e.g. small LED icons / skin highlights between lines)
    if using_chroma and len(rows) >= 3:
        row_peaks = [int(np.max(row_counts[ry1:ry2])) for ry1, ry2 in rows]
        row_totals = [int(np.sum(row_counts[ry1:ry2])) for ry1, ry2 in rows]
        ref_peak = float(np.percentile(row_peaks, 75))
        ref_total = float(np.percentile(row_totals, 75))
        filtered = [
            r for r, pk, tot in zip(rows, row_peaks, row_totals)
            if not (tot < ref_total * 0.16 and pk < ref_peak * 0.16)
        ]
        if filtered:
            rows = filtered

    if not rows:
        return [analyze_text_color(crop, bg_bgr)]

    colors: List[str] = []
    for ry1, ry2 in rows:
        row_crop = crop[ry1:ry2, :]
        colors.append(analyze_text_color(row_crop, bg_bgr))
    return colors


def collapse_color_turns(row_colors: List[str]) -> List[str]:
    """
    Collapses consecutive visual row colors belonging to the same perceptual color family
    into speaker-turn colors (e.g. a 3-row green paragraph followed by a 1-row red paragraph -> [green, red]).
    """
    if not row_colors:
        return []
    turns: List[str] = [row_colors[0]]
    prev_fam = _classify_color_family(row_colors[0])
    for c in row_colors[1:]:
        fam = _classify_color_family(c)
        if fam != prev_fam:
            turns.append(c)
            prev_fam = fam
    return turns


def refresh_blocks_text_colors(original_img: Optional[np.ndarray], blocks: Optional[List[Any]]) -> None:
    """
    Refreshes text_color and line_colors for each block directly from original_img.
    Repairs stale cached blocks (e.g. legacy #3CE8F5 values) and populates multi-speaker line_colors.
    """
    if original_img is None or not hasattr(original_img, "shape") or original_img.size == 0 or not blocks:
        return

    h_img, w_img = original_img.shape[:2]
    if h_img <= 0 or w_img <= 0:
        return

    for b in blocks:
        is_dict = isinstance(b, dict)
        xmin_pct = float(b.get("xmin", 0.0) if is_dict else getattr(b, "xmin", 0.0))
        ymin_pct = float(b.get("ymin", 0.0) if is_dict else getattr(b, "ymin", 0.0))
        xmax_pct = float(b.get("xmax", 0.0) if is_dict else getattr(b, "xmax", 0.0))
        ymax_pct = float(b.get("ymax", 0.0) if is_dict else getattr(b, "ymax", 0.0))

        x1 = max(0, min(w_img - 1, int(round((xmin_pct / 100.0) * w_img))))
        y1 = max(0, min(h_img - 1, int(round((ymin_pct / 100.0) * h_img))))
        x2 = max(x1 + 1, min(w_img, int(round((xmax_pct / 100.0) * w_img))))
        y2 = max(y1 + 1, min(h_img, int(round((ymax_pct / 100.0) * h_img))))

        cur_tc = str((b.get("text_color") if is_dict else getattr(b, "text_color", None)) or "").strip()
        cur_lc = b.get("line_colors") if is_dict else getattr(b, "line_colors", None)
        is_legacy_cyan = (cur_tc.upper() == "#3CE8F5")

        # Fast path: if block already has valid non-legacy text_color and populated line_colors, skip heavy CV analysis
        if cur_tc and not is_legacy_cyan and isinstance(cur_lc, list) and len(cur_lc) > 0:
            continue

        crop = original_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        bg_r, bg_g, bg_b = get_background_color_rgb(crop)
        bg_bgr = (bg_b, bg_g, bg_r)

        line_cols = extract_line_text_colors(crop, bg_bgr)
        dom_col = analyze_text_color(crop, bg_bgr)

        # Update line_colors and repair legacy #3CE8F5 or multi-line chromatic colors
        has_multi_chroma = len({ _classify_color_family(c) for c in line_cols }) > 1
        is_legacy_cyan = cur_tc.upper() == "#3CE8F5"
        is_default_bw_with_chroma = (
            cur_tc.upper() in ("#000000", "#FFFFFF")
            and _classify_color_family(dom_col) not in ("black", "white")
        )
        should_update_tc = (not cur_tc) or is_legacy_cyan or has_multi_chroma or is_default_bw_with_chroma
        if should_update_tc or not cur_lc:
            eff_lc = line_cols if (should_update_tc or dom_col.upper() == cur_tc.upper()) else [cur_tc]
            if is_dict:
                if should_update_tc:
                    b["text_color"] = dom_col
                b["line_colors"] = eff_lc
            else:
                if should_update_tc:
                    setattr(b, "text_color", dom_col)
                setattr(b, "line_colors", eff_lc)


def dilate_mask(mask: np.ndarray, dilation_pixels: int = 3) -> np.ndarray:
    if dilation_pixels <= 0 or mask is None:
        return mask
    kernel_size = dilation_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)
