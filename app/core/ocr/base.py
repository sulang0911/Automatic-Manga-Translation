"""
app/core/ocr/base.py
Abstract base class and utility functions for OCR detection and recognition engines.
"""
import math
import cv2
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from app.core.models import TranslationBlock
from app.core.inpaint.color_analyzer import get_background_color_hex


class BaseOCREngine(ABC):
    """
    Abstract interface for manga text detection and recognition backends.
    """

    @abstractmethod
    def detect_and_recognize(
        self,
        image: np.ndarray,
        lang: str = "japan",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[TranslationBlock]:
        """
        Executes text detection and recognition on an OpenCV image (BGR).
        Returns a list of TranslationBlock objects with normalized coordinates [0.0, 100.0].
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if dependencies and weights for this engine are loaded."""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Returns backend engine details, device string, and hardware info."""
        pass


def is_solid_color_page(image: np.ndarray, std_thresh: float = 8.0) -> bool:
    """
    Fast bypass detector: returns True if image has near-zero pixel variance (e.g. blank spacer page).
    """
    if image is None or image.size == 0:
        return True
    std = float(np.std(image))
    return std < std_thresh


def _is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or      # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or      # CJK Extension A
        0x20000 <= cp <= 0x2FA1F or    # CJK Extension B-F & Compatibility Ideographs
        0x3000 <= cp <= 0x303F or      # CJK Symbols and Punctuation (、。〈〉《》「」『』【】〜・ etc.)
        0x3040 <= cp <= 0x309F or      # Hiragana
        0x30A0 <= cp <= 0x30FF or      # Katakana
        0x31F0 <= cp <= 0x31FF or      # Katakana Phonetic Extensions
        0xAC00 <= cp <= 0xD7AF or      # Hangul Syllables
        0x1100 <= cp <= 0x11FF or      # Hangul Jamo
        0xFF00 <= cp <= 0xFFEF         # Halfwidth and Fullwidth Forms
    )


def _join_line_texts(texts: List[str]) -> str:
    cleaned = [t.strip() for t in texts if t and str(t).strip()]
    if not cleaned:
        return ""
    res = cleaned[0]
    for nxt in cleaned[1:]:
        if _is_cjk_char(res[-1]) or _is_cjk_char(nxt[0]):
            res += nxt
        else:
            res += " " + nxt
    return res


def calculate_polygon_angle(pts: Any, threshold: float = 15.0) -> float:
    """
    Calculates text line orientation angle in degrees in [-90.0, +90.0].
    Uses baseline vector arctan2 for 4-point quadrilaterals and
    oriented cv2.minAreaRect fallback for general polygons.
    Applies deadband threshold (|deg| < threshold returns 0.0, default 15.0).
    """
    if pts is None:
        return 0.0
    pts_arr = np.asarray(pts, dtype=np.float32)
    if len(pts_arr) < 2:
        return 0.0

    if len(pts_arr) == 4:
        v_top = pts_arr[1] - pts_arr[0]
        v_bot = pts_arr[2] - pts_arr[3]
        dx = float(v_top[0] + v_bot[0]) / 2.0
        dy = float(v_top[1] + v_bot[1]) / 2.0
    else:
        rect = cv2.minAreaRect(pts_arr)
        box = cv2.boxPoints(rect)
        e1 = box[1] - box[0]
        e2 = box[2] - box[1]
        long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        if long_e[0] < 0:
            long_e = -long_e
        dx, dy = float(long_e[0]), float(long_e[1])

    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return 0.0

    deg = math.degrees(math.atan2(dy, dx))
    while deg > 90.0:
        deg -= 180.0
    while deg < -90.0:
        deg += 180.0

    return round(deg, 1) if abs(deg) >= threshold else 0.0


def order_polygon_vertices(pts: Any, angle_deg: float) -> List[List[int]]:
    """
    Canonically orders 4 vertices [TL, TR, BR, BL] aligned with angle_deg:
    - TL -> TR: baseline vector along angle_deg (reading direction)
    - TR -> BR: height vector downwards (angle_deg + 90 deg)
    - BR -> BL: baseline vector reversed
    - BL -> TL: height vector upwards
    """
    if pts is None:
        return []
    pts_arr = np.asarray(pts, dtype=np.float32)
    if len(pts_arr) != 4:
        return [[int(round(p[0])), int(round(p[1]))] for p in pts_arr]

    center = np.mean(pts_arr, axis=0)
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    u = np.array([cos_a, sin_a], dtype=np.float32)
    v = np.array([-sin_a, cos_a], dtype=np.float32)

    rel = pts_arr - center
    proj_u = rel @ u
    proj_v = rel @ v

    tl_idx = int(np.argmin(proj_u + proj_v))
    tr_idx = int(np.argmax(proj_u - proj_v))
    br_idx = int(np.argmax(proj_u + proj_v))
    bl_idx = int(np.argmin(proj_u - proj_v))

    indices = [tl_idx, tr_idx, br_idx, bl_idx]
    if len(set(indices)) == 4:
        ordered = pts_arr[indices]
    else:
        tl = int(min(range(4), key=lambda i: (proj_u[i] + proj_v[i])))
        br = int(max(range(4), key=lambda i: (proj_u[i] + proj_v[i])))
        rem = [i for i in range(4) if i not in (tl, br)]
        if proj_u[rem[0]] - proj_v[rem[0]] > proj_u[rem[1]] - proj_v[rem[1]]:
            tr, bl = rem[0], rem[1]
        else:
            tr, bl = rem[1], rem[0]
        ordered = pts_arr[[tl, tr, br, bl]]

    return [[int(round(p[0])), int(round(p[1]))] for p in ordered]


def estimate_line_height(box: Dict[str, Any], has_cjk: bool) -> float:
    """
    Estimates the effective line height of a text box.
    If explicit line_count or splitlines() > 1 exists, derives line_h from that.
    Otherwise, for horizontal multi-word paragraph boxes, estimates line count to prevent
    the line height from exploding to the full paragraph height and falsely inflating merge gaps.
    """
    h = max(1.0, float(max(box["ymin"], box["ymax"]) - min(box["ymin"], box["ymax"])))
    w = max(1.0, float(max(box["xmin"], box["xmax"]) - min(box["xmin"], box["xmax"])))

    explicit_lines = box.get("line_count")
    if explicit_lines and explicit_lines > 1:
        return h / float(explicit_lines)

    text = str(box.get("text", "")).strip()
    split_lines = text.splitlines()
    if len(split_lines) > 1:
        return h / float(len(split_lines))

    if not has_cjk and text:
        words = text.split()
        num_words = len(words)
        if num_words >= 4 or (h >= 50.0 and h >= w * 0.40):
            avg_char_w = max(6.0, min(16.0, w / 15.0))
            chars_per_line = max(8.0, w / avg_char_w)
            est_lines = max(1.0, math.ceil(len(text) / chars_per_line))
            clamped_lines = max(est_lines, h / 45.0)
            return h / float(clamped_lines)

    return h


def compute_bubble_labels(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Computes connected-component labels for continuous light speech bubble closures.
    Uses adaptive binary thresholding, morphological ellipse closing, and cv2.connectedComponentsWithStats.
    Excludes large page backgrounds and margins spanning across the outer borders of the page.
    Returns an integer label matrix of shape (H, W), or None if image is None or invalid.
    """
    if image is None or not hasattr(image, "size") or image.size == 0:
        return None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        # In manga, dialogue bubbles typically have white/light interior (>= 215)
        # surrounded by dark outlines.
        _, binary = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        # Morphological closing with ellipse kernel bridges text strokes inside the bubble
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
        if num_labels <= 1:
            return labels

        total_pixels = float(labels.shape[0] * labels.shape[1])
        H_lbl, W_lbl = labels.shape[:2]

        # In manga, page margins, white panel backgrounds, and inter-panel gutters
        # often form giant connected components spanning across the entire page.
        # A single dialogue bubble never spans the entire page or covers > 35% of the page area.
        for lbl_idx in range(1, num_labels):
            area = stats[lbl_idx, cv2.CC_STAT_AREA]
            area_ratio = area / total_pixels
            left = stats[lbl_idx, cv2.CC_STAT_LEFT]
            top = stats[lbl_idx, cv2.CC_STAT_TOP]
            width = stats[lbl_idx, cv2.CC_STAT_WIDTH]
            height = stats[lbl_idx, cv2.CC_STAT_HEIGHT]

            touches = [
                left <= 2,
                top <= 2,
                left + width >= W_lbl - 2,
                top + height >= H_lbl - 2
            ]
            border_touch_count = sum(touches)

            is_page_bg = False
            if border_touch_count >= 3 and area_ratio > 0.10:
                is_page_bg = True
            elif border_touch_count >= 2 and area_ratio > 0.15:
                is_page_bg = True
            elif border_touch_count >= 1 and area_ratio > 0.45:
                is_page_bg = True

            if is_page_bg:
                labels[labels == lbl_idx] = 0

        return labels
    except Exception:
        return None



def can_merge_pair(
    b1: Dict[str, Any],
    b2: Dict[str, Any],
    img_w: int,
    img_h: int,
    v_thresh_ratio: float = 0.025,
    h_thresh_ratio: float = 0.035,
    image: Optional[np.ndarray] = None,
    bubble_labels: Optional[np.ndarray] = None,
    adaptive_spacing: bool = False,
    qr_regions: Optional[List[Any]] = None
) -> bool:
    """
    Evaluates whether two detected text boxes belong to the same dialogue bubble / text region.
    Prevents merging adjacent independent speech bubbles or cross-QR bridges while correctly
    aggregating multi-line text and same-line / same-column fragments with adaptive spacing.
    """
    # Angle compatibility guard: if both b1 and b2 have an angle specified, reject if |a1 - a2| > 10.0 degrees
    if "angle" in b1 and "angle" in b2 and b1["angle"] is not None and b2["angle"] is not None:
        try:
            a1 = float(b1["angle"])
            a2 = float(b2["angle"])
            ang_diff = abs(a1 - a2)
            if ang_diff > 90.0:
                ang_diff = abs(ang_diff - 180.0)
            if ang_diff > 10.0:
                return False
            # Off-bubble slanted text guard: reject if one is clearly slanted and the other is horizontal dialogue
            if min(abs(a1), abs(a2)) < 3.0 and max(abs(a1), abs(a2)) >= 8.0 and ang_diff >= 8.0:
                return False
        except (ValueError, TypeError):
            pass

    # Visual background color difference guard: reject if high contrast between candidate boxes
    c1 = b1.get("bg_color")
    c2 = b2.get("bg_color")
    if image is not None and (c1 is None or c2 is None):
        try:
            bx1 = max(0, min(int(round(b1["xmin"])), img_w - 1))
            by1 = max(0, min(int(round(b1["ymin"])), img_h - 1))
            bx2 = max(bx1 + 1, min(int(round(b1["xmax"])), img_w))
            by2 = max(by1 + 1, min(int(round(b1["ymax"])), img_h))
            if c1 is None and by2 > by1 and bx2 > bx1:
                c1 = get_background_color_hex(image[by1:by2, bx1:bx2])
                b1["bg_color"] = c1

            bx1_2 = max(0, min(int(round(b2["xmin"])), img_w - 1))
            by1_2 = max(0, min(int(round(b2["ymin"])), img_h - 1))
            bx2_2 = max(bx1_2 + 1, min(int(round(b2["xmax"])), img_w))
            by2_2 = max(by1_2 + 1, min(int(round(b2["ymax"])), img_h))
            if c2 is None and by2_2 > by1_2 and bx2_2 > bx1_2:
                c2 = get_background_color_hex(image[by1_2:by2_2, bx1_2:bx2_2])
                b2["bg_color"] = c2
        except Exception:
            pass

    if c1 and c2:
        try:
            def _parse_color(c):
                if isinstance(c, str):
                    c = c.lstrip("#")
                    if len(c) == 6:
                        return tuple(int(c[k:k+2], 16) for k in (0, 2, 4))
                elif isinstance(c, (list, tuple)) and len(c) >= 3:
                    return tuple(int(x) for x in c[:3])
                return None
            rgb1 = _parse_color(c1)
            rgb2 = _parse_color(c2)
            if rgb1 and rgb2:
                dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(rgb1, rgb2)))
                if dist > 60.0:
                    return False
        except Exception:
            pass

    x1_min = int(round(min(b1["xmin"], b1["xmax"])))
    x1_max = int(round(max(b1["xmin"], b1["xmax"])))
    y1_min = int(round(min(b1["ymin"], b1["ymax"])))
    y1_max = int(round(max(b1["ymin"], b1["ymax"])))

    x2_min = int(round(min(b2["xmin"], b2["xmax"])))
    x2_max = int(round(max(b2["xmin"], b2["xmax"])))
    y2_min = int(round(min(b2["ymin"], b2["ymax"])))
    y2_max = int(round(max(b2["ymin"], b2["ymax"])))

    w1, h1 = max(1, x1_max - x1_min), max(1, y1_max - y1_min)
    w2, h2 = max(1, x2_max - x2_min), max(1, y2_max - y2_min)

    min_w, max_w = min(w1, w2), max(w1, w2)
    min_h, max_h = min(h1, h2), max(h1, h2)

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    x_gap = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
    y_gap = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))

    x_overlap_ratio = x_overlap / min_w
    y_overlap_ratio = y_overlap / min_h

    mid_x1 = (x1_min + x1_max) / 2.0
    mid_x2 = (x2_min + x2_max) / 2.0
    mid_y1 = (y1_min + y1_max) / 2.0
    mid_y2 = (y2_min + y2_max) / 2.0

    # QR Barrier Guard: reject if combined bounding box spans or intersects any QR code
    if qr_regions is None and image is not None:
        try:
            from app.core.ocr.qr_filter import QRCodeFilter
            qr_regions = QRCodeFilter().detect_regions(image)
        except Exception:
            qr_regions = None

    if qr_regions:
        is_norm = (max(x1_max, x2_max) <= 100.0 and max(y1_max, y2_max) <= 100.0 and img_w > 100)
        if is_norm:
            c_xmin_px = (min(x1_min, x2_min) / 100.0) * img_w
            c_ymin_px = (min(y1_min, y2_min) / 100.0) * img_h
            c_xmax_px = (max(x1_max, x2_max) / 100.0) * img_w
            c_ymax_px = (max(y1_max, y2_max) / 100.0) * img_h
        else:
            c_xmin_px = min(x1_min, x2_min)
            c_ymin_px = min(y1_min, y2_min)
            c_xmax_px = max(x1_max, x2_max)
            c_ymax_px = max(y1_max, y2_max)

        for reg in qr_regions:
            if hasattr(reg, "bbox"):
                qx1, qy1, qx2, qy2 = reg.bbox
            elif isinstance(reg, dict) and "bbox" in reg:
                qx1, qy1, qx2, qy2 = reg["bbox"]
            elif isinstance(reg, dict) and "xmin" in reg:
                qx1, qy1, qx2, qy2 = reg["xmin"], reg["ymin"], reg["xmax"], reg["ymax"]
            elif isinstance(reg, (tuple, list)) and len(reg) == 4:
                qx1, qy1, qx2, qy2 = reg
            else:
                continue

            pad = 8
            inter_w = max(0, min(c_xmax_px, qx2 + pad) - max(c_xmin_px, qx1 - pad))
            inter_h = max(0, min(c_ymax_px, qy2 + pad) - max(c_ymin_px, qy1 - pad))
            if inter_w > 0 and inter_h > 0:
                return False

    # Connected component / Bubble Closure Awareness
    if bubble_labels is None and image is not None:
        bubble_labels = compute_bubble_labels(image)

    same_bubble = False
    if bubble_labels is not None:
        H_lbl, W_lbl = bubble_labels.shape[:2]
        is_norm = (max(x1_max, x2_max) <= 100.0 and max(y1_max, y2_max) <= 100.0 and img_w > 100)

        def _sample_label(xmin_c, ymin_c, xmax_c, ymax_c, mid_x_c, mid_y_c):
            if is_norm:
                px_x = int(round((mid_x_c / 100.0) * W_lbl))
                px_y = int(round((mid_y_c / 100.0) * H_lbl))
                x1_p = int(round((xmin_c / 100.0) * W_lbl))
                y1_p = int(round((ymin_c / 100.0) * H_lbl))
                x2_p = int(round((xmax_c / 100.0) * W_lbl))
                y2_p = int(round((ymax_c / 100.0) * H_lbl))
            else:
                px_x = int(round(mid_x_c))
                px_y = int(round(mid_y_c))
                x1_p = int(round(xmin_c))
                y1_p = int(round(ymin_c))
                x2_p = int(round(xmax_c))
                y2_p = int(round(ymax_c))

            px_x = max(0, min(W_lbl - 1, px_x))
            px_y = max(0, min(H_lbl - 1, px_y))
            lbl = int(bubble_labels[px_y, px_x])
            if lbl == 0:
                # Text stroke fallback: sample non-zero mode inside box interior
                x1_cl = max(0, min(W_lbl - 1, x1_p))
                y1_cl = max(0, min(H_lbl - 1, y1_p))
                x2_cl = max(x1_cl + 1, min(W_lbl, x2_p))
                y2_cl = max(y1_cl + 1, min(H_lbl, y2_p))
                patch = bubble_labels[y1_cl:y2_cl, x1_cl:x2_cl]
                pos = patch[patch > 0]
                if pos.size > 0:
                    lbl = int(np.bincount(pos).argmax())
            return lbl

        lbl1 = _sample_label(x1_min, y1_min, x1_max, y1_max, mid_x1, mid_y1)
        lbl2 = _sample_label(x2_min, y2_min, x2_max, y2_max, mid_x2, mid_y2)

        if lbl1 > 0 and lbl2 > 0:
            if lbl1 == lbl2:
                same_bubble = True
            else:
                # Distinct bubbles separated by borders: strictly reject merging
                return False

    # Sensitivity scales derived from threshold ratios
    v_scale = max(0.2, float(v_thresh_ratio) / 0.025)
    h_scale = max(0.2, float(h_thresh_ratio) / 0.035)

    def _is_cjk_block(t: str) -> bool:
        if not t:
            return False
        cjk_cnt = sum(1 for c in t if _is_cjk_char(c))
        if cjk_cnt == 0:
            return False
        latin_cnt = sum(1 for c in t if ('a' <= c <= 'z' or 'A' <= c <= 'Z'))
        if latin_cnt > 0 and (cjk_cnt / float(cjk_cnt + latin_cnt)) < 0.35:
            return False
        return True

    has_cjk = _is_cjk_block(str(b1.get("text", ""))) or _is_cjk_block(str(b2.get("text", "")))

    # Criterion 1: Significant 2D overlap / Inclusion (>= 50% area of smaller box)
    inter_area = x_overlap * y_overlap
    min_area = min_w * min_h
    if min_area > 0 and (inter_area / min_area) >= 0.50:
        return True

    # Criterion 2: Same-line fragments (horizontal text collinear fragments / words)
    is_not_vert_columns = not (has_cjk and h1 >= w1 * 1.20 and h2 >= w2 * 1.20)
    if is_not_vert_columns:
        a1 = float(b1.get("angle", 0.0) or 0.0)
        a2 = float(b2.get("angle", 0.0) or 0.0)
        eff_ang = a1 if abs(a1) >= 15.0 else a2
        if abs(eff_ang) >= 15.0:
            # Slanted collinearity check along slant line
            dx_mid = mid_x2 - mid_x1
            expected_y_diff = dx_mid * math.tan(math.radians(eff_ang))
            actual_y_diff = mid_y2 - mid_y1
            y_deviation = abs(actual_y_diff - expected_y_diff)
            slant_gap = math.hypot(x_gap, y_gap)
            max_word_gap = max(10, int(1.2 * max_h * h_scale))
            if y_deviation <= max_h * 0.50 and (slant_gap <= max_word_gap or x_gap <= max_word_gap):
                return True
        else:
            if (y_overlap_ratio >= 0.45 or abs(mid_y1 - mid_y2) <= max_h * 0.40) and (min_h / max_h) >= 0.40:
                max_word_gap = max(8, int((1.20 if same_bubble else 0.75) * max_h * h_scale))
                if x_gap <= max_word_gap:
                    return True

    # Criterion 3: Same-column fragments (vertical text collinear fragments, CJK only)
    if has_cjk:
        is_not_horiz_lines = not (w1 >= h1 * 1.35 and w2 >= h2 * 1.35)
        if is_not_horiz_lines and (min_w / max_w) >= 0.40:
            if (x_overlap_ratio >= 0.45 or abs(mid_x1 - mid_x2) <= max_w * 0.40):
                max_char_gap = max(8, int((1.20 if same_bubble else 0.85) * max_w * v_scale))
                if y_gap <= max_char_gap:
                    return True

    # Criterion 4: Horizontal multi-line text (lines stacked vertically within bubble/dialogue block)
    line_h1 = estimate_line_height(b1, has_cjk)
    line_h2 = estimate_line_height(b2, has_cjk)
    min_line_h = min(line_h1, line_h2)
    max_line_h = max(line_h1, line_h2)

    is_horiz_candidate = (not has_cjk) or (h1 <= w1 * 1.50 and h2 <= w2 * 1.50) or (
        max(w1, w2) >= max_line_h * 1.5
    )
    if is_horiz_candidate and (min_line_h / max_line_h) >= 0.30:
        if y1_min <= y2_min:
            t_ymin, t_ymax = y1_min, y1_max
            b_ymin, b_ymax = y2_min, y2_max
            t_line_h, b_line_h = line_h1, line_h2
        else:
            t_ymin, t_ymax = y2_min, y2_max
            b_ymin, b_ymax = y1_min, y1_max
            t_line_h, b_line_h = line_h2, line_h1

        if b_ymin >= t_ymin and b_ymax >= t_ymax:
            curr_y_gap = b_ymin - t_ymax
            est_line_h = (t_line_h + b_line_h) / 2.0
            if curr_y_gap < 0:
                y_overlap_len = -curr_y_gap
                y_gap_ok = y_overlap_len <= max(6, int((0.70 if same_bubble else 0.50) * est_line_h))
            else:
                strong_h_align = (x_overlap_ratio >= 0.70 and abs(mid_x1 - mid_x2) <= max_w * 0.25)
                is_coord_only_default = (image is None and bubble_labels is None and not adaptive_spacing)

                if not is_coord_only_default:
                    if same_bubble:
                        gap_mult = 2.2
                    elif adaptive_spacing:
                        gap_mult = 2.2 if strong_h_align else 1.8
                    else:
                        gap_mult = 0.70
                else:
                    gap_mult = 0.70

                max_line_gap = max(4, int(gap_mult * min_line_h * v_scale))
                y_gap_ok = curr_y_gap <= max_line_gap

            if y_gap_ok:
                x_aligned = (
                    (x_overlap_ratio >= 0.35 and abs(mid_x1 - mid_x2) <= max_w * 0.65)
                    or (x_overlap > 0 and (x_overlap / max_w >= 0.30))
                    or (abs(x1_min - x2_min) <= max(12, int(0.20 * max_w)) and x_overlap > 0)
                    or (abs(x1_max - x2_max) <= max(12, int(0.20 * max_w)) and x_overlap > 0)
                    or (abs(mid_x1 - mid_x2) <= max(12, int(0.20 * max_w)) and x_overlap > 0)
                    or (same_bubble and (x_overlap > 0 or abs(mid_x1 - mid_x2) <= max_w * 0.50))
                )
                if x_aligned:
                    return True

    # Criterion 5: Vertical multi-column text (columns side-by-side in same bubble, CJK only)
    if has_cjk:
        cols1 = max(1, len(str(b1.get("text", "")).splitlines()))
        cols2 = max(1, len(str(b2.get("text", "")).splitlines()))
        col_w1 = w1 / float(cols1)
        col_w2 = w2 / float(cols2)
        min_col_w = min(col_w1, col_w2)
        max_col_w = max(col_w1, col_w2)

        is_vert_candidate = (
            not (w1 >= h1 * 1.50 and w2 >= h2 * 1.50)
            and (min_col_w / max_col_w) >= 0.30
            and max(h1, h2) >= max_col_w * 0.8
        )
        if is_vert_candidate:
            if x1_min >= x2_min:
                r_xmin, r_xmax = x1_min, x1_max
                l_xmin, l_xmax = x2_min, x2_max
                r_col_w, l_col_w = col_w1, col_w2
            else:
                r_xmin, r_xmax = x2_min, x2_max
                l_xmin, l_xmax = x1_min, x1_max
                r_col_w, l_col_w = col_w2, col_w1

            if r_xmin >= l_xmin and r_xmax >= l_xmax:
                curr_x_gap = r_xmin - l_xmax
                est_col_w = (r_col_w + l_col_w) / 2.0
                if curr_x_gap < 0:
                    x_overlap_len = -curr_x_gap
                    x_gap_ok = x_overlap_len <= max(6, int((0.70 if same_bubble else 0.50) * est_col_w))
                else:
                    is_coord_only_default = (image is None and bubble_labels is None and not adaptive_spacing)
                    if not is_coord_only_default:
                        if same_bubble:
                            col_gap_mult = 2.0
                        elif adaptive_spacing:
                            col_gap_mult = 1.9
                        else:
                            col_gap_mult = 0.70
                    else:
                        col_gap_mult = 0.70

                    max_col_gap = max(4, int(col_gap_mult * min_col_w * h_scale))
                    x_gap_ok = curr_x_gap <= max_col_gap

                if x_gap_ok:
                    y_aligned = (
                        (y_overlap_ratio >= 0.35 and abs(mid_y1 - mid_y2) <= max_h * 0.65)
                        or (y_overlap > 0 and (y_overlap / max_h >= 0.30 or y_overlap / min_h >= 0.50))
                        or (y_overlap > 0 and abs(y1_min - y2_min) <= max_w * 2.0)
                        or (abs(y1_min - y2_min) <= max(12, int(0.20 * max_h)) and y_overlap > 0)
                        or (abs(y1_max - y2_max) <= max(12, int(0.20 * max_h)) and y_overlap > 0)
                        or (same_bubble and (y_overlap > 0 or abs(mid_y1 - mid_y2) <= max_h * 0.60))
                    )
                    if y_aligned:
                        return True

    return False


def merge_adjacent_boxes(
    raw_boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    v_thresh_ratio: float = 0.025,
    h_thresh_ratio: float = 0.035,
    image: Optional[np.ndarray] = None,
    adaptive_spacing: bool = False,
    qr_regions: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    """
    Merges closely adjacent or overlapping text lines within speech bubbles.
    Effectively prevents merging adjacent independent dialogue bubbles or cross-QR bridges
    while preserving multi-line text aggregation for both horizontal and vertical text layouts.
    """
    if not raw_boxes:
        return []

    img_w = max(1, int(img_w))
    img_h = max(1, int(img_h))

    bubble_labels = None
    if image is not None:
        bubble_labels = compute_bubble_labels(image)
        if qr_regions is None:
            try:
                from app.core.ocr.qr_filter import QRCodeFilter
                qr_regions = QRCodeFilter().detect_regions(image)
            except Exception:
                qr_regions = None

        for b in raw_boxes:
            if not b.get("bg_color"):
                bx1 = max(0, min(int(round(b["xmin"])), img_w - 1))
                by1 = max(0, min(int(round(b["ymin"])), img_h - 1))
                bx2 = max(bx1 + 1, min(int(round(b["xmax"])), img_w))
                by2 = max(by1 + 1, min(int(round(b["ymax"])), img_h))
                crop = image[by1:by2, bx1:bx2]
                if crop.size > 0:
                    b["bg_color"] = get_background_color_hex(crop)

    n = len(raw_boxes)
    if n == 1:
        b = raw_boxes[0]
        ang = float(b.get("angle", 0.0) or 0.0)
        bx1 = int(round(min(b["xmin"], b["xmax"])))
        by1 = int(round(min(b["ymin"], b["ymax"])))
        bx2 = int(round(max(b["xmin"], b["xmax"])))
        by2 = int(round(max(b["ymin"], b["ymax"])))
        poly = b.get("polygon")
        if poly is None:
            poly = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]
        return [{
            "xmin": bx1,
            "ymin": by1,
            "xmax": bx2,
            "ymax": by2,
            "text": str(b.get("text", "")).strip(),
            "conf": float(b.get("conf", 1.0) if b.get("conf") is not None else 1.0),
            "line_count": 1,
            "angle": ang,
            "polygon": poly,
            "bg_color": b.get("bg_color")
        }]

    # Build adjacency graph based on precise pairing criteria
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if can_merge_pair(
                raw_boxes[i],
                raw_boxes[j],
                img_w,
                img_h,
                v_thresh_ratio=v_thresh_ratio,
                h_thresh_ratio=h_thresh_ratio,
                image=image,
                bubble_labels=bubble_labels,
                adaptive_spacing=adaptive_spacing,
                qr_regions=qr_regions
            ):
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components (bubble clusters)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            component = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(component)

    # Preserve natural document reading order by earliest appearance
    clusters.sort(key=lambda comp: min(comp))

    merged = []
    for comp in clusters:
        comp_boxes = [raw_boxes[idx] for idx in comp]
        c_xmin = min(min(b["xmin"], b["xmax"]) for b in comp_boxes)
        c_ymin = min(min(b["ymin"], b["ymax"]) for b in comp_boxes)
        c_xmax = max(max(b["xmin"], b["xmax"]) for b in comp_boxes)
        c_ymax = max(max(b["ymin"], b["ymax"]) for b in comp_boxes)

        has_cjk = any(any(_is_cjk_char(c) for c in str(b.get("text", ""))) for b in comp_boxes)
        if not has_cjk:
            is_vertical_cluster = False
        else:
            cluster_w = max(1, c_xmax - c_xmin)
            cluster_h = max(1, c_ymax - c_ymin)
            if cluster_h >= cluster_w * 1.20:
                is_vertical_cluster = True
            elif cluster_w >= cluster_h * 1.20:
                is_vertical_cluster = False
            else:
                vert_boxes = sum(1 for b in comp_boxes if abs(b["ymax"] - b["ymin"]) >= abs(b["xmax"] - b["xmin"]) * 1.30 and abs(b["xmax"] - b["xmin"]) >= 14)
                horiz_boxes = sum(1 for b in comp_boxes if abs(b["xmax"] - b["xmin"]) >= abs(b["ymax"] - b["ymin"]) * 1.30)
                if vert_boxes > horiz_boxes:
                    is_vertical_cluster = True
                elif horiz_boxes > vert_boxes:
                    is_vertical_cluster = False
                else:
                    is_vertical_cluster = cluster_h > cluster_w

        confs = [float(b.get("conf", 1.0) if b.get("conf") is not None else 1.0) for b in comp_boxes]

        if is_vertical_cluster:
            # Group into vertical columns (Japanese text: columns from Right to Left)
            sorted_boxes = sorted(
                comp_boxes,
                key=lambda b: (-max(b["xmin"], b["xmax"]), min(b["ymin"], b["ymax"]))
            )
            columns: List[List[Dict[str, Any]]] = []
            for b in sorted_boxes:
                b_xmin = min(b["xmin"], b["xmax"])
                b_xmax = max(b["xmin"], b["xmax"])
                b_w = max(1, b_xmax - b_xmin)
                matched_col = None
                for col in columns:
                    col_xmin = min(min(x["xmin"], x["xmax"]) for x in col)
                    col_xmax = max(max(x["xmin"], x["xmax"]) for x in col)
                    col_w = max(1, col_xmax - col_xmin)
                    x_ov = max(0, min(b_xmax, col_xmax) - max(b_xmin, col_xmin))
                    if x_ov >= 0.40 * min(b_w, col_w):
                        matched_col = col
                        break
                if matched_col is not None:
                    matched_col.append(b)
                else:
                    columns.append([b])

            # Ensure columns are ordered strictly from Right to Left
            columns.sort(key=lambda col: -max(max(b["xmin"], b["xmax"]) for b in col))

            column_texts = []
            for col in columns:
                col_sorted = sorted(col, key=lambda b: min(b["ymin"], b["ymax"]))
                col_text = _join_line_texts([str(b.get("text", "")).strip() for b in col_sorted])
                if col_text:
                    column_texts.append(col_text)

            final_text = "\n".join(column_texts)
            line_count = len(column_texts) if column_texts else 1
        else:
            # Group into horizontal lines/rows (Top to Bottom)
            sorted_boxes = sorted(
                comp_boxes,
                key=lambda b: (min(b["ymin"], b["ymax"]), min(b["xmin"], b["xmax"]))
            )
            rows: List[List[Dict[str, Any]]] = []
            for b in sorted_boxes:
                b_ymin = min(b["ymin"], b["ymax"])
                b_ymax = max(b["ymin"], b["ymax"])
                b_h = max(1, b_ymax - b_ymin)
                matched_row = None
                for row in rows:
                    row_ymin = min(min(x["ymin"], x["ymax"]) for x in row)
                    row_ymax = max(max(x["ymin"], x["ymax"]) for x in row)
                    row_h = max(1, row_ymax - row_ymin)
                    y_ov = max(0, min(b_ymax, row_ymax) - max(b_ymin, row_ymin))
                    if y_ov >= 0.40 * min(b_h, row_h):
                        matched_row = row
                        break
                if matched_row is not None:
                    matched_row.append(b)
                else:
                    rows.append([b])

            # Ensure rows are ordered strictly from Top to Bottom
            rows.sort(key=lambda row: min(min(b["ymin"], b["ymax"]) for b in row))

            row_texts = []
            for row in rows:
                row_sorted = sorted(row, key=lambda b: min(b["xmin"], b["xmax"]))
                row_text = _join_line_texts([str(b.get("text", "")).strip() for b in row_sorted])
                if row_text:
                    row_texts.append(row_text)

            final_text = "\n".join(row_texts)
            line_count = len(row_texts) if row_texts else 1

        angles = [float(b.get("angle", 0.0)) for b in comp_boxes if "angle" in b and b.get("angle") is not None]
        avg_angle = float(np.median(angles)) if angles else 0.0

        all_pts = []
        for b in comp_boxes:
            if b.get("polygon"):
                all_pts.extend(b["polygon"])
            else:
                bx1 = int(round(min(b["xmin"], b["xmax"])))
                by1 = int(round(min(b["ymin"], b["ymax"])))
                bx2 = int(round(max(b["xmin"], b["xmax"])))
                by2 = int(round(max(b["ymin"], b["ymax"])))
                all_pts.extend([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]])

        if len(comp_boxes) == 1 and comp_boxes[0].get("polygon"):
            single_poly = comp_boxes[0]["polygon"]
            merged_poly = order_polygon_vertices(single_poly, avg_angle) if abs(avg_angle) >= 2.5 and len(single_poly) == 4 else single_poly
        elif abs(avg_angle) >= 2.5 and len(all_pts) >= 3:
            pts_arr = np.asarray(all_pts, dtype=np.float32)
            rad = math.radians(avg_angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            u = np.array([cos_a, sin_a], dtype=np.float32)
            v = np.array([-sin_a, cos_a], dtype=np.float32)
            proj_u = pts_arr @ u
            proj_v = pts_arr @ v
            u_min, u_max = float(np.min(proj_u)), float(np.max(proj_u))
            v_min, v_max = float(np.min(proj_v)), float(np.max(proj_v))
            tl = u_min * u + v_min * v
            tr = u_max * u + v_min * v
            br = u_max * u + v_max * v
            bl = u_min * u + v_max * v
            merged_poly = [[int(round(p[0])), int(round(p[1]))] for p in [tl, tr, br, bl]]
            c_xmin = min(c_xmin, min(p[0] for p in merged_poly))
            c_ymin = min(c_ymin, min(p[1] for p in merged_poly))
            c_xmax = max(c_xmax, max(p[0] for p in merged_poly))
            c_ymax = max(c_ymax, max(p[1] for p in merged_poly))
        else:
            merged_poly = [
                [int(round(c_xmin)), int(round(c_ymin))],
                [int(round(c_xmax)), int(round(c_ymin))],
                [int(round(c_xmax)), int(round(c_ymax))],
                [int(round(c_xmin)), int(round(c_ymax))]
            ]

        merged.append({
            "xmin": int(round(c_xmin)),
            "ymin": int(round(c_ymin)),
            "xmax": int(round(c_xmax)),
            "ymax": int(round(c_ymax)),
            "text": final_text,
            "conf": float(np.mean(confs)) if confs else 1.0,
            "line_count": line_count,
            "angle": avg_angle,
            "polygon": merged_poly,
            "bg_color": comp_boxes[0].get("bg_color")
        })

    return merged
