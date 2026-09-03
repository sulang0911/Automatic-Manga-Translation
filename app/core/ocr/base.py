"""
app/core/ocr/base.py
Abstract base class and utility functions for OCR detection and recognition engines.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from app.core.models import TranslationBlock


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


def can_merge_pair(
    b1: Dict[str, Any],
    b2: Dict[str, Any],
    img_w: int,
    img_h: int,
    v_thresh_ratio: float = 0.025,
    h_thresh_ratio: float = 0.035
) -> bool:
    """
    Evaluates whether two detected text boxes belong to the same dialogue bubble / text region.
    Prevents merging adjacent independent speech bubbles while correctly aggregating multi-line text
    and same-line / same-column fragments.
    """
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

    # Sensitivity scales derived from threshold ratios
    v_scale = max(0.2, float(v_thresh_ratio) / 0.025)
    h_scale = max(0.2, float(h_thresh_ratio) / 0.035)

    has_cjk = any(_is_cjk_char(c) for c in str(b1.get("text", ""))) or any(_is_cjk_char(c) for c in str(b2.get("text", "")))

    # Criterion 1: Significant 2D overlap / Inclusion (>= 60% area of smaller box)
    inter_area = x_overlap * y_overlap
    min_area = min_w * min_h
    if min_area > 0 and (inter_area / min_area) >= 0.60:
        return True

    # Criterion 2: Same-line fragments (horizontal text collinear fragments / words)
    is_not_vert_columns = not (has_cjk and h1 >= w1 * 1.20 and h2 >= w2 * 1.20)
    if is_not_vert_columns:
        if (y_overlap_ratio >= 0.45 or abs(mid_y1 - mid_y2) <= max_h * 0.40) and (min_h / max_h) >= 0.40:
            max_word_gap = max(8, int(0.75 * max_h * h_scale))
            if x_gap <= max_word_gap:
                return True

    # Criterion 3: Same-column fragments (vertical text collinear fragments, CJK only)
    if has_cjk:
        is_not_horiz_lines = not (w1 >= h1 * 1.35 and w2 >= h2 * 1.35)
        if is_not_horiz_lines and (min_w / max_w) >= 0.40:
            if (x_overlap_ratio >= 0.45 or abs(mid_x1 - mid_x2) <= max_w * 0.40):
                max_char_gap = max(8, int(0.85 * max_w * v_scale))
                if y_gap <= max_char_gap:
                    return True

    # Criterion 4: Horizontal multi-line text (lines stacked vertically within bubble)
    # Western text is always horizontal lines stacked vertically.
    # CJK horizontal text has horizontally oriented lines or wide aspect ratio.
    is_horiz_candidate = (not has_cjk) or (h1 <= w1 * 1.35 and h2 <= w2 * 1.35) or (
        max(w1, w2) >= max_h * 1.2 and min_h / max_h >= 0.40
    )
    if is_horiz_candidate and (min_h / max_h) >= 0.40:
        if y1_min <= y2_min:
            t_ymin, t_ymax = y1_min, y1_max
            b_ymin, b_ymax = y2_min, y2_max
        else:
            t_ymin, t_ymax = y2_min, y2_max
            b_ymin, b_ymax = y1_min, y1_max

        if b_ymin >= t_ymin and b_ymax >= t_ymax:
            curr_y_gap = b_ymin - t_ymax
            if curr_y_gap < 0:
                y_overlap_len = -curr_y_gap
                y_gap_ok = y_overlap_len <= int(0.35 * min_h)
            else:
                max_line_gap = max(4, int(0.70 * min_h * v_scale))
                y_gap_ok = curr_y_gap <= max_line_gap

            if y_gap_ok:
                x_aligned = (
                    (x_overlap_ratio >= 0.35 and abs(mid_x1 - mid_x2) <= max_w * 0.65)
                    or (x_overlap > 0 and (x_overlap / max_w >= 0.30))
                )
                if x_aligned:
                    return True

    # Criterion 5: Vertical multi-column text (columns side-by-side in same bubble, CJK only)
    if has_cjk:
        is_vert_candidate = (
            not (w1 >= h1 * 1.35 and w2 >= h2 * 1.35)
            and (min_w / max_w) >= 0.40
            and max(h1, h2) >= max_w * 0.8
        )
        if is_vert_candidate:
            if x1_min >= x2_min:
                r_xmin, r_xmax = x1_min, x1_max
                l_xmin, l_xmax = x2_min, x2_max
            else:
                r_xmin, r_xmax = x2_min, x2_max
                l_xmin, l_xmax = x1_min, x1_max

            if r_xmin >= l_xmin and r_xmax >= l_xmax:
                curr_x_gap = r_xmin - l_xmax
                if curr_x_gap < 0:
                    x_overlap_len = -curr_x_gap
                    x_gap_ok = x_overlap_len <= int(0.35 * min_w)
                else:
                    max_col_gap = max(4, int(0.70 * min_w * h_scale))
                    x_gap_ok = curr_x_gap <= max_col_gap

                if x_gap_ok:
                    y_aligned = (
                        (y_overlap_ratio >= 0.35 and abs(mid_y1 - mid_y2) <= max_h * 0.65)
                        or (y_overlap > 0 and (y_overlap / max_h >= 0.30 or y_overlap / min_h >= 0.50))
                        or (y_overlap > 0 and abs(y1_min - y2_min) <= max_w * 2.0)
                    )
                    if y_aligned:
                        return True

    return False


def merge_adjacent_boxes(
    raw_boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    v_thresh_ratio: float = 0.025,
    h_thresh_ratio: float = 0.035
) -> List[Dict[str, Any]]:
    """
    Merges closely adjacent or overlapping text lines within speech bubbles.
    Effectively prevents merging adjacent independent dialogue bubbles while preserving
    multi-line text aggregation for both horizontal and vertical text layouts.
    """
    if not raw_boxes:
        return []

    img_w = max(1, int(img_w))
    img_h = max(1, int(img_h))

    n = len(raw_boxes)
    if n == 1:
        b = raw_boxes[0]
        return [{
            "xmin": int(round(min(b["xmin"], b["xmax"]))),
            "ymin": int(round(min(b["ymin"], b["ymax"]))),
            "xmax": int(round(max(b["xmin"], b["xmax"]))),
            "ymax": int(round(max(b["ymin"], b["ymax"]))),
            "text": str(b.get("text", "")).strip(),
            "conf": float(b.get("conf", 1.0) if b.get("conf") is not None else 1.0),
            "line_count": 1
        }]

    # Build adjacency graph based on precise pairing criteria
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if can_merge_pair(raw_boxes[i], raw_boxes[j], img_w, img_h, v_thresh_ratio, h_thresh_ratio):
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

            row_texts = []
            for row in rows:
                row_sorted = sorted(row, key=lambda b: min(b["xmin"], b["xmax"]))
                row_text = _join_line_texts([str(b.get("text", "")).strip() for b in row_sorted])
                if row_text:
                    row_texts.append(row_text)

            final_text = "\n".join(row_texts)
            line_count = len(row_texts) if row_texts else 1

        merged.append({
            "xmin": int(round(c_xmin)),
            "ymin": int(round(c_ymin)),
            "xmax": int(round(c_xmax)),
            "ymax": int(round(c_ymax)),
            "text": final_text,
            "conf": float(np.mean(confs)) if confs else 1.0,
            "line_count": line_count
        })

    return merged
