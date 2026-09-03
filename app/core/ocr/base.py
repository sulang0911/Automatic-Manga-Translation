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


def merge_adjacent_boxes(
    raw_boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    v_thresh_ratio: float = 0.025,
    h_thresh_ratio: float = 0.035
) -> List[Dict[str, Any]]:
    """
    Merges closely adjacent or overlapping text lines within speech bubbles.
    """
    if not raw_boxes:
        return []

    v_thresh = max(4, int(img_h * v_thresh_ratio))
    h_thresh = max(4, int(img_w * h_thresh_ratio))

    merged = []
    used = [False] * len(raw_boxes)

    for i in range(len(raw_boxes)):
        if used[i]:
            continue
        cur_xmin = raw_boxes[i]["xmin"]
        cur_ymin = raw_boxes[i]["ymin"]
        cur_xmax = raw_boxes[i]["xmax"]
        cur_ymax = raw_boxes[i]["ymax"]
        cur_texts = [raw_boxes[i]["text"]]
        cur_confs = [raw_boxes[i].get("conf", 1.0)]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(raw_boxes)):
                if used[j]:
                    continue
                b = raw_boxes[j]
                x_overlap = not (b["xmax"] < cur_xmin - h_thresh or b["xmin"] > cur_xmax + h_thresh)
                y_overlap = not (b["ymax"] < cur_ymin - v_thresh or b["ymin"] > cur_ymax + v_thresh)

                if x_overlap and y_overlap:
                    cur_xmin = min(cur_xmin, b["xmin"])
                    cur_ymin = min(cur_ymin, b["ymin"])
                    cur_xmax = max(cur_xmax, b["xmax"])
                    cur_ymax = max(cur_ymax, b["ymax"])
                    cur_texts.append(b["text"])
                    cur_confs.append(b.get("conf", 1.0))
                    used[j] = True
                    changed = True

        merged.append({
            "xmin": cur_xmin,
            "ymin": cur_ymin,
            "xmax": cur_xmax,
            "ymax": cur_ymax,
            "text": "\n".join(cur_texts),
            "conf": float(np.mean(cur_confs)),
            "line_count": len(cur_texts)
        })

    return merged
