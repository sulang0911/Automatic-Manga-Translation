"""
app/core/ocr/ctd_engine.py
Comic-Text-Detector (CTD) engine for comic balloon, dialogue, and slanted text detection.
Pure PyTorch implementation optimized for manga layouts.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch

from app.core.ocr.base import calculate_polygon_angle

logger = logging.getLogger(__name__)

# Default search locations for comictextdetector.pt
DEFAULT_CTD_WEIGHT_PATHS = [
    Path.home() / ".cache" / "manga-ocr" / "comictextdetector.pt",
    Path("models") / "comictextdetector.pt",
    Path("weights") / "comictextdetector.pt",
]


class ComicTextDetectorEngine:
    """
    Wrapper around comic_text_detector.inference.TextDetector.
    Detects dialogue bubbles, onomatopoeia, and slanted text lines with high recall.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        input_size: int = 1024,
        conf_thresh: float = 0.40,
        nms_thresh: float = 0.35,
    ):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.model_path = self._resolve_model_path(model_path)
        self._detector = None

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Optional[Path]:
        if model_path:
            p = Path(model_path)
            if p.is_file():
                return p

        for candidate in DEFAULT_CTD_WEIGHT_PATHS:
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Checks if comic_text_detector package and weights are both present."""
        try:
            import comic_text_detector
            for candidate in DEFAULT_CTD_WEIGHT_PATHS:
                if candidate.is_file():
                    return True
            return False
        except ImportError:
            return False

    def _ensure_loaded(self):
        if self._detector is not None:
            return
        if not self.model_path or not self.model_path.is_file():
            raise FileNotFoundError(
                f"Comic-Text-Detector weights not found at {self.model_path}. "
                f"Searched: {[str(p) for p in DEFAULT_CTD_WEIGHT_PATHS]}"
            )

        try:
            from comic_text_detector.inference import TextDetector
            logger.info(f"Loading Comic-Text-Detector from '{self.model_path}' on device '{self.device}'...")
            self._detector = TextDetector(
                str(self.model_path),
                input_size=self.input_size,
                device=self.device,
                act="leaky",
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
            )
            logger.info("Comic-Text-Detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Comic-Text-Detector: {e}", exc_info=True)
            raise

    def detect(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Detects text blocks and lines from a BGR image.

        Returns:
            Tuple of (raw_boxes, refined_mask) where:
            - raw_boxes is a list of dicts with keys:
              xmin, ymin, xmax, ymax, angle, polygon, conf, lines
            - refined_mask is a numpy uint8 binary/grayscale mask of text strokes (if available)
        """
        if image is None or image.size == 0:
            return [], None

        self._ensure_loaded()
        h_img, w_img = image.shape[:2]

        try:
            mask, mask_refined, blk_list = self._detector(image)
        except Exception as e:
            logger.warning(f"CTD inference error: {e}", exc_info=True)
            return [], None

        raw_boxes = []
        for blk in blk_list:
            xyxy = getattr(blk, "xyxy", None)
            if not xyxy or len(xyxy) < 4:
                continue

            bx1 = int(round(max(0, min(w_img - 1, xyxy[0]))))
            by1 = int(round(max(0, min(h_img - 1, xyxy[1]))))
            bx2 = int(round(max(bx1 + 1, min(w_img, xyxy[2]))))
            by2 = int(round(max(by1 + 1, min(h_img, xyxy[3]))))

            if bx2 <= bx1 or by2 <= by1:
                continue

            # Angle extraction from CTD block (CTD returns integer degrees)
            blk_angle = float(getattr(blk, "angle", 0.0) or 0.0)
            if abs(blk_angle) < 15.0:
                eff_angle = 0.0
            else:
                eff_angle = round(blk_angle, 1)

            # Polygons and lines extraction
            lines = getattr(blk, "lines", [])
            lines_data = []
            all_line_pts = []
            for ln in lines:
                ln_arr = np.asarray(ln, dtype=np.int32)
                if len(ln_arr) >= 4:
                    lines_data.append(ln_arr.tolist())
                    all_line_pts.extend(ln_arr.tolist())

            if abs(eff_angle) >= 15.0 and len(all_line_pts) >= 4:
                # Compute oriented polygon from points
                rect = cv2.minAreaRect(np.array(all_line_pts, dtype=np.float32))
                box_pts = cv2.boxPoints(rect).astype(int).tolist()
                poly = box_pts
            else:
                poly = [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]]

            conf = float(getattr(blk, "prob", 1.0) or 1.0)

            raw_boxes.append({
                "xmin": bx1,
                "ymin": by1,
                "xmax": bx2,
                "ymax": by2,
                "text": "",  # To be filled by recognizer
                "conf": conf,
                "polygon": poly,
                "angle": eff_angle,
                "lines": lines_data,
                "line_count": max(1, len(lines_data)),
            })

        return raw_boxes, mask_refined
