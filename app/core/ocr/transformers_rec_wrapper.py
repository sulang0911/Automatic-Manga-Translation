"""
app/core/ocr/transformers_rec_wrapper.py
Pure-PyTorch text-line recognition via HuggingFace `transformers` (`PP-OCRv6_medium_rec`,
`PP-OCRv5_server_rec`, `korean_PP-OCRv5_mobile_rec`) with CUDA acceleration and
8GB-RAM-safe LRU caching.
"""
from __future__ import annotations

import gc
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported lightweight pure-PyTorch text-line recognizers (no PaddlePaddle CUDA needed)
TRANSFORMER_OCR_MODELS: Dict[str, str] = {
    "ppocr_v6": "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
    "ppocr_v5_server": "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
    "korean_ppocr_v5": "PaddlePaddle/korean_PP-OCRv5_mobile_rec_safetensors",
}

_SPEAKER_TAG_RE = re.compile(r"^\s*[\(\[\{（【〔][^)\]\}）】〕\n]{1,14}[\)\]\}）】〕]")
_GLOBAL_REC_CACHE: "OrderedDict[Tuple[str, str], TransformersOCRRecognizer]" = OrderedDict()
_MAX_CACHED_MODELS = 2  # Cap concurrent models in memory on 8GB RAM systems


def _release_torch_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xFF00 <= code <= 0xFFEF
    )


def join_recognized_lines(lines: Sequence[str]) -> str:
    """Joins recognized line fragments naturally for both Latin dialogue and CJK bubbles."""
    cleaned = [ln.strip() for ln in lines if ln and ln.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    out = cleaned[0]
    for nxt in cleaned[1:]:
        if _SPEAKER_TAG_RE.match(nxt):
            out = f"{out}\n{nxt}"
            continue
        last_ch = out[-1]
        first_ch = nxt[0]
        if _is_cjk_char(last_ch) and _is_cjk_char(first_ch):
            out = f"{out}{nxt}"
        elif last_ch.isspace() or first_ch in ",.!?;:)]}”’。，！？：；、」』】）":
            out = f"{out}{nxt}"
        else:
            out = f"{out} {nxt}"
    return out.strip()


def _rotate_line_patch(img_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Extracts and deskews a text-line polygon crop into an upright horizontal strip."""
    h_img, w_img = img_bgr.shape[:2]
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1 = max(0, int(np.floor(xs.min())) - 2)
    y1 = max(0, int(np.floor(ys.min())) - 2)
    x2 = min(w_img, int(np.ceil(xs.max())) + 3)
    y2 = min(h_img, int(np.ceil(ys.max())) + 3)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    patch = img_bgr[y1:y2, x1:x2]
    ph, pw = patch.shape[:2]
    # Vertical CJK line strip: rotate 90 deg CCW so SVTR reads left-to-right
    if ph > pw * 1.45 and ph >= 32:
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Orient along the longest box edge (immune to OpenCV 4.x minAreaRect 0/90-deg axis swaps)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    bp = cv2.boxPoints(rect)
    e0 = bp[1] - bp[0]
    e1 = bp[2] - bp[1]
    vec = e0 if float(np.dot(e0, e0)) >= float(np.dot(e1, e1)) else e1
    if vec[0] < 0:
        vec = -vec
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))

    if abs(angle) >= 3.0 and pw >= 16 and ph >= 12:
        mat = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, 1.0)
        return cv2.warpAffine(patch, mat, (pw, ph), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return patch


def _split_horizontal_text_lines(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """Splits a multi-line horizontal box into line strips when explicit CTD lines are unavailable."""
    h, w = crop_bgr.shape[:2]
    if h < 52 or w > h * 3.8 or h > w * 2.2:
        return [crop_bgr]

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # High gradient magnitude locates text strokes reliably on both dark and light backgrounds
    grad_x = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    row_energy = np.mean(grad_x + grad_y, axis=1)
    peak = float(row_energy.max()) if row_energy.size else 0.0
    if peak < 18.0:
        return [crop_bgr]

    active = row_energy > max(6.0, peak * 0.16)
    spans: List[Tuple[int, int]] = []
    in_span = False
    start = 0
    for y, flag in enumerate(active):
        if flag and not in_span:
            start = y
            in_span = True
        elif not flag and in_span:
            if y - start >= 8:
                spans.append((start, y))
            in_span = False
    if in_span and h - start >= 8:
        spans.append((start, h))

    # Merge tiny vertical gaps caused by accents/descenders (< 4px)
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    if len(merged) <= 1:
        return [crop_bgr]

    crops: List[np.ndarray] = []
    for s, e in merged:
        pad_s = max(0, s - 3)
        pad_e = min(h, e + 3)
        if pad_e - pad_s >= 10:
            crops.append(crop_bgr[pad_s:pad_e, :])
    return crops or [crop_bgr]


def extract_line_crops_for_box(image_bgr: np.ndarray, box: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
    """
    Extracts ordered upright text-line strips for a bubble box.
    Uses Comic-Text-Detector's `lines` polygons when present; falls back cleanly otherwise.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    raw_lines = (box or {}).get("lines") or []
    if raw_lines:
        parsed_lines: List[Tuple[float, float, float, float, np.ndarray]] = []
        vertical_votes = 0
        for ln in raw_lines:
            pts = np.asarray(ln, dtype=np.float32)
            if pts.ndim != 2 or len(pts) < 4:
                continue
            lw = float(pts[:, 0].max() - pts[:, 0].min())
            lh = float(pts[:, 1].max() - pts[:, 1].min())
            if lh > lw * 1.45:
                vertical_votes += 1
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            parsed_lines.append((cx, cy, lw, lh, pts))

        if parsed_lines:
            is_vertical_block = vertical_votes > len(parsed_lines) / 2.0
            if is_vertical_block:
                parsed_lines.sort(key=lambda item: (-item[0], item[1]))
            else:
                # Cluster fragments on the same horizontal row, sort left-to-right, and merge adjacent chunks
                parsed_lines.sort(key=lambda item: (item[1], item[0]))
                rows: List[List[Tuple[float, float, float, float, np.ndarray]]] = []
                for item in parsed_lines:
                    placed = False
                    for row in rows:
                        ref_cy = sum(r[1] for r in row) / len(row)
                        ref_h = max(r[3] for r in row)
                        if abs(item[1] - ref_cy) <= max(6.0, min(item[3], ref_h) * 0.55):
                            row.append(item)
                            placed = True
                            break
                    if not placed:
                        rows.append([item])

                rows.sort(key=lambda r: sum(x[1] for x in r) / len(r))
                merged_pts_list: List[np.ndarray] = []
                for row in rows:
                    row.sort(key=lambda x: x[4][:, 0].min())
                    cur_pts = row[0][4]
                    for nxt in row[1:]:
                        gap_x = float(nxt[4][:, 0].min() - cur_pts[:, 0].max())
                        row_h = float(max(cur_pts[:, 1].max() - cur_pts[:, 1].min(), nxt[3]))
                        if gap_x <= max(24.0, row_h * 1.5):
                            cur_pts = np.vstack([cur_pts, nxt[4]])
                        else:
                            merged_pts_list.append(cur_pts)
                            cur_pts = nxt[4]
                    merged_pts_list.append(cur_pts)

                parsed_lines = [
                    (float(p[:, 0].mean()), float(p[:, 1].mean()),
                     float(p[:, 0].max() - p[:, 0].min()), float(p[:, 1].max() - p[:, 1].min()), p)
                    for p in merged_pts_list
                ]

            strips = [_rotate_line_patch(image_bgr, item[4]) for item in parsed_lines]
            strips = [s for s in strips if s.size > 0 and s.shape[0] >= 4 and s.shape[1] >= 4]
            if strips:
                return strips

    # No explicit CTD lines: crop bbox if provided, else image_bgr IS already the crop
    crop = image_bgr
    if box and all(k in box for k in ("xmin", "ymin", "xmax", "ymax")):
        h_img, w_img = image_bgr.shape[:2]
        x1 = max(0, min(w_img - 1, int(box["xmin"])))
        y1 = max(0, min(h_img - 1, int(box["ymin"])))
        x2 = max(x1 + 1, min(w_img, int(box["xmax"])))
        y2 = max(y1 + 1, min(h_img, int(box["ymax"])))
        if (x2 - x1) < w_img or (y2 - y1) < h_img:
            crop = image_bgr[y1:y2, x1:x2]

    return _split_horizontal_text_lines(crop)


class TransformersOCRRecognizer:
    """Lazy-loading pure-PyTorch line recognizer powered by HuggingFace `transformers`."""

    def __init__(self, model_key: str = "ppocr_v6", use_gpu: bool = True):
        self.model_key = model_key
        self.repo_id = TRANSFORMER_OCR_MODELS.get(model_key, model_key)
        self.use_gpu = bool(use_gpu)
        self._device = "cpu"
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModelForTextRecognition

        self._device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        logger.info("Loading Transformers OCR model '%s' on %s...", self.repo_id, self._device)
        self._processor = AutoImageProcessor.from_pretrained(self.repo_id)
        self._model = AutoModelForTextRecognition.from_pretrained(self.repo_id).to(self._device).eval()
        logger.info("Transformers OCR model '%s' ready.", self.repo_id)

    def recognize_lines(self, line_crops_bgr: Sequence[np.ndarray], batch_size: int = 1) -> List[Tuple[str, float]]:
        """Recognizes a sequence of upright single-line BGR crops (batch_size=1 avoids variable-width padding noise)."""
        valid_indices: List[int] = []
        pil_images: List[Any] = []
        from PIL import Image

        for idx, crop in enumerate(line_crops_bgr):
            if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))
            valid_indices.append(idx)

        results: List[Tuple[str, float]] = [("", 0.0) for _ in line_crops_bgr]
        if not pil_images:
            return results

        self._ensure_loaded()
        import torch

        decoded: List[Tuple[str, float]] = []
        for start in range(0, len(pil_images), batch_size):
            batch_pils = pil_images[start : start + batch_size]
            inputs = self._processor(images=batch_pils, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            preds = self._processor.post_process_text_recognition(outputs)
            for item in preds:
                text = (item.get("text") or "").strip() if isinstance(item, dict) else str(item).strip()
                score = float(item.get("score", 0.0) or 0.0) if isinstance(item, dict) else 0.0
                decoded.append((text, score))

        for pos, (text, score) in zip(valid_indices, decoded):
            results[pos] = (text, score)
        return results

    def recognize_box(self, image_bgr: np.ndarray, box: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """Recognizes an entire speech bubble/box by extracting and joining its constituent text lines."""
        strips = extract_line_crops_for_box(image_bgr, box=box)
        if not strips:
            return "", 0.0
        line_preds = self.recognize_lines(strips)
        texts = [t for t, _ in line_preds if t]
        confs = [c for t, c in line_preds if t]
        if not texts:
            return "", 0.0
        return join_recognized_lines(texts), float(sum(confs) / len(confs))

    def unload(self):
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            del self._model
        self._model = None
        self._processor = None
        _release_torch_memory()


def get_transformers_ocr(model_key: str = "ppocr_v6", use_gpu: bool = True) -> TransformersOCRRecognizer:
    """Returns a cached TransformersOCRRecognizer, evicting LRU models to protect 8GB RAM."""
    import torch
    resolved_dev = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    cache_key = (model_key, resolved_dev)

    if cache_key in _GLOBAL_REC_CACHE:
        _GLOBAL_REC_CACHE.move_to_end(cache_key)
        return _GLOBAL_REC_CACHE[cache_key]

    while len(_GLOBAL_REC_CACHE) >= _MAX_CACHED_MODELS:
        _, oldest = _GLOBAL_REC_CACHE.popitem(last=False)
        oldest.unload()

    rec = TransformersOCRRecognizer(model_key=model_key, use_gpu=use_gpu)
    _GLOBAL_REC_CACHE[cache_key] = rec
    return rec
