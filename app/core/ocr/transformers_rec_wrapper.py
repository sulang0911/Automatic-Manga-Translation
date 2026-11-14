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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported lightweight pure-PyTorch text-line recognizers (no PaddlePaddle CUDA needed)
TRANSFORMER_OCR_MODELS: Dict[str, str] = {
    "svtrv2": "topdu/OpenOCR",
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


def expand_condensed_strip(
    crop_bgr: np.ndarray,
    scale_x: float = 1.45,
    min_aspect: float = 6.0,
    force: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Widens condensed/long horizontal dialogue strips so CTC maintains >=1.5 frames per glyph."""
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0 or (not force and (w / float(h)) < min_aspect):
        return crop_bgr, False
    new_w = max(w + 1, int(round(w * scale_x)))
    return cv2.resize(crop_bgr, (new_w, h), interpolation=cv2.INTER_CUBIC), True


def refine_condensed_line(
    orig_crop: np.ndarray,
    text: str,
    conf: float,
    infer_fn: Callable[[np.ndarray], Tuple[str, float]],
) -> Tuple[str, float]:
    """Retries CTC recognition with 1.45x horizontal expansion when a strip appears under-sampled."""
    h, w = orig_crop.shape[:2]
    if h <= 0 or (w / float(h)) < 3.5:
        return text, conf
    stripped = (text or "").strip()
    if stripped and conf >= 0.78 and len(stripped) >= (w / max(1.0, h * 1.4)):
        return text, conf
    wider, _ = expand_condensed_strip(orig_crop, force=True)
    alt_text, alt_conf = infer_fn(wider)
    if alt_text and (not stripped or alt_conf > conf + 0.03 or len(alt_text) > len(stripped) * 1.4):
        return alt_text, alt_conf
    return text, conf


def recognize_box_lines(
    image_bgr: np.ndarray,
    box: Optional[Dict[str, Any]],
    recognize_lines_fn: Callable[[Sequence[np.ndarray]], List[Tuple[str, float]]],
) -> Tuple[str, float]:
    """Extracts lines for a speech bubble/box, recognizes them, and merges text + mean confidence."""
    strips = extract_line_crops_for_box(image_bgr, box=box)
    if not strips:
        return "", 0.0
    line_preds = recognize_lines_fn(strips)
    texts = [t for t, _ in line_preds if t]
    confs = [c for t, c in line_preds if t]
    if not texts:
        return "", 0.0
    return join_recognized_lines(texts), float(sum(confs) / len(confs))


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

    pw, ph = x2 - x1, y2 - y1
    # Vertical CJK line strip: rotate 90 deg CCW so SVTR reads left-to-right
    if ph > pw * 1.45 and ph >= 32:
        return cv2.rotate(img_bgr[y1:y2, x1:x2], cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Orient along the longest box edge (immune to OpenCV 4.x minAreaRect 0/90-deg axis swaps)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (rcx, rcy), _, _ = rect
    bp = cv2.boxPoints(rect)
    e0 = bp[1] - bp[0]
    e1 = bp[2] - bp[1]
    if float(np.dot(e0, e0)) >= float(np.dot(e1, e1)):
        vec, long_len, short_len = e0, float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    else:
        vec, long_len, short_len = e1, float(np.linalg.norm(e1)), float(np.linalg.norm(e0))
    if vec[0] < 0:
        vec = -vec
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))

    if abs(angle) >= 3.0 and long_len >= 16 and short_len >= 12:
        # Warp tightly around rotated box center so adjacent stacked tilted lines aren't captured
        tw = max(16, int(np.ceil(long_len)) + 6)
        th = max(12, int(np.ceil(short_len)) + 6)
        mat = cv2.getRotationMatrix2D((float(rcx), float(rcy)), angle, 1.0)
        mat[0, 2] += (tw / 2.0) - float(rcx)
        mat[1, 2] += (th / 2.0) - float(rcy)
        return cv2.warpAffine(img_bgr, mat, (tw, th), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img_bgr[y1:y2, x1:x2]


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
            rect_dims = cv2.minAreaRect(pts)[1]
            thick = float(min(rect_dims)) if min(rect_dims) > 1.0 else lh
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            parsed_lines.append((cx, cy, lw, thick, pts))

        if parsed_lines:
            is_vertical_block = vertical_votes > len(parsed_lines) / 2.0
            if is_vertical_block:
                parsed_lines.sort(key=lambda item: (-item[0], item[1]))
            else:
                # Cluster fragments on the same horizontal row, sort left-to-right, and merge adjacent chunks
                parsed_lines.sort(key=lambda item: (item[1], item[0]))
                rows: List[List[Tuple[float, float, float, float, np.ndarray]]] = []
                for item in parsed_lines:
                    ix1 = float(item[4][:, 0].min())
                    ix2 = float(item[4][:, 0].max())
                    placed = False
                    for row in rows:
                        ref_cy = sum(r[1] for r in row) / len(row)
                        ref_h = max(r[3] for r in row)
                        if abs(item[1] - ref_cy) > max(6.0, min(item[3], ref_h) * 0.55):
                            continue
                        # Stacked lines overlap heavily in X; only co-linear fragments belong on the same row
                        if any(
                            (min(ix2, float(r[4][:, 0].max())) - max(ix1, float(r[4][:, 0].min())))
                            > max(12.0, min(item[2], r[2]) * 0.30)
                            for r in row
                        ):
                            continue
                        row.append(item)
                        placed = True
                        break
                    if not placed:
                        rows.append([item])

                rows.sort(key=lambda r: sum(x[1] for x in r) / len(r))
                box_w = float(max(0, (box or {}).get("xmax", 0) - (box or {}).get("xmin", 0)))
                merged_pts_list: List[np.ndarray] = []
                for row in rows:
                    row.sort(key=lambda x: x[4][:, 0].min())
                    cur_pts = row[0][4]
                    for nxt in row[1:]:
                        cur_w = float(cur_pts[:, 0].max() - cur_pts[:, 0].min())
                        gap_x = float(nxt[4][:, 0].min() - cur_pts[:, 0].max())
                        row_h = float(max(cur_pts[:, 1].max() - cur_pts[:, 1].min(), nxt[3]))
                        if -max(12.0, min(cur_w, nxt[2]) * 0.25) <= gap_x <= max(48.0, row_h * 8.0, box_w * 0.65):
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
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.repo_id, local_files_only=True)
            self._model = AutoModelForTextRecognition.from_pretrained(self.repo_id, local_files_only=True).to(self._device).eval()
        except Exception:
            self._processor = AutoImageProcessor.from_pretrained(self.repo_id)
            self._model = AutoModelForTextRecognition.from_pretrained(self.repo_id).to(self._device).eval()
        logger.info("Transformers OCR model '%s' ready.", self.repo_id)

    def _infer_pil(self, batch_pils: Sequence[Any]) -> List[Tuple[str, float]]:
        import torch

        inputs = self._processor(images=list(batch_pils), return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        preds = self._processor.post_process_text_recognition(outputs)
        decoded: List[Tuple[str, float]] = []
        for item in preds:
            text = (item.get("text") or "").strip() if isinstance(item, dict) else str(item).strip()
            score = float(item.get("score", 0.0) or 0.0) if isinstance(item, dict) else 0.0
            decoded.append((text, score))
        return decoded

    def recognize_lines(self, line_crops_bgr: Sequence[np.ndarray], batch_size: int = 1) -> List[Tuple[str, float]]:
        """Recognizes a sequence of upright single-line BGR crops (batch_size=1 avoids variable-width padding noise)."""
        valid: List[Tuple[int, np.ndarray, Any, bool]] = []
        from PIL import Image

        use_condensed_fix = (self.model_key == "ppocr_v6")
        for idx, crop in enumerate(line_crops_bgr):
            if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            prepped, stretched = expand_condensed_strip(crop) if use_condensed_fix else (crop, False)
            rgb = cv2.cvtColor(prepped, cv2.COLOR_BGR2RGB)
            valid.append((idx, crop, Image.fromarray(rgb), stretched))

        results: List[Tuple[str, float]] = [("", 0.0) for _ in line_crops_bgr]
        if not valid:
            return results

        self._ensure_loaded()
        for start in range(0, len(valid), batch_size):
            chunk = valid[start : start + batch_size]
            decoded = self._infer_pil([item[2] for item in chunk])
            for (pos, orig_crop, _, stretched), (text, score) in zip(chunk, decoded):
                if use_condensed_fix and not stretched:
                    text, score = refine_condensed_line(
                        orig_crop,
                        text,
                        score,
                        lambda img: self._infer_pil([Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))])[0],
                    )
                results[pos] = (text, score)
        return results

    def recognize_box(self, image_bgr: np.ndarray, box: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """Recognizes an entire speech bubble/box by extracting and joining its constituent text lines."""
        return recognize_box_lines(image_bgr, box, self.recognize_lines)

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


def get_transformers_ocr(model_key: str = "ppocr_v6", use_gpu: bool = True) -> Any:
    """Returns a cached pure-PyTorch text-line recognizer, evicting LRU models to protect 8GB RAM."""
    import torch
    resolved_dev = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    cache_key = (model_key, resolved_dev)

    if cache_key in _GLOBAL_REC_CACHE:
        _GLOBAL_REC_CACHE.move_to_end(cache_key)
        return _GLOBAL_REC_CACHE[cache_key]

    while len(_GLOBAL_REC_CACHE) >= _MAX_CACHED_MODELS:
        _, oldest = _GLOBAL_REC_CACHE.popitem(last=False)
        oldest.unload()

    if model_key == "svtrv2":
        from app.core.ocr.svtrv2_wrapper import SVTRv2OCRRecognizer
        rec: Any = SVTRv2OCRRecognizer(model_key=model_key, use_gpu=use_gpu)
    else:
        rec = TransformersOCRRecognizer(model_key=model_key, use_gpu=use_gpu)
    _GLOBAL_REC_CACHE[cache_key] = rec
    return rec
