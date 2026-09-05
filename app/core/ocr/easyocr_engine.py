"""
app/core/ocr/easyocr_engine.py
PyTorch-accelerated OCR backend based on EasyOCR.
Reliable GPU execution across all NVIDIA architectures (including Pascal GTX 10-series).
"""
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
import cv2
import numpy as np
import torch

from app.core.models import TranslationBlock, BlockType, TextDirection
from app.core.ocr.base import (
    BaseOCREngine,
    is_solid_color_page,
    merge_adjacent_boxes,
    calculate_polygon_angle
)
from app.core.ocr.reading_order import sort_reading_order
from app.core.inpaint.color_analyzer import get_background_color_hex, analyze_text_color

try:
    from app.core.pipeline import clean_ocr_syntax, clean_translation_syntax, normalize_domain_slang, prioritize_english_routing
except Exception:
    import re
    def clean_ocr_syntax(t: str) -> str:
        if not t: return ""
        t = re.sub(r'(\b\w+)\s*[:;]\s*([a-zA-Z]+)\b', r'\1 \2', t.strip())
        t = re.sub(r'[:;]\s*$', '', t)
        t = t.replace('》', '!').replace('《', '').replace('「', '"').replace('」', '"')
        return re.sub(r'\s{2,}', ' ', t).strip()
    def clean_translation_syntax(t: str, o: str = "") -> str:
        if not t: return ""
        t = re.sub(r'[：:]\s*[\u4e00-\u9fa5a-zA-Z]{1,4}\s*$', '', t.strip())
        return re.sub(r'[：:;；\-_]\s*$', '', t).strip()
    def normalize_domain_slang(t: str) -> str:
        if not t: return ""
        t = re.sub(r'\bGF\b', 'girlfriend', t)
        t = re.sub(r"\bBF['’]s\b", "boyfriend's", t)
        t = re.sub(r'\bBF\b', 'boyfriend', t)
        return re.sub(r'\b(no[-_ ]?fap)\b', 'no-fap', t, flags=re.IGNORECASE)
    def prioritize_english_routing(lang=None, image=None, text=None) -> bool:
        return bool(lang and any(w in str(lang).lower() for w in ["en", "eng", "english"]))

logger = logging.getLogger(__name__)


def sort_easyocr_fragments_2d(results: List[Any]) -> Tuple[str, float]:
    """
    Sorts 2D OCR text line fragments in natural reading order:
    Groups fragments into horizontal rows (lines), sorts each row left-to-right,
    and sorts lines top-to-bottom.
    Prevents scrambled lines such as "2 and1 masturlation later days".
    """
    if not results:
        return "", 0.0

    valid_items = []
    for r in results:
        if len(r) < 2:
            continue
        bbox, text = r[0], str(r[1]).strip()
        conf = float(r[2]) if len(r) > 2 else 1.0
        if not text:
            continue
        pts = np.asarray(bbox, dtype=np.float32)
        xmin = float(np.min(pts[:, 0]))
        ymin = float(np.min(pts[:, 1]))
        xmax = float(np.max(pts[:, 0]))
        ymax = float(np.max(pts[:, 1]))
        h = max(1.0, ymax - ymin)
        yc = (ymin + ymax) / 2.0
        valid_items.append({
            "text": text,
            "conf": conf,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "yc": yc, "h": h
        })

    if not valid_items:
        return "", 0.0
    if len(valid_items) == 1:
        return valid_items[0]["text"], valid_items[0]["conf"]

    valid_items.sort(key=lambda item: item["ymin"])

    rows: List[List[Dict[str, Any]]] = []
    row_bounds: List[Tuple[float, float, float]] = []

    for item in valid_items:
        matched_row_idx = -1
        for idx, (r_ymin, r_ymax, r_h) in enumerate(row_bounds):
            overlap = max(0.0, min(r_ymax, item["ymax"]) - max(r_ymin, item["ymin"]))
            min_h = min(r_h, item["h"])
            r_yc = (r_ymin + r_ymax) / 2.0
            if (overlap / max(1.0, min_h) >= 0.35) or (abs(item["yc"] - r_yc) <= 0.45 * min_h):
                matched_row_idx = idx
                break

        if matched_row_idx >= 0:
            rows[matched_row_idx].append(item)
            r_ymin, r_ymax, r_h = row_bounds[matched_row_idx]
            new_ymin = min(r_ymin, item["ymin"])
            new_ymax = max(r_ymax, item["ymax"])
            row_bounds[matched_row_idx] = (new_ymin, new_ymax, new_ymax - new_ymin)
        else:
            rows.append([item])
            row_bounds.append((item["ymin"], item["ymax"], item["h"]))

    for r in rows:
        r.sort(key=lambda item: item["xmin"])

    row_order = sorted(range(len(rows)), key=lambda idx: row_bounds[idx][0])

    ordered_lines = []
    all_confs = []
    for idx in row_order:
        line_text = " ".join(it["text"] for it in rows[idx] if it["text"])
        if line_text:
            ordered_lines.append(line_text)
            all_confs.extend(it["conf"] for it in rows[idx])

    joined = " ".join(ordered_lines)
    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    return joined, avg_conf


def rotate_crop_upright(crop: np.ndarray, angle: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Affine rotates crop upright by -angle so slanted text lines become horizontal.
    """
    if abs(angle) < 2.5 or crop is None or crop.size == 0:
        return crop, None
    h, w = crop.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos_val = abs(rot_mat[0, 0])
    sin_val = abs(rot_mat[0, 1])
    new_w = max(w, int(np.ceil((h * sin_val) + (w * cos_val))))
    new_h = max(h, int(np.ceil((h * cos_val) + (w * sin_val))))
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(
        crop, rot_mat, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, rot_mat


class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        self.languages = languages or ['ja', 'en']
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()
        self._reader = None
        self._reader_en = None

    def _get_reader(self, is_english: bool = False):
        if is_english:
            if self._reader_en is None:
                import easyocr
                logger.info(f"Initializing EasyOCR for English with langs=['en'], gpu={self.use_gpu}...")
                self._reader_en = easyocr.Reader(['en'], gpu=self.use_gpu)
            return self._reader_en
        if self._reader is None:
            import easyocr
            logger.info(f"Initializing EasyOCR with langs={self.languages}, gpu={self.use_gpu}...")
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader

    def is_available(self) -> bool:
        try:
            import easyocr
            return True
        except ImportError:
            return False

    def get_device_info(self) -> Dict[str, Any]:
        device_name = "CPU"
        if self.use_gpu and torch.cuda.is_available():
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        return {
            "engine": "EasyOCR",
            "device": device_name,
            "cuda_available": torch.cuda.is_available(),
            "languages": self.languages
        }

    def detect_and_recognize(
        self,
        image: np.ndarray,
        lang: str = "japan",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[TranslationBlock]:
        if image is None or image.size == 0 or is_solid_color_page(image):
            return []

        h_img, w_img = image.shape[:2]
        if progress_callback:
            progress_callback(15, "正在加载 EasyOCR 识别模型...")

        is_english = any(w in str(lang).lower() for w in ['en', 'eng', 'english', 'latin'])
        reader = self._get_reader(is_english=is_english)

        if progress_callback:
            progress_callback(35, "正在执行文字检测与多语言识别...")

        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_results = reader.readtext(rgb_img)

        # Pre-detect language if auto/default and check if page is actually English
        if not is_english and raw_results:
            sample_text = " ".join(str(r[1]).strip() for r in raw_results if len(r) > 1)
            import re
            lat_w = re.findall(r'[a-zA-Z]{2,}', sample_text)
            cjk_c = [c for c in sample_text if ('\u4e00' <= c <= '\u9fff') or ('\u3040' <= c <= '\u309f') or ('\u30a0' <= c <= '\u30ff')]
            if (len(lat_w) >= 3 and len("".join(lat_w)) > len(cjk_c) * 2) or (len(lat_w) >= 2 and len(cjk_c) == 0):
                is_english = True
                reader = self._get_reader(is_english=True)
                raw_results = reader.readtext(rgb_img)

        raw_boxes = []
        for bbox, text, conf in raw_results:
            clean_text = clean_ocr_syntax(str(text).strip())
            if not clean_text or conf < 0.20:
                continue
            pts = np.array(bbox, dtype=np.int32)
            angle = calculate_polygon_angle(pts)
            raw_boxes.append({
                "xmin": int(np.min(pts[:, 0])),
                "ymin": int(np.min(pts[:, 1])),
                "xmax": int(np.max(pts[:, 0])),
                "ymax": int(np.max(pts[:, 1])),
                "text": clean_text,
                "conf": float(conf),
                "polygon": pts.astype(int).tolist(),
                "angle": angle
            })

        # Filter spurious OCR boxes falling inside QR codes
        qr_regions = None
        try:
            from app.core.ocr.qr_filter import QRCodeFilter
            qr_filter = QRCodeFilter()
            qr_regions = qr_filter.detect_regions(image)
            if qr_regions:
                raw_boxes = qr_filter.filter_spurious_ocr_boxes(raw_boxes, qr_regions)
        except Exception:
            qr_regions = None

        if progress_callback:
            progress_callback(70, "正在合并临近对话段落...")

        merged_boxes = merge_adjacent_boxes(raw_boxes, w_img, h_img, image=image, qr_regions=qr_regions)

        blocks: List[TranslationBlock] = []
        for b in merged_boxes:
            xmin = max(0, min(b["xmin"], w_img - 1))
            ymin = max(0, min(b["ymin"], h_img - 1))
            xmax = max(xmin + 1, min(b["xmax"], w_img))
            ymax = max(ymin + 1, min(b["ymax"], h_img))

            crop = image[ymin:ymax, xmin:xmax]
            bg_hex = get_background_color_hex(crop)
            # Parse RGB background for text color distance analysis
            bg_r = int(bg_hex[1:3], 16)
            bg_g = int(bg_hex[3:5], 16)
            bg_b = int(bg_hex[5:7], 16)
            text_hex = analyze_text_color(crop, (bg_b, bg_g, bg_r))

            aspect = (xmax - xmin) / max(1, (ymax - ymin))
            is_bubble = not (aspect > 4.0 or aspect < 0.15)
            # Aspect-ratio de-biasing for wide narration banners with uniform background
            if aspect > 4.0 and crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                h_c, w_c = gray.shape[:2]
                if h_c >= 4 and w_c >= 4:
                    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
                    border_std = float(np.std(border_pixels))
                    if border_std < 22.0 and (b.get("conf", 1.0) >= 0.20 or len(b.get("text", "").strip()) >= 3):
                        is_bubble = True

            box_h = ymax - ymin
            box_w = xmax - xmin
            if box_h > box_w * 1.5:
                ocr_direction = TextDirection.VERTICAL.value
            elif box_w >= box_h:
                ocr_direction = TextDirection.HORIZONTAL.value
            else:
                ocr_direction = TextDirection.AUTO.value  # borderline – let renderer decide

            block = TranslationBlock.from_pixel_box(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                img_width=w_img,
                img_height=h_img,
                original_text=b["text"],
                bg_color=bg_hex,
                text_color=text_hex,
                confidence=b["conf"],
                line_count=b.get("line_count", 1),
                type=BlockType.BUBBLE.value if is_bubble else BlockType.ONOMATOPOEIA.value,
                direction=ocr_direction,
                polygon=b.get("polygon"),
                angle=float(b.get("angle", 0.0) or 0.0)
            )
            blocks.append(block)

        if progress_callback:
            progress_callback(85, "正在进行漫画阅读流空间排序...")

        from app.core.models import ReadingOrderMode
        reading_mode = ReadingOrderMode.WESTERN_LTR.value if is_english else ReadingOrderMode.MANGA_RTL.value
        sorted_blocks = sort_reading_order(blocks, mode=reading_mode)

        if progress_callback:
            progress_callback(100, f"OCR 提取完成，共识别 {len(sorted_blocks)} 个文本气泡")

        return sorted_blocks
