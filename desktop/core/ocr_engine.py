import os
import sys
import math
import torch  # CRITICAL WINDOWS DLL PROTECTION: Must import torch before paddle
import uuid
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

try:
    from app.core.ocr.base import merge_adjacent_boxes, calculate_polygon_angle
    from app.core.ocr.reading_order import sort_reading_order
    from app.core.models import TranslationBlock, ReadingOrderMode
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from app.core.ocr.base import merge_adjacent_boxes, calculate_polygon_angle
    from app.core.ocr.reading_order import sort_reading_order
    from app.core.models import TranslationBlock, ReadingOrderMode

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



def compute_box_iou(b1: Dict[str, Any], b2: Dict[str, Any]) -> Tuple[float, float]:
    """
    Computes Intersection over Union (IoU) and Coverage ratio between two bounding boxes.
    Coverage is defined as inter_area / min(area1, area2).
    """
    x1 = max(float(b1.get('xmin', 0)), float(b2.get('xmin', 0)))
    y1 = max(float(b1.get('ymin', 0)), float(b2.get('ymin', 0)))
    x2 = min(float(b1.get('xmax', 0)), float(b2.get('xmax', 0)))
    y2 = min(float(b1.get('ymax', 0)), float(b2.get('ymax', 0)))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0, 0.0
    a1 = max(1.0, float(b1.get('xmax', 0) - b1.get('xmin', 0)) * float(b1.get('ymax', 0) - b1.get('ymin', 0)))
    a2 = max(1.0, float(b2.get('xmax', 0) - b2.get('xmin', 0)) * float(b2.get('ymax', 0) - b2.get('ymin', 0)))
    union = a1 + a2 - inter
    iou = inter / max(1.0, union)
    coverage = inter / min(a1, a2)
    return float(iou), float(coverage)


def fuse_detected_boxes(
    primary_boxes: List[Dict[str, Any]],
    secondary_boxes: List[Dict[str, Any]],
    iou_thresh: float = 0.35,
    coverage_thresh: float = 0.60,
    min_w: int = 12,
    min_h: int = 10
) -> List[Dict[str, Any]]:
    """
    Fuses candidate detection boxes from a primary detector (e.g. CTD) and a secondary detector (e.g. EasyOCR).
    - If a secondary box overlaps with an existing primary box (IoU >= iou_thresh or Coverage >= coverage_thresh),
      it is considered already captured, preserving the primary box's geometry (polygons etc.) while attaching
      secondary text/conf for recognition arbitration.
    - If a secondary box has no match and meets minimum dimensions, it is appended as a newly recovered text box.
    """
    fused: List[Dict[str, Any]] = [dict(b) for b in primary_boxes]
    for s_box in secondary_boxes:
        w = s_box.get('xmax', 0) - s_box.get('xmin', 0)
        h = s_box.get('ymax', 0) - s_box.get('ymin', 0)
        if w < min_w or h < min_h:
            continue

        matched_idx = -1
        best_cov = 0.0
        for idx, p_box in enumerate(fused):
            iou, cov = compute_box_iou(p_box, s_box)
            if iou >= iou_thresh or cov >= coverage_thresh:
                if cov > best_cov:
                    best_cov = cov
                    matched_idx = idx

        if matched_idx >= 0:
            # Attach secondary detector's text & conf to matched primary box if available
            p_box = fused[matched_idx]
            if "sec_text" not in p_box and s_box.get("text"):
                p_box["sec_text"] = s_box.get("text", "")
                p_box["sec_conf"] = float(s_box.get("conf", 0.0))
        else:
            # Recovered box: ensure standard fields
            new_box = dict(s_box)
            if "polygon" not in new_box:
                x1, y1 = new_box.get("xmin", 0), new_box.get("ymin", 0)
                x2, y2 = new_box.get("xmax", 0), new_box.get("ymax", 0)
                new_box["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            if "angle" not in new_box:
                new_box["angle"] = 0.0
            fused.append(new_box)

    return fused


def ensemble_recognize_text(
    text_pri: str,
    conf_pri: float,
    text_sec: str,
    conf_sec: float,
    target_lang: str = "japan"
) -> Tuple[str, float]:
    """
    Arbitrates and fuses text recognition results from primary (e.g. Manga-OCR) and secondary (e.g. EasyOCR) engines.
    Prevents Manga-OCR Kana hallucinations on alphanumeric strings (names, numbers, English SFX)
    while preserving genuine Japanese Kana/Kanji and Chinese characters.
    """
    t1 = (text_pri or "").strip()
    t2 = (text_sec or "").strip()
    if not t1 and not t2:
        return "", 0.0
    if not t1:
        return t2, float(conf_sec)
    if not t2:
        return t1, float(conf_pri)
    if t1 == t2:
        return t1, min(1.0, max(float(conf_pri), float(conf_sec)) + 0.05)

    is_japanese = any(w in str(target_lang).lower() for w in ["japan", "ja"])
    is_chinese = any(w in str(target_lang).lower() for w in ["ch", "chinese", "zh"])
    is_english = any(w in str(target_lang).lower() for w in ["en", "eng", "english"])

    def is_cjk(c: str) -> bool:
        return ('\u4e00' <= c <= '\u9fff') or ('\u3040' <= c <= '\u309f') or ('\u30a0' <= c <= '\u30ff')

    cjk_cnt1 = sum(1 for c in t1 if is_cjk(c))
    cjk_cnt2 = sum(1 for c in t2 if is_cjk(c))
    latin_letters_cnt1 = sum(1 for c in t1 if ('a' <= c <= 'z' or 'A' <= c <= 'Z'))
    latin_letters_cnt2 = sum(1 for c in t2 if ('a' <= c <= 'z' or 'A' <= c <= 'Z'))
    alphanumeric_cnt1 = sum(1 for c in t1 if (c.isalnum() or c in ".,!?'\"-()") and not is_cjk(c))
    alphanumeric_cnt2 = sum(1 for c in t2 if (c.isalnum() or c in ".,!?'\"-()") and not is_cjk(c))

    # Case 1: Alphanumeric protection against Manga-OCR Japanese kana hallucination
    # e.g., t1="アリス(19)" or "フレンドA" or "ハハ" when original comic text is pure English "Chris (19)" or "Friend A" or "Haha"
    # Secondary recognizer detects clean English/numbers with 0 CJK characters:
    if latin_letters_cnt2 >= 2 and cjk_cnt2 == 0 and conf_sec >= 0.75:
        # If primary has no CJK at all:
        if cjk_cnt1 == 0:
            return t2, float(conf_sec)
        # If primary mixed kana with alphanumeric (e.g. "アリス(19)" having "(19)", or "フレンドA" having "A"):
        if alphanumeric_cnt1 >= 1:
            return t2, float(conf_sec)
        # If secondary is clearly a known English word or has higher confidence than primary:
        if conf_sec >= conf_pri or latin_letters_cnt2 >= 4:
            return t2, float(conf_sec)

    # Case 2: Genuine CJK protection for Japanese/Chinese manga
    # Manga-OCR is vastly superior for authentic Japanese kana/kanji over EasyOCR:
    if (is_japanese or is_chinese) and cjk_cnt1 > 0:
        # If secondary has no CJK or lower confidence, trust Manga-OCR:
        if cjk_cnt2 == 0 or conf_pri >= conf_sec:
            return t1, float(conf_pri)

    # Case 3: English target language
    if is_english:
        if latin_letters_cnt2 > latin_letters_cnt1 and conf_sec >= 0.4:
            return t2, float(conf_sec)
        if latin_letters_cnt1 >= latin_letters_cnt2:
            return t1, float(conf_pri)

    # Case 4: Significant confidence gap (> 0.25)
    if conf_sec > conf_pri + 0.25:
        return t2, float(conf_sec)
    if conf_pri > conf_sec + 0.25:
        return t1, float(conf_pri)

    # Default to primary model
    return t1, float(conf_pri)


def get_background_color_hex(crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return "#FFFFFF"
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return "#FFFFFF"
    border_pixels = []
    border_pixels.extend(crop[0, :])
    border_pixels.extend(crop[h - 1, :])
    if h > 2:
        border_pixels.extend(crop[1:h - 1, 0])
        border_pixels.extend(crop[1:h - 1, w - 1])
    border_pixels = np.array(border_pixels)
    if len(border_pixels) == 0:
        return "#FFFFFF"
    median_color = np.median(border_pixels, axis=0).astype(int)
    # BGR to RGB
    b, g, r = median_color[0], median_color[1], median_color[2]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}".upper()


def mask_crop_with_lines(crop: np.ndarray, box: Dict[str, Any], bx1: int, by1: int) -> np.ndarray:
    """
    Masks out pixels outside the segmented CTD text lines with the background color.
    Prevents recognition bleed where axis-aligned bounding box corners contain neighboring text.
    """
    if crop is None or crop.size == 0:
        return crop
    lines = box.get("lines")
    if not lines or len(lines) == 0:
        return crop
    try:
        masked_crop = crop.copy()
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        for ln in lines:
            pts = np.asarray(ln, dtype=np.int32) - np.array([bx1, by1], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)))
        bg_hex = box.get("bg_color") or get_background_color_hex(crop)
        c_str = bg_hex.lstrip("#")
        if len(c_str) == 6:
            r_c, g_c, b_c = int(c_str[0:2], 16), int(c_str[2:4], 16), int(c_str[4:6], 16)
        else:
            r_c, g_c, b_c = 255, 255, 255
        masked_crop[mask == 0] = [b_c, g_c, r_c]
        return masked_crop
    except Exception:
        return crop


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

    # Initial sort by vertical position
    valid_items.sort(key=lambda item: item["ymin"])

    # Cluster into horizontal text lines (rows)
    rows: List[List[Dict[str, Any]]] = []
    row_bounds: List[Tuple[float, float, float]] = []  # (ymin, ymax, h)

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

    # Sort each row left-to-right by xmin
    for r in rows:
        r.sort(key=lambda item: item["xmin"])

    # Sort rows top-to-bottom by row_ymin
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
    Expands bounding box to prevent clipping characters.
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


class OCREngine:
    def __init__(
        self,
        engine_type: str = "paddle",
        use_gpu: bool = True,
        lang: str = "japan",
        reading_direction: Optional[str] = None,
        enable_ensemble_detection: bool = False,
        enable_ensemble_recognition: bool = False
    ):
        self.engine_type = engine_type
        if self.engine_type == "cpu_paddle":
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu
        self.lang = lang
        self.reading_direction = reading_direction
        self.enable_ensemble_detection = enable_ensemble_detection
        self.enable_ensemble_recognition = enable_ensemble_recognition
        self._paddle_ocr = None
        self._easyocr_reader = None
        self._easyocr_en_reader = None
        self._manga_ocr = None
        self._ctd_detector = None

    def _init_ctd(self):
        if self._ctd_detector is None:
            from app.core.ocr.ctd_engine import ComicTextDetectorEngine
            self._ctd_detector = ComicTextDetectorEngine(use_gpu=self.use_gpu)

    def _init_paddle(self):
        if self._paddle_ocr is None:
            import os
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["FLAGS_use_onednn"] = "0"
            os.environ["FLAGS_use_mkldnn"] = "0"

            has_paddle_cuda = False
            try:
                import paddle
                paddle.set_flags({"FLAGS_use_onednn": False})
                paddle.set_num_threads(1)
                has_paddle_cuda = paddle.device.is_compiled_with_cuda() and (paddle.device.cuda.device_count() > 0)
            except Exception:
                pass

            from paddleocr import PaddleOCR
            is_paddleocr_3x = hasattr(PaddleOCR, "_paddlex_pipeline_name")

            if is_paddleocr_3x:
                # PaddleOCR 3.x / PaddleX pipeline
                can_use_gpu = bool(self.use_gpu and has_paddle_cuda)
                if self.use_gpu and not has_paddle_cuda:
                    print("[*] PaddlePaddle CUDA not available in current environment. Using CPU mode.")
                    self.use_gpu = False

                device_str = "gpu" if can_use_gpu else "cpu"
                print(f"[*] Initializing PaddleOCR 3.x (lang={self.lang}, device={device_str}, model=PP-OCRv3 Mobile)...")

                kwargs = {
                    "lang": self.lang,
                    "ocr_version": "PP-OCRv3",  # 默认使用轻量级 Mobile 模型，内存仅占 ~150MB，避免 Medium 模型耗尽系统内存导致死机
                    "device": device_str,
                    "cpu_threads": 2,
                    "use_textline_orientation": True,
                    "enable_mkldnn": False,  # 禁用 oneDNN 规避 Windows CPU PIR double attribute 转换异常
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                }

                try:
                    self._paddle_ocr = PaddleOCR(**kwargs)
                except Exception as e:
                    if device_str == "gpu":
                        print(f"[!] Warning: PaddleOCR GPU init failed: {e}. Retrying CPU mode...")
                        self.use_gpu = False
                        kwargs["device"] = "cpu"
                        kwargs.pop("ocr_version", None)
                        self._paddle_ocr = PaddleOCR(**kwargs)
                    else:
                        raise
            else:
                # PaddleOCR 2.x legacy or unit test mock
                use_gpu_flag = bool(self.use_gpu)
                device_str = "gpu" if use_gpu_flag else "cpu"
                print(f"[*] Initializing PaddleOCR (lang={self.lang}, device={device_str}, use_gpu={use_gpu_flag})...")
                try:
                    self._paddle_ocr = PaddleOCR(
                        lang=self.lang,
                        device=device_str,
                        use_gpu=use_gpu_flag,
                        use_textline_orientation=True
                    )
                except Exception as e:
                    print(f"[!] Warning: PaddleOCR init with use_gpu={use_gpu_flag} failed: {e}. Retrying CPU mode...")
                    self.use_gpu = False
                    try:
                        self._paddle_ocr = PaddleOCR(
                            lang=self.lang,
                            device="cpu",
                            use_gpu=False,
                            use_textline_orientation=True
                        )
                    except Exception as e2:
                        print(f"[!] Warning: PaddleOCR CPU init failed: {e2}. Fallback to generic init...")
                        self._paddle_ocr = PaddleOCR(lang=self.lang)

    def _init_easyocr(self, is_english: bool = False):
        if is_english:
            if getattr(self, "_easyocr_en_reader", None) is None:
                import easyocr
                print(f"[*] Initializing EasyOCR for English (langs=['en'], gpu={self.use_gpu})...")
                self._easyocr_en_reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        else:
            if self._easyocr_reader is None:
                import easyocr
                langs = ['ja', 'en'] if self.lang in ['japan', 'ja'] else ['ch_sim', 'en']
                print(f"[*] Initializing EasyOCR (langs={langs}, gpu={self.use_gpu})...")
                self._easyocr_reader = easyocr.Reader(langs, gpu=self.use_gpu)

    def _get_easyocr_reader(self, is_english: bool = False):
        if is_english:
            if getattr(self, "_easyocr_en_reader", None) is not None:
                return self._easyocr_en_reader
            if getattr(self, "_easyocr_reader", None) is not None:
                return self._easyocr_reader
            self._init_easyocr(is_english=True)
            return self._easyocr_en_reader
        else:
            if self._easyocr_reader is None:
                self._init_easyocr(is_english=False)
            return self._easyocr_reader

    def detect_page_language(
        self,
        image: np.ndarray,
        candidate_boxes: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str]:
        """
        Detects whether the image is an English manga page:
        1. If self.lang explicitly specifies English, returns (True, 'en').
        2. Fast sampling on candidate boxes or whole page using English EasyOCR.
        3. Returns (is_english, detected_lang).
        """
        if any(w in str(self.lang).lower() for w in ["en", "eng", "english", "latin"]):
            return True, "en"

        reader = self._get_easyocr_reader(is_english=True)
        sampled_texts = []

        if candidate_boxes:
            sorted_boxes = sorted(
                candidate_boxes,
                key=lambda b: (b.get("xmax", 0) - b.get("xmin", 0)) * (b.get("ymax", 0) - b.get("ymin", 0)),
                reverse=True
            )
            for b in sorted_boxes[:4]:
                w_b = b.get("xmax", 0) - b.get("xmin", 0)
                h_b = b.get("ymax", 0) - b.get("ymin", 0)
                if w_b >= 30 and h_b >= 20:
                    bx1 = max(0, min(b["xmin"], image.shape[1] - 1))
                    by1 = max(0, min(b["ymin"], image.shape[0] - 1))
                    bx2 = max(bx1 + 1, min(b["xmax"], image.shape[1]))
                    by2 = max(by1 + 1, min(b["ymax"], image.shape[0]))
                    crop = image[by1:by2, bx1:bx2]
                    if crop.size > 0:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        res = reader.readtext(rgb)
                        for r in res:
                            if len(r) > 1 and str(r[1]).strip():
                                sampled_texts.append(str(r[1]).strip())

        if not sampled_texts and image is not None and image.size > 0:
            h, w = image.shape[:2]
            scale = 1024.0 / max(h, w) if max(h, w) > 1024 else 1.0
            small_img = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image
            rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
            res = reader.readtext(rgb)
            for r in res:
                if len(r) > 1 and str(r[1]).strip():
                    sampled_texts.append(str(r[1]).strip())

        # If candidate boxes already contain authentic CJK text, it's definitely not English
        if candidate_boxes:
            for b in candidate_boxes:
                b_text = str(b.get("text", ""))
                if any(('\u4e00' <= c <= '\u9fff') or ('\u3040' <= c <= '\u309f') or ('\u30a0' <= c <= '\u30ff') for c in b_text):
                    return False, str(self.lang)

        combined_text = " ".join(sampled_texts)
        import re
        latin_words = re.findall(r'[a-zA-Z]{2,}', combined_text)
        distinct_words = set(w.lower() for w in latin_words)
        cjk_chars = [
            c for c in combined_text
            if ('\u4e00' <= c <= '\u9fff') or ('\u3040' <= c <= '\u309f') or ('\u30a0' <= c <= '\u30ff')
        ]
        total_latin = sum(len(w) for w in latin_words)
        total_cjk = len(cjk_chars)

        # Require substantial English evidence (multiple distinct words and characters)
        if (len(distinct_words) >= 3 and total_latin >= 10 and total_latin > total_cjk * 2) or \
           (len(distinct_words) >= 2 and total_latin >= 8 and total_cjk == 0 and len(latin_words) >= 3):
            return True, "en"

        return False, str(self.lang)

    def _read_block_easyocr(self, crop: np.ndarray, angle: float = 0.0, is_english: bool = True) -> Tuple[str, float]:
        """
        Reads text in a cropped dialogue or narration box:
        - If slanted (abs(angle) >= 2.5), affine rotates crop upright before recognition.
        - Sorts all detected fragments in 2D reading order (top-to-bottom, left-to-right).
        - Cleans up trailing colons or broken words.
        """
        if crop is None or crop.size == 0:
            return "", 0.0

        reader = self._get_easyocr_reader(is_english=is_english)

        if abs(angle) >= 2.5:
            crop_upright, _ = rotate_crop_upright(crop, angle)
            rgb_upright = cv2.cvtColor(crop_upright, cv2.COLOR_BGR2RGB)
            res_upright = reader.readtext(rgb_upright)
            text_up, conf_up = sort_easyocr_fragments_2d(res_upright)

            rgb_raw = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res_raw = reader.readtext(rgb_raw)
            text_raw, conf_raw = sort_easyocr_fragments_2d(res_raw)

            if len(text_up.strip()) >= len(text_raw.strip()) and conf_up >= 0.25:
                text, conf = text_up, conf_up
            elif len(text_raw.strip()) > 0:
                text, conf = text_raw, conf_raw
            else:
                text, conf = text_up, conf_up
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = reader.readtext(rgb)
            text, conf = sort_easyocr_fragments_2d(res)

        text = clean_ocr_syntax(text)
        return text, conf

    def detect_and_recognize(self, image: np.ndarray, progress_callback=None, reading_direction: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Runs OCR on an OpenCV BGR image and returns a list of translation blocks.
        Each block: {
            "id": str,
            "original_text": str,
            "translated_text": str,
            "xmin": float (0-100),
            "ymin": float (0-100),
            "xmax": float (0-100),
            "ymax": float (0-100),
            "bg_color": str (#HEX),
            "text_color": str (#HEX),
            "type": "bubble" | "onomatopoeia"
        }
        """
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        raw_boxes = []
        page_is_english = any(w in str(self.lang).lower() for w in ["en", "eng", "english", "latin"])

        if progress_callback:
            progress_callback(10, "正在加载 OCR 识别引擎...")

        # 1. Run detection
        if self.engine_type in ("ctd", "pure_pytorch", "manga_ocr"):
            try:
                self._init_ctd()
                if progress_callback:
                    progress_callback(30, "正在使用 Comic-Text-Detector 提取漫画气泡 (纯 PyTorch)...")
                raw_boxes, _ = self._ctd_detector.detect(image)
            except Exception as e_ctd:
                print(f"[-] Comic-Text-Detector 未能运行: {e_ctd}，自动回退到 EasyOCR...")
                raw_boxes = []

            # Dual-Model Detection Fusion (CTD + General detector)
            if self.enable_ensemble_detection:
                try:
                    self._init_easyocr()
                    if progress_callback:
                        progress_callback(40, "正在运行辅助检测器进行双模型联合找框...")
                    rgb_full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    easy_results = self._easyocr_reader.readtext(rgb_full)
                    sec_boxes = []
                    for bbox, t_text, t_conf in easy_results:
                        if not str(t_text).strip() or t_conf < 0.2:
                            continue
                        pts = np.array(bbox, dtype=np.int32)
                        sec_boxes.append({
                            "xmin": int(np.min(pts[:, 0])),
                            "ymin": int(np.min(pts[:, 1])),
                            "xmax": int(np.max(pts[:, 0])),
                            "ymax": int(np.max(pts[:, 1])),
                            "text": str(t_text).strip(),
                            "conf": float(t_conf),
                            "polygon": pts.astype(int).tolist(),
                            "angle": calculate_polygon_angle(pts)
                        })
                    raw_boxes = fuse_detected_boxes(raw_boxes, sec_boxes)
                    if progress_callback:
                        progress_callback(48, f"双模型找框完成，融合候选框共 {len(raw_boxes)} 个")
                except Exception as e_ens_det:
                    print(f"[-] 双模型联合找框异常: {e_ens_det}，保持主模型检测结果")

            # If CTD produced candidate boxes, perform smart recognition
            if raw_boxes:
                page_is_english, _ = self.detect_page_language(image, raw_boxes)

                if page_is_english:
                    if progress_callback:
                        progress_callback(55, "探知为英文漫画，自动切换 EasyOCR 准确提取英文 (硬禁用 Manga-OCR)...")
                    # Hard-disable Manga-OCR to strictly eliminate Japanese kana/hiragana/katakana hallucinations
                    self._manga_ocr = None
                    for box in raw_boxes:
                        bx1 = max(0, min(box["xmin"], w_img - 1))
                        by1 = max(0, min(box["ymin"], h_img - 1))
                        bx2 = max(bx1 + 1, min(box["xmax"], w_img))
                        by2 = max(by1 + 1, min(box["ymax"], h_img))
                        crop = image[by1:by2, bx1:bx2]
                        if crop.size > 0:
                            crop = mask_crop_with_lines(crop, box, bx1, by1)
                            txt, conf = self._read_block_easyocr(crop, angle=box.get("angle", 0.0), is_english=True)
                            if txt:
                                box["text"] = txt
                                box["conf"] = conf
                else:
                    if self._manga_ocr is None:
                        from app.core.ocr.manga_ocr_wrapper import get_manga_ocr
                        self._manga_ocr = get_manga_ocr(force_cpu=not self.use_gpu)
                    if progress_callback:
                        progress_callback(55, "正在使用 Manga-OCR 高精度识别日文...")
                    for box in raw_boxes:
                        bx1 = max(0, min(box["xmin"], w_img - 1))
                        by1 = max(0, min(box["ymin"], h_img - 1))
                        bx2 = max(bx1 + 1, min(box["xmax"], w_img))
                        by2 = max(by1 + 1, min(box["ymax"], h_img))
                        crop = image[by1:by2, bx1:bx2]
                        if crop.size > 0:
                            crop = mask_crop_with_lines(crop, box, bx1, by1)
                        txt_pri = self._manga_ocr.recognize_crop(crop, angle=box.get("angle", 0.0))
                        conf_pri = max(float(box.get("conf", 0.8)), 0.95) if txt_pri else 0.0

                        if self.enable_ensemble_recognition:
                            txt_sec = box.get("sec_text", "")
                            conf_sec = float(box.get("sec_conf", 0.0))
                            if not txt_sec and crop.size > 0:
                                try:
                                    self._init_easyocr(is_english=False)
                                    rgb_c = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    sec_res = self._easyocr_reader.readtext(rgb_c)
                                    if sec_res:
                                        txt_sec, conf_sec = sort_easyocr_fragments_2d(sec_res)
                                except Exception:
                                    pass
                            final_t, final_c = ensemble_recognize_text(
                                txt_pri, conf_pri, txt_sec, conf_sec, target_lang=self.lang
                            )
                            if final_t:
                                box["text"] = final_t
                                box["conf"] = final_c
                        else:
                            if txt_pri:
                                box["text"] = txt_pri
                                box["conf"] = conf_pri

                raw_boxes = [b for b in raw_boxes if b.get("text", "").strip()]
            else:
                # Fallback to EasyOCR
                try:
                    self._init_easyocr()
                    if progress_callback:
                        progress_callback(30, "正在使用 EasyOCR 回退分析图像文字...")
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self._easyocr_reader.readtext(rgb)
                    for bbox, text, conf in results:
                        if not text.strip() or conf < 0.2:
                            continue
                        pts = np.array(bbox, dtype=np.int32)
                        xmin = int(np.min(pts[:, 0]))
                        ymin = int(np.min(pts[:, 1]))
                        xmax = int(np.max(pts[:, 0]))
                        ymax = int(np.max(pts[:, 1]))
                        raw_boxes.append({
                            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                            "text": text.strip(), "conf": float(conf),
                            "polygon": pts.astype(int).tolist(),
                            "angle": calculate_polygon_angle(pts)
                        })
                except Exception as e_easy:
                    print(f"[-] EasyOCR 回退识别亦发生异常: {e_easy}")
        elif self.engine_type == "easyocr":
            try:
                self._init_easyocr()
                if progress_callback:
                    progress_callback(30, "正在使用 EasyOCR 分析图像文字...")
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self._easyocr_reader.readtext(rgb)
                for bbox, text, conf in results:
                    if not text.strip() or conf < 0.2:
                        continue
                    pts = np.array(bbox, dtype=np.int32)
                    xmin = int(np.min(pts[:, 0]))
                    ymin = int(np.min(pts[:, 1]))
                    xmax = int(np.max(pts[:, 0]))
                    ymax = int(np.max(pts[:, 1]))
                    raw_boxes.append({
                        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                        "text": text.strip(), "conf": float(conf),
                        "polygon": pts.astype(int).tolist(),
                        "angle": calculate_polygon_angle(pts)
                    })
            except Exception as e:
                print(f"[-] EasyOCR error: {e}")
        else:
            # PaddleOCR default
            try:
                self._init_paddle()
                if progress_callback:
                    progress_callback(30, "正在使用 PaddleOCR 识别漫画对话...")
                
                # PaddleOCR inference
                results = self._paddle_ocr.ocr(image)
                if results and len(results) > 0 and results[0]:
                    if isinstance(results[0], dict):
                        # PaddleOCR 3.x (PaddleX pipeline) format
                        res_dict = results[0]
                        rec_texts = res_dict.get('rec_texts', [])
                        rec_scores = res_dict.get('rec_scores', [])
                        rec_polys = res_dict.get('rec_polys', [])
                        if (rec_polys is None or len(rec_polys) == 0) and 'dt_polys' in res_dict:
                            rec_polys = res_dict.get('dt_polys', [])

                        for i in range(len(rec_texts)):
                            text = str(rec_texts[i]).strip()
                            conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                            if not text or conf < 0.25:
                                continue
                            poly = rec_polys[i] if (rec_polys is not None and i < len(rec_polys)) else None
                            if poly is not None and len(poly) >= 4:
                                pts = np.array(poly, dtype=np.int32)
                                xmin = int(np.min(pts[:, 0]))
                                ymin = int(np.min(pts[:, 1]))
                                xmax = int(np.max(pts[:, 0]))
                                ymax = int(np.max(pts[:, 1]))
                                raw_boxes.append({
                                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                                    "text": text, "conf": conf,
                                    "polygon": pts.astype(int).tolist(),
                                    "angle": calculate_polygon_angle(pts)
                                })
                            elif 'rec_boxes' in res_dict and i < len(res_dict['rec_boxes']):
                                box = res_dict['rec_boxes'][i]
                                bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                                raw_boxes.append({
                                    "xmin": bx1, "ymin": by1,
                                    "xmax": bx2, "ymax": by2,
                                    "text": text, "conf": conf,
                                    "polygon": [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                                    "angle": 0.0
                                })
                    elif isinstance(results[0], list):
                        # PaddleOCR 2.x list format
                        for line in results[0]:
                            try:
                                pts = np.array(line[0], dtype=np.int32)
                                text_info = line[1]
                                text = text_info[0]
                                conf = float(text_info[1])
                                if not text.strip() or conf < 0.25:
                                    continue
                                xmin = int(np.min(pts[:, 0]))
                                ymin = int(np.min(pts[:, 1]))
                                xmax = int(np.max(pts[:, 0]))
                                ymax = int(np.max(pts[:, 1]))
                                raw_boxes.append({
                                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                                    "text": text.strip(), "conf": conf,
                                    "polygon": pts.astype(int).tolist(),
                                    "angle": calculate_polygon_angle(pts)
                                })
                            except Exception:
                                continue
            except Exception as e:
                print(f"[-] PaddleOCR 发生异常: {e}。正在自动无缝回退至 EasyOCR 引擎...")
                try:
                    self._init_easyocr()
                    if progress_callback:
                        progress_callback(40, "PaddleOCR 异常，已自动切换为 EasyOCR 识别...")
                    res = self._easyocr_reader.readtext(image)
                    for bbox, text, conf in res:
                        if not text.strip() or conf < 0.25:
                            continue
                        pts = np.array(bbox, dtype=np.int32)
                        xmin = int(np.min(pts[:, 0]))
                        ymin = int(np.min(pts[:, 1]))
                        xmax = int(np.max(pts[:, 0]))
                        ymax = int(np.max(pts[:, 1]))
                        raw_boxes.append({
                            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                            "text": text.strip(), "conf": float(conf),
                            "polygon": pts.astype(int).tolist(),
                            "angle": calculate_polygon_angle(pts)
                        })
                except Exception as e_easy:
                    print(f"[-] EasyOCR 回退识别亦发生异常: {e_easy}")

        # Decoupled recognition / Recognition ensemble: Manga-OCR for Paddle Japanese text crops
        if not page_is_english and (self.engine_type == "paddle_manga" or (self.engine_type == "paddle" and self.enable_ensemble_recognition)) and any(w in str(self.lang).lower() for w in ["japan", "ja"]):
            try:
                if self._manga_ocr is None:
                    from app.core.ocr.manga_ocr_wrapper import get_manga_ocr
                    self._manga_ocr = get_manga_ocr(force_cpu=not self.use_gpu)
                if progress_callback:
                    progress_callback(55, "正在使用 Manga-OCR 高精度识别日文...")
                for box in raw_boxes:
                    bx1 = max(0, min(box["xmin"], w_img - 1))
                    by1 = max(0, min(box["ymin"], h_img - 1))
                    bx2 = max(bx1 + 1, min(box["xmax"], w_img))
                    by2 = max(by1 + 1, min(box["ymax"], h_img))
                    crop = image[by1:by2, bx1:bx2]
                    txt = self._manga_ocr.recognize_crop(crop, angle=box.get("angle", 0.0))
                    if self.enable_ensemble_recognition:
                        paddle_txt = box.get("text", "")
                        paddle_conf = float(box.get("conf", 0.8))
                        final_t, final_c = ensemble_recognize_text(txt, 0.95, paddle_txt, paddle_conf, target_lang=self.lang)
                        if final_t:
                            box["text"] = final_t
                            box["conf"] = final_c
                    else:
                        if txt:
                            box["text"] = txt
                            box["conf"] = max(float(box.get("conf", 0.8)), 0.95)
            except Exception as e:
                print(f"[-] Manga-OCR decoupled/ensemble recognition failed, keeping default text: {e}")

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
            progress_callback(70, "正在智能聚合气泡与段落排版...")

        # 2. Merge overlapping / closely adjacent text lines in speech bubbles
        merged_boxes = self._merge_adjacent_boxes(raw_boxes, w_img, h_img, image=image, qr_regions=qr_regions)

        # 3. Convert to TranslationBlock objects for high-precision sorting & coordinate normalization
        tb_blocks: List[TranslationBlock] = []
        for box in merged_boxes:
            xmin = max(0, min(box["xmin"], w_img - 1))
            ymin = max(0, min(box["ymin"], h_img - 1))
            xmax = max(0, min(box["xmax"], w_img))
            ymax = max(0, min(box["ymax"], h_img))
            if xmax <= xmin or ymax <= ymin:
                continue

            crop = image[ymin:ymax, xmin:xmax]
            bg_color = get_background_color_hex(crop)
            r_val = int(bg_color[1:3], 16) if len(bg_color) >= 7 else 255
            g_val = int(bg_color[3:5], 16) if len(bg_color) >= 7 else 255
            b_val = int(bg_color[5:7], 16) if len(bg_color) >= 7 else 255

            try:
                from app.core.inpaint.color_analyzer import analyze_text_color
                detected_text_color = analyze_text_color(crop, (b_val, g_val, r_val))
            except Exception:
                bg_lum = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
                detected_text_color = "#FFFFFF" if bg_lum < 128 else "#000000"

            # Determine type (bubble vs onomatopoeia) based on aspect ratio & background
            is_bubble = True
            aspect = (xmax - xmin) / max(1, (ymax - ymin))
            if aspect > 4.0 or aspect < 0.15:
                is_uniform_banner = False
                if aspect > 4.0 and crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    h_c, w_c = gray.shape[:2]
                    if h_c >= 4 and w_c >= 4:
                        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
                        border_std = float(np.std(border_pixels))
                        conf = float(box.get("conf", 1.0))
                        text_str = str(box.get("text", "")).strip()
                        if border_std < 22.0 and (conf >= 0.20 or len(text_str) >= 3):
                            is_uniform_banner = True
                if not is_uniform_banner:
                    is_bubble = False

            tb = TranslationBlock(
                id=str(uuid.uuid4())[:8],
                original_text=box["text"],
                translated_text="",
                xmin=round((xmin / w_img) * 100.0, 2),
                ymin=round((ymin / h_img) * 100.0, 2),
                xmax=round((xmax / w_img) * 100.0, 2),
                ymax=round((ymax / h_img) * 100.0, 2),
                polygon=box.get("polygon"),
                bg_color=bg_color,
                text_color=detected_text_color,
                type="bubble" if is_bubble else "onomatopoeia",
                confidence=float(box.get("conf", 1.0)),
                line_count=int(box.get("line_count", 1)),
                angle=float(box.get("angle", 0.0) or 0.0)
            )
            tb_blocks.append(tb)

        # 4. Resolve reading order mode based on language / comic format
        eff_direction = reading_direction or getattr(self, "reading_direction", None)
        if eff_direction:
            resolved_mode = eff_direction
        elif page_is_english or any(w in str(self.lang).lower() for w in ["en", "eng", "english", "latin"]):
            resolved_mode = ReadingOrderMode.WESTERN_LTR.value
        elif (h_img / max(1, w_img)) >= 2.2:
            resolved_mode = ReadingOrderMode.WEBTOON_TTB.value
        else:
            resolved_mode = ReadingOrderMode.MANGA_RTL.value

        sorted_tb_blocks = sort_reading_order(tb_blocks, mode=resolved_mode)
        blocks = [b.to_dict() for b in sorted_tb_blocks]

        if progress_callback:
            progress_callback(95, f"OCR 识别完成，共提取 {len(blocks)} 个对话气泡")

        return blocks

    def _merge_adjacent_boxes(
        self,
        boxes: List[Dict[str, Any]],
        w_img: int,
        h_img: int,
        image: Optional[np.ndarray] = None,
        qr_regions: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        return merge_adjacent_boxes(boxes, w_img, h_img, image=image, qr_regions=qr_regions)

    def _sort_manga_reading_order(self, boxes: List[Dict[str, Any]], w_img: int) -> List[Dict[str, Any]]:
        """
        Sorts boxes Right-to-Left and Top-to-Bottom using tier interval clustering.
        """
        if not boxes:
            return []
        w_val = max(1, int(w_img)) if w_img else 1000
        max_y = max(float(b.get("ymax", 0)) for b in boxes)
        h_val = max(w_val, int(max_y) + 10)
        dummy_blocks = [
            TranslationBlock.from_pixel_box(
                float(b.get("xmin", 0)),
                float(b.get("ymin", 0)),
                float(b.get("xmax", 0)),
                float(b.get("ymax", 0)),
                w_val,
                h_val,
                id=str(idx),
                original_text=b.get("text", "")
            )
            for idx, b in enumerate(boxes)
        ]
        sorted_dummy = sort_reading_order(dummy_blocks, mode=ReadingOrderMode.MANGA_RTL.value)
        order_map = {int(b.id): rank for rank, b in enumerate(sorted_dummy)}
        return sorted(boxes, key=lambda b: order_map.get(boxes.index(b), 0))
