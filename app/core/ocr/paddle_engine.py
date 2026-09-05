"""
app/core/ocr/paddle_engine.py
PaddleOCR backend with hardware detection, Pascal GPU compatibility handling, and Windows thread limiting.
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import cv2

# CRITICAL WINDOWS DLL PROTECTION: Must import torch before paddle
import torch

from app.core.models import TranslationBlock, BlockType, TextDirection
from app.core.ocr.base import (
    BaseOCREngine,
    is_solid_color_page,
    merge_adjacent_boxes,
    calculate_polygon_angle
)
from app.core.ocr.reading_order import sort_reading_order
from app.core.hardware import is_legacy_pascal_or_maxwell_gpu
from app.core.inpaint.color_analyzer import get_background_color_hex, analyze_text_color

logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    def __init__(
        self,
        lang: str = "japan",
        force_cpu: bool = False,
        fallback_to_easyocr: bool = True,
        use_manga_ocr: bool = False
    ):
        self.lang = lang
        self.force_cpu = force_cpu
        self.fallback_to_easyocr = fallback_to_easyocr
        self.use_manga_ocr = use_manga_ocr
        self._ocr = None
        self._manga_ocr = None
        self._is_old_gpu = is_legacy_pascal_or_maxwell_gpu()

    def _get_ocr(self):
        if self._ocr is None:
            # Set thread limits and memory safety on Windows
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            import paddle
            try:
                paddle.set_flags({"FLAGS_use_onednn": False})
                paddle.set_num_threads(1)
            except Exception:
                pass

            has_paddle_cuda = (
                paddle.device.is_compiled_with_cuda()
                and (paddle.device.cuda.device_count() > 0)
                and not self.force_cpu
            )
            device_str = "gpu" if has_paddle_cuda else "cpu"

            # PaddleOCR 3.7.0 parameters:
            kwargs: Dict[str, Any] = {
                "lang": self.lang,
                "ocr_version": "PP-OCRv3",  # 默认使用轻量级 Mobile 模型 (~150MB)，防止内存耗尽导致系统假死或崩溃
                "device": device_str,
                "cpu_threads": 2,
                "use_textline_orientation": True,
                "enable_mkldnn": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False
            }

            logger.info(f"Initializing PaddleOCR (device={device_str}, kwargs={kwargs})...")
            try:
                self._ocr = PaddleOCR(**kwargs)
            except Exception as e:
                if device_str == "gpu":
                    logger.warning(f"PaddleOCR GPU init failed: {e}. Retrying with CPU mode...")
                    kwargs["device"] = "cpu"
                    kwargs.pop("ocr_version", None)
                    self._ocr = PaddleOCR(**kwargs)
                else:
                    raise
        return self._ocr

    def is_available(self) -> bool:
        try:
            import paddleocr
            import paddle
            return True
        except ImportError:
            return False

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "engine": "PaddleOCR",
            "is_legacy_gpu": self._is_old_gpu,
            "force_cpu": self.force_cpu,
            "lang": self.lang
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

        try:
            if progress_callback:
                progress_callback(15, "正在初始化 PaddleOCR 引擎...")

            ocr = self._get_ocr()

            if progress_callback:
                progress_callback(40, "正在执行漫画对话文字检测与识别...")

            results = ocr.ocr(image)
        except Exception as e:
            logger.warning(f"PaddleOCR execution failed: {e}")
            if self.fallback_to_easyocr:
                logger.info("Falling back to EasyOCREngine...")
                from app.core.ocr.easyocr_engine import EasyOCREngine
                target_lang = lang or self.lang
                lang_map = {
                    "japan": ["ja", "en"],
                    "ch": ["ch_sim", "en"],
                    "korean": ["ko", "en"],
                    "en": ["en"],
                }
                mapped_langs = lang_map.get(target_lang, [target_lang, "en"] if isinstance(target_lang, str) else ["ja", "en"])
                easy_engine = EasyOCREngine(languages=mapped_langs)
                return easy_engine.detect_and_recognize(image, lang=target_lang, progress_callback=progress_callback)
            raise

        raw_boxes = []

        if results and len(results) > 0 and results[0]:
            # Handle PaddleOCR 3.x dict format
            if isinstance(results[0], dict):
                res_dict = results[0]
                rec_texts = res_dict.get('rec_texts', [])
                rec_scores = res_dict.get('rec_scores', [])
                rec_polys = res_dict.get('rec_polys', [])
                if (rec_polys is None or len(rec_polys) == 0) and 'dt_polys' in res_dict:
                    rec_polys = res_dict.get('dt_polys', [])
                for i in range(len(rec_texts)):
                    text = str(rec_texts[i]).strip()
                    conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                    if not text or conf < 0.20:
                        continue
                    poly = rec_polys[i] if (rec_polys is not None and i < len(rec_polys)) else None
                    if poly is not None and len(poly) >= 4:
                        pts = np.array(poly, dtype=np.int32)
                        angle = calculate_polygon_angle(pts)
                        raw_boxes.append({
                            "xmin": int(np.min(pts[:, 0])),
                            "ymin": int(np.min(pts[:, 1])),
                            "xmax": int(np.max(pts[:, 0])),
                            "ymax": int(np.max(pts[:, 1])),
                            "text": text,
                            "conf": conf,
                            "polygon": pts.astype(int).tolist(),
                            "angle": angle
                        })
                    elif 'rec_boxes' in res_dict and i < len(res_dict['rec_boxes']):
                        box = res_dict['rec_boxes'][i]
                        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        raw_boxes.append({
                            "xmin": bx1, "ymin": by1,
                            "xmax": bx2, "ymax": by2,
                            "text": text,
                            "conf": conf,
                            "polygon": [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                            "angle": 0.0
                        })
            # Handle PaddleOCR 2.x list format
            elif isinstance(results[0], list):
                for line in results[0]:
                    try:
                        pts = np.array(line[0], dtype=np.int32)
                        text_info = line[1]
                        text = str(text_info[0]).strip()
                        conf = float(text_info[1])
                        if not text or conf < 0.20:
                            continue
                        angle = calculate_polygon_angle(pts)
                        raw_boxes.append({
                            "xmin": int(np.min(pts[:, 0])),
                            "ymin": int(np.min(pts[:, 1])),
                            "xmax": int(np.max(pts[:, 0])),
                            "ymax": int(np.max(pts[:, 1])),
                            "text": text,
                            "conf": conf,
                            "polygon": pts.astype(int).tolist(),
                            "angle": angle
                        })
                    except Exception:
                        continue

        # Decoupled recognition: Manga-OCR for Japanese text crops
        if self.use_manga_ocr and (self.lang == "japan" or target_lang in ("japan", "ja")):
            try:
                from app.core.ocr.manga_ocr_wrapper import get_manga_ocr
                if self._manga_ocr is None:
                    self._manga_ocr = get_manga_ocr(force_cpu=self.force_cpu)
                if progress_callback:
                    progress_callback(55, "正在使用 Manga-OCR 高精度识别日文...")
                for box in raw_boxes:
                    bx1 = max(0, min(box["xmin"], w_img - 1))
                    by1 = max(0, min(box["ymin"], h_img - 1))
                    bx2 = max(bx1 + 1, min(box["xmax"], w_img))
                    by2 = max(by1 + 1, min(box["ymax"], h_img))
                    crop = image[by1:by2, bx1:bx2]
                    txt = self._manga_ocr.recognize_crop(crop, angle=box.get("angle", 0.0))
                    if txt:
                        box["text"] = txt
                        box["conf"] = max(float(box.get("conf", 0.8)), 0.95)
            except Exception as e:
                logger.warning(f"Manga-OCR decoupled recognition failed, keeping PaddleOCR text: {e}")

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
            progress_callback(70, "正在进行气泡聚类与文本段落合并...")

        merged_boxes = merge_adjacent_boxes(raw_boxes, w_img, h_img, image=image, qr_regions=qr_regions)

        blocks: List[TranslationBlock] = []
        for b in merged_boxes:
            xmin = max(0, min(b["xmin"], w_img - 1))
            ymin = max(0, min(b["ymin"], h_img - 1))
            xmax = max(xmin + 1, min(b["xmax"], w_img))
            ymax = max(ymin + 1, min(b["ymax"], h_img))

            crop = image[ymin:ymax, xmin:xmax]
            bg_hex = get_background_color_hex(crop)
            bg_r = int(bg_hex[1:3], 16)
            bg_g = int(bg_hex[3:5], 16)
            bg_b = int(bg_hex[5:7], 16)
            text_hex = analyze_text_color(crop, (bg_b, bg_g, bg_r))

            aspect = (xmax - xmin) / max(1, (ymax - ymin))
            is_bubble = not (aspect > 4.0 or aspect < 0.15)
            is_vertical = (ymax - ymin) > (xmax - xmin) * 1.15

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
                direction=TextDirection.VERTICAL.value if is_vertical else TextDirection.HORIZONTAL.value,
                polygon=b.get("polygon"),
                angle=float(b.get("angle", 0.0) or 0.0)
            )
            blocks.append(block)

        sorted_blocks = sort_reading_order(blocks)
        if progress_callback:
            progress_callback(100, f"识别完成，提取 {len(sorted_blocks)} 个文本气泡")

        return sorted_blocks
