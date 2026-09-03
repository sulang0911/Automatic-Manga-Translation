"""
app/core/ocr/easyocr_engine.py
PyTorch-accelerated OCR backend based on EasyOCR.
Reliable GPU execution across all NVIDIA architectures (including Pascal GTX 10-series).
"""
import logging
from typing import List, Dict, Any, Optional, Callable
import cv2
import numpy as np
import torch

from app.core.models import TranslationBlock, BlockType, TextDirection
from app.core.ocr.base import BaseOCREngine, is_solid_color_page, merge_adjacent_boxes
from app.core.ocr.reading_order import sort_reading_order
from app.core.inpaint.color_analyzer import get_background_color_hex, analyze_text_color

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        self.languages = languages or ['ja', 'en']
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()
        self._reader = None

    def _get_reader(self):
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

        reader = self._get_reader()

        if progress_callback:
            progress_callback(35, "正在执行文字检测与多语言识别...")

        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_results = reader.readtext(rgb_img)

        raw_boxes = []
        for bbox, text, conf in raw_results:
            clean_text = text.strip()
            if not clean_text or conf < 0.20:
                continue
            pts = np.array(bbox, dtype=np.int32)
            raw_boxes.append({
                "xmin": int(np.min(pts[:, 0])),
                "ymin": int(np.min(pts[:, 1])),
                "xmax": int(np.max(pts[:, 0])),
                "ymax": int(np.max(pts[:, 1])),
                "text": clean_text,
                "conf": float(conf)
            })

        if progress_callback:
            progress_callback(70, "正在合并临近对话段落...")

        merged_boxes = merge_adjacent_boxes(raw_boxes, w_img, h_img)

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
                direction=TextDirection.VERTICAL.value if is_vertical else TextDirection.HORIZONTAL.value
            )
            blocks.append(block)

        if progress_callback:
            progress_callback(85, "正在进行漫画阅读流空间排序...")

        sorted_blocks = sort_reading_order(blocks)

        if progress_callback:
            progress_callback(100, f"OCR 提取完成，共识别 {len(sorted_blocks)} 个文本气泡")

        return sorted_blocks
