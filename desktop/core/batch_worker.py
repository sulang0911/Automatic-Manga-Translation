import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import List, Dict, Any

from .ocr_engine import OCREngine
from .inpaint_engine import InpaintEngine
from .translation_engine import TranslationEngine
from .typography_engine import TypographyEngine

class BatchWorker(QThread):
    sig_batch_progress = pyqtSignal(int, int, str, int, str) # current, total, filename, pct, msg
    sig_item_completed = pyqtSignal(str, dict) # image_id, result
    sig_batch_finished = pyqtSignal(int, int) # success_count, fail_count
    sig_item_failed = pyqtSignal(str, str) # image_id, error_msg

    def __init__(self, queue_items: List[Dict[str, Any]], config: Dict[str, Any], export_dir: str = "", parent=None):
        super().__init__(parent)
        self.queue_items = queue_items
        self.config = config
        self.export_dir = export_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        total = len(self.queue_items)
        success_count = 0
        fail_count = 0

        # Shared engines for batch reuse
        ocr_eng = OCREngine(
            engine_type=self.config.get("ocr_engine", "paddle"),
            use_gpu=self.config.get("use_gpu", True),
            lang=self.config.get("ocr_lang", "japan")
        )
        inpaint_eng = InpaintEngine(mode=self.config.get("inpaint_engine", "auto"))
        trans_eng = TranslationEngine(
            provider=self.config.get("provider", "deepseek"),
            api_key=self.config.get("api_key", ""),
            model=self.config.get("model", "deepseek-chat"),
            custom_endpoint=self.config.get("custom_endpoint", ""),
            target_lang=self.config.get("target_lang", "简体中文"),
            source_lang=self.config.get("source_lang", "日语"),
            temperature=self.config.get("temperature", 0.3),
            system_prompt=self.config.get("system_prompt", "")
        )
        typo_eng = TypographyEngine()

        for idx, item in enumerate(self.queue_items):
            if self._is_cancelled:
                break

            img_id = item["id"]
            img_path = item["path"]
            filename = os.path.basename(img_path)

            try:
                self.sig_batch_progress.emit(idx + 1, total, filename, 10, "正在读取图像...")

                stream = open(img_path, "rb")
                bytes_data = bytearray(stream.read())
                stream.close()
                nparr = np.asarray(bytes_data, dtype=np.uint8)
                original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if original_img is None:
                    raise RuntimeError("无法解码图像文件")

                # OCR
                self.sig_batch_progress.emit(idx + 1, total, filename, 30, "正在识别文字...")
                blocks = ocr_eng.detect_and_recognize(original_img)

                # Inpaint
                self.sig_batch_progress.emit(idx + 1, total, filename, 55, "正在消除背景...")
                erased_img = inpaint_eng.inpaint(
                    original_img, blocks,
                    bubble_dilation=self.config.get("bubble_dilation", 3),
                    onomatopoeia_dilation=self.config.get("onomatopoeia_dilation", 6),
                    feather_radius=self.config.get("feather_radius", 4)
                )

                # Translate
                self.sig_batch_progress.emit(idx + 1, total, filename, 75, "正在大模型翻译...")
                blocks = trans_eng.translate_blocks(blocks)

                # Render
                self.sig_batch_progress.emit(idx + 1, total, filename, 90, "正在生成排版...")
                base_bg = erased_img if erased_img is not None else original_img
                translated_img = typo_eng.render_translations(base_bg, blocks, self.config)

                # Auto save if export_dir specified
                export_path = None
                if self.export_dir and os.path.exists(self.export_dir):
                    name_without_ext = os.path.splitext(filename)[0]
                    export_path = os.path.join(self.export_dir, f"{name_without_ext}_translated.png")
                    _, buf = cv2.imencode(".png", translated_img)
                    with open(export_path, "wb") as f:
                        f.write(buf.tobytes())

                self.sig_batch_progress.emit(idx + 1, total, filename, 100, "完成")
                self.sig_item_completed.emit(img_id, {
                    "original_img": original_img,
                    "blocks": blocks,
                    "erased_img": erased_img,
                    "translated_img": translated_img,
                    "export_path": export_path
                })
                success_count += 1

            except Exception as e:
                print(f"[-] Batch item failed: {filename}: {e}")
                self.sig_item_failed.emit(img_id, str(e))
                fail_count += 1

        self.sig_batch_finished.emit(success_count, fail_count)
