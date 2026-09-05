import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any, List, Optional

from .ocr_engine import OCREngine
from .inpaint_engine import InpaintEngine
from .translation_engine import TranslationEngine
from .typography_engine import TypographyEngine

class PipelineWorker(QThread):
    sig_progress = pyqtSignal(int, str)
    sig_step_done = pyqtSignal(str, object)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(self, image_path: str, config: Dict[str, Any], existing_blocks: Optional[List[Dict[str, Any]]] = None,
                 existing_erased: Optional[np.ndarray] = None, mode: str = "full", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.config = config
        self.existing_blocks = existing_blocks
        self.existing_erased = existing_erased
        self.mode = mode  # "full" | "ocr_only" | "inpaint_only" | "translate_only" | "render_only"
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if not os.path.exists(self.image_path):
                self.sig_error.emit(f"文件不存在: {self.image_path}")
                return

            self.sig_progress.emit(5, "正在读取原始漫画图像...")
            
            # Read image using numpy for proper unicode path handling on Windows
            stream = open(self.image_path, "rb")
            bytes_data = bytearray(stream.read())
            stream.close()
            nparr = np.asarray(bytes_data, dtype=np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if original_img is None:
                self.sig_error.emit("无法解码图像文件，请检查格式。")
                return

            blocks = self.existing_blocks
            erased_img = self.existing_erased
            translated_img = None

            # 1. OCR Stage
            if blocks is None or self.mode in ["full", "ocr_only"]:
                if self._is_cancelled: return
                self.sig_progress.emit(15, "正在执行本地高精度 OCR 识别...")
                ocr_eng = OCREngine(
                    engine_type=self.config.get("ocr_engine", "paddle"),
                    use_gpu=self.config.get("use_gpu", True),
                    lang=self.config.get("ocr_lang", "japan"),
                    enable_ensemble_detection=self.config.get("ocr_ensemble_detection", False),
                    enable_ensemble_recognition=self.config.get("ocr_ensemble_recognition", False)
                )
                
                def ocr_cb(pct, msg):
                    self.sig_progress.emit(int(15 + pct * 0.25), msg)

                blocks = ocr_eng.detect_and_recognize(original_img, progress_callback=ocr_cb)
                self.sig_step_done.emit("ocr", blocks)

                if self.mode == "ocr_only":
                    self.sig_finished.emit({
                        "original_img": original_img,
                        "blocks": blocks,
                        "erased_img": erased_img,
                        "translated_img": None
                    })
                    return

            # 2. Inpainting Stage
            if erased_img is None or self.mode in ["full", "inpaint_only"]:
                if self._is_cancelled: return
                self.sig_progress.emit(45, "正在执行图像背景文字清除与智能修复...")
                inpaint_eng = InpaintEngine(mode=self.config.get("inpaint_engine", "auto"))
                
                def inpaint_cb(pct, msg):
                    self.sig_progress.emit(int(45 + pct * 0.25), msg)

                erased_img = inpaint_eng.inpaint(
                    original_img, blocks,
                    bubble_dilation=self.config.get("bubble_dilation", 3),
                    onomatopoeia_dilation=self.config.get("onomatopoeia_dilation", 6),
                    feather_radius=self.config.get("feather_radius", 4),
                    progress_callback=inpaint_cb
                )
                self.sig_step_done.emit("inpaint", erased_img)

                if self.mode == "inpaint_only":
                    self.sig_finished.emit({
                        "original_img": original_img,
                        "blocks": blocks,
                        "erased_img": erased_img,
                        "translated_img": None
                    })
                    return

            # 3. Translation Stage
            if self.mode in ["full", "translate_only"] or any(not b.get("translated_text") for b in blocks):
                if self._is_cancelled: return
                self.sig_progress.emit(72, "正在调用大语言模型进行精准翻译...")
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

                def trans_cb(pct, msg):
                    self.sig_progress.emit(int(72 + pct * 0.18), msg)

                blocks = trans_eng.translate_blocks(blocks, progress_callback=trans_cb)
                self.sig_step_done.emit("translate", blocks)

            # 4. Typography Rendering Stage
            if self._is_cancelled: return
            self.sig_progress.emit(92, "正在生成高保真排版与字体渲染...")
            typo_eng = TypographyEngine()
            base_bg = erased_img if erased_img is not None else original_img
            translated_img = typo_eng.render_translations(base_bg, blocks, self.config)
            self.sig_step_done.emit("render", translated_img)

            self.sig_progress.emit(100, "处理完成！")
            self.sig_finished.emit({
                "original_img": original_img,
                "blocks": blocks,
                "erased_img": erased_img,
                "translated_img": translated_img
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.sig_error.emit(f"处理失败: {str(e)}")
