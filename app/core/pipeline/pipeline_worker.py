"""
app/core/pipeline/pipeline_worker.py
Dedicated QThread worker executing the single-page translation pipeline off the UI thread.
Coordinating OCR, Inpainting, Translation, and Typography Rendering with cooperative cancellation.
"""
import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any, List, Optional

from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine
from app.core.typography.engine import TypographyEngine
from app.core.translation import TranslationManager, ProviderConfig
from app.core.models import TranslationBlock
from app.core.cache.cache_manager import get_cache_manager


class PipelineWorker(QThread):
    sig_progress = pyqtSignal(int, str)
    sig_step_done = pyqtSignal(str, object)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(
        self,
        image_path: str,
        config: Dict[str, Any],
        existing_blocks: Optional[List[Dict[str, Any]]] = None,
        existing_erased: Optional[np.ndarray] = None,
        mode: str = "full",
        parent=None
    ):
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

            cache_mgr = get_cache_manager()
            cache_status = cache_mgr.has_cache(self.image_path)

            # 1. OCR Stage
            if blocks is None or self.mode in ["full", "ocr_only"]:
                if self._is_cancelled:
                    return
                # Check cache for blocks if available
                if cache_status["blocks"] and self.mode != "ocr_only":
                    self.sig_progress.emit(25, "从本地缓存恢复 OCR 识别结果...")
                    blocks = cache_mgr.load_page_cache(self.image_path, load_images=False)["blocks"]
                    blocks = [b.to_dict() if hasattr(b, "to_dict") else b for b in blocks]
                    self.sig_step_done.emit("ocr", blocks)
                else:
                    self.sig_progress.emit(15, "正在执行本地高精度 OCR 识别...")
                    ocr_cfg = self.config.get("ocr", {}) if isinstance(self.config.get("ocr"), dict) else {}
                    ocr_eng = OCREngine(
                        engine_type=ocr_cfg.get("engine", self.config.get("ocr_engine", "easyocr")),
                        use_gpu=not ocr_cfg.get("force_cpu", False) if "force_cpu" in ocr_cfg else self.config.get("use_gpu", True),
                        lang=ocr_cfg.get("lang", self.config.get("ocr_lang", "japan")),
                        enable_ensemble_detection=ocr_cfg.get("ensemble_detection", self.config.get("ocr_ensemble_detection", False)),
                        enable_ensemble_recognition=ocr_cfg.get("ensemble_recognition", self.config.get("ocr_ensemble_recognition", False)),
                    )

                    def ocr_cb(pct, msg):
                        self.sig_progress.emit(int(15 + pct * 0.25), msg)

                    blocks = ocr_eng.detect_and_recognize(original_img, progress_callback=ocr_cb)
                    cache_mgr.save_page_cache(self.image_path, blocks=blocks)
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
            if erased_img is None and cache_status["erased"] and self.mode != "inpaint_only":
                self.sig_progress.emit(50, "从本地缓存直接载入无字底图(免重复擦除)...")
                erased_img = cache_mgr.load_page_cache(self.image_path, load_images=True)["erased_img"]
                self.sig_step_done.emit("inpaint", erased_img)
            elif erased_img is None or self.mode in ["full", "inpaint_only"]:
                if self._is_cancelled:
                    return
                self.sig_progress.emit(45, "正在执行图像背景文字清除与智能修复...")
                inp_cfg = self.config.get("inpaint", {}) if isinstance(self.config.get("inpaint"), dict) else {}
                inpaint_eng = InpaintEngine(mode=inp_cfg.get("engine", self.config.get("inpaint_engine", "auto")))

                def inpaint_cb(pct, msg):
                    self.sig_progress.emit(int(45 + pct * 0.25), msg)

                erased_img = inpaint_eng.inpaint(
                    original_img, blocks,
                    bubble_dilation=self.config.get("bubble_dilation", 3),
                    onomatopoeia_dilation=self.config.get("onomatopoeia_dilation", 6),
                    feather_radius=self.config.get("feather_radius", 4),
                    progress_callback=inpaint_cb
                )
                cache_mgr.save_page_cache(self.image_path, erased_img=erased_img, blocks=blocks)
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
            if self.mode in ["full", "translate_only"] or any(not b.get("translated_text") for b in (blocks or [])):
                if self._is_cancelled:
                    return
                self.sig_progress.emit(72, "正在调用大语言模型进行精准翻译...")
                llm_cfg = self.config.get("llm", {}) if isinstance(self.config.get("llm"), dict) else {}
                provider_name = llm_cfg.get("provider", self.config.get("provider", "openai"))
                api_key = llm_cfg.get("api_key", self.config.get("api_key", ""))
                model = llm_cfg.get("model", self.config.get("model", "gpt-4o-mini"))
                endpoint = llm_cfg.get("endpoint", self.config.get("custom_endpoint", ""))
                temperature = float(llm_cfg.get("temperature", self.config.get("temperature", 0.3)))
                timeout = float(llm_cfg.get("timeout_seconds", self.config.get("timeout_seconds", 60.0)))
                max_retries = int(llm_cfg.get("max_retries", self.config.get("max_retries", 3)))
                proxy_url = llm_cfg.get("proxy_url", self.config.get("proxy_url", ""))

                mgr = TranslationManager.get_instance()
                mgr.set_active_provider(
                    provider_name,
                    ProviderConfig(
                        provider_name=provider_name,
                        api_key=api_key,
                        model=model,
                        endpoint=endpoint,
                        temperature=temperature,
                        timeout_seconds=timeout,
                        max_retries=max_retries,
                        proxy_url=proxy_url
                    )
                )

                def trans_cb(pct, msg):
                    self.sig_progress.emit(int(72 + pct * 0.18), msg)

                source_lang = self.config.get("source_lang", "自动识别")
                target_lang = self.config.get("target_lang", "简体中文")

                blocks = mgr.translate(
                    blocks=blocks,
                    mode="text",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    progress_callback=trans_cb
                )
                cache_mgr.save_page_cache(self.image_path, blocks=blocks)
                self.sig_step_done.emit("translate", blocks)

            # 4. Typography Rendering Stage
            if self._is_cancelled:
                return
            self.sig_progress.emit(92, "正在生成高保真排版与字体渲染...")
            typo_eng = TypographyEngine()
            base_bg = erased_img if erased_img is not None else original_img
            translated_img = typo_eng.render_translations(base_bg, blocks, self.config)
            cache_mgr.save_page_cache(self.image_path, erased_img=erased_img, blocks=blocks, rendered_img=translated_img)
            self.sig_step_done.emit("render", translated_img)

            self.sig_progress.emit(100, "处理完成并已存入缓存！")
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
