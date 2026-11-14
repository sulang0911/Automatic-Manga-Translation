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
from app.core.inpaint.color_analyzer import refresh_blocks_text_colors
from app.core.translation import TranslationManager, ProviderConfig
from app.core.models import TranslationBlock
from app.core.cache.cache_manager import get_cache_manager
from app.core.hardware import cleanup_gpu_memory, configure_low_resource_runtime, lower_worker_priority


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
        manual_mask: Optional[np.ndarray] = None,
        mode: str = "full",
        parent=None
    ):
        super().__init__(parent)
        self.image_path = image_path
        self.config = config
        self.existing_blocks = existing_blocks
        self.existing_erased = existing_erased
        self.manual_mask = manual_mask
        self.mode = mode  # "full" | "ocr_only" | "inpaint_only" | "translate_only" | "render_only"
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        lower_worker_priority()
        configure_low_resource_runtime()
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
            del bytes_data, nparr

            if original_img is None:
                self.sig_error.emit("无法解码图像文件，请检查格式。")
                return

            # When running a full single-page translation ("full"), always re-run OCR and Inpainting
            # from scratch rather than reusing stale cached blocks or erased images.
            blocks = None if self.mode == "full" else self.existing_blocks
            erased_img = None if self.mode in ("full", "inpaint_only") else self.existing_erased
            translated_img = None

            cache_mgr = get_cache_manager()
            cache_status = cache_mgr.has_cache(self.image_path)
            auto_save = self.config.get("auto_save_cache", True) if isinstance(self.config, dict) else getattr(self.config, "auto_save_cache", True)

            # 1. OCR Stage
            if blocks is None or self.mode in ["full", "ocr_only"]:
                if self._is_cancelled:
                    return
                from app.core.pipeline.utils import normalize_source_lang, is_auto_source_lang, source_lang_to_ocr_lang
                effective_source = self.config.get("source_lang") or self.config.get("ocr", {}).get("lang") or self.config.get("ocr_lang")
                norm_src = normalize_source_lang(effective_source)
                reuse_cache = False

                if cache_status["blocks"] and self.mode not in ("full", "ocr_only"):
                    cached_blocks = cache_mgr.load_page_cache(self.image_path, load_images=False)["blocks"]
                    if self.mode == "translate_only" or is_auto_source_lang(norm_src):
                        reuse_cache = True
                    else:
                        sample_text = " ".join(str(b.get("original_text", "")) if isinstance(b, dict) else str(getattr(b, "original_text", "")) for b in cached_blocks[:5])
                        has_kana = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in sample_text)
                        has_hangul = any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in sample_text)
                        if norm_src == "en" and (has_kana or has_hangul):
                            reuse_cache = False
                        elif norm_src == "ko" and has_kana and not has_hangul:
                            reuse_cache = False
                        elif norm_src == "ja" and has_hangul and not has_kana:
                            reuse_cache = False
                        else:
                            reuse_cache = True

                if reuse_cache:
                    self.sig_progress.emit(25, "从本地缓存恢复 OCR 识别结果...")
                    blocks = [b.to_dict() if hasattr(b, "to_dict") else b for b in cached_blocks]
                    refresh_blocks_text_colors(original_img, blocks)
                    self.sig_step_done.emit("ocr", blocks)
                else:
                    self.sig_progress.emit(15, "正在执行本地高精度 OCR 识别...")
                    from app.core.pipeline.utils import build_ocr_engine_kwargs
                    ocr_lang = source_lang_to_ocr_lang(effective_source)
                    is_manual = not is_auto_source_lang(norm_src)
                    ocr_eng = OCREngine(**build_ocr_engine_kwargs(
                        self.config, lang=ocr_lang, is_manual=is_manual, source_lang=effective_source
                    ))

                    def ocr_cb(pct, msg):
                        self.sig_progress.emit(int(15 + pct * 0.25), msg)

                    blocks = ocr_eng.detect_and_recognize(original_img, progress_callback=ocr_cb)
                    if auto_save:
                        cache_mgr.save_page_cache(self.image_path, blocks=blocks)
                    cleanup_gpu_memory(trim_ram=False)
                    # Since OCR produced fresh blocks, force Stage 2 to re-inpaint cleanly
                    erased_img = None

                ocr_cfg = self.config.get("ocr", {}) if isinstance(self.config.get("ocr"), dict) else {}
                self.sig_step_done.emit("ocr", blocks)

                if self.mode == "ocr_only":
                    self.sig_finished.emit({
                        "original_img": original_img,
                        "blocks": blocks,
                        "erased_img": erased_img,
                        "translated_img": None
                    })
                    return

            from app.core.pipeline.utils import (
                extract_onomatopoeia_mode,
                heal_mislabeled_onomatopoeia_blocks,
                is_block_ignored_onomatopoeia,
                partition_render_blocks,
                prepare_render_base,
            )
            onoma_mode = extract_onomatopoeia_mode(self.config)

            if blocks:
                if self.mode == "translate_only":
                    heal_mislabeled_onomatopoeia_blocks(blocks)
                refresh_blocks_text_colors(original_img, blocks)

            # 2. Translation Stage (before Inpainting so LLM-identified onomatopoeia stays un-erased when ignored)
            if self.mode not in ("ocr_only", "inpaint_only") and (
                self.mode in ["full", "translate_only"] or any(
                    bool((getattr(b, "original_text", "") if hasattr(b, "original_text") else b.get("original_text", "")).strip())
                    and not (getattr(b, "translated_text", "") if hasattr(b, "translated_text") else b.get("translated_text", ""))
                    and not is_block_ignored_onomatopoeia(b, onoma_mode)
                    for b in (blocks or [])
                )
            ):
                if self._is_cancelled:
                    return
                self.sig_progress.emit(48, "正在调用大语言模型进行精准翻译...")
                llm_cfg = self.config.get("llm", {}) if isinstance(self.config.get("llm"), dict) else {}
                provider_name = llm_cfg.get("provider", self.config.get("provider", "openai"))
                api_key = llm_cfg.get("api_key", self.config.get("api_key", ""))
                model = llm_cfg.get("model", self.config.get("model", "gpt-4o-mini"))
                endpoint = llm_cfg.get("endpoint", self.config.get("custom_endpoint", ""))
                temperature = float(llm_cfg.get("temperature", self.config.get("temperature", 0.3)))
                timeout = float(llm_cfg.get("timeout_seconds", self.config.get("timeout_seconds", 60.0)))
                max_retries = int(llm_cfg.get("max_retries", self.config.get("max_retries", 3)))
                proxy_url = llm_cfg.get("proxy_url", self.config.get("proxy_url", ""))
                system_prompt = llm_cfg.get("system_prompt", self.config.get("system_prompt", ""))

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
                        proxy_url=proxy_url,
                        system_prompt=system_prompt
                    )
                )

                def trans_cb(pct, msg):
                    self.sig_progress.emit(int(48 + pct * 0.18), msg)

                source_lang = self.config.get("source_lang", "自动识别")
                target_lang = self.config.get("target_lang", "简体中文")

                if self.mode == "translate_only" and blocks:
                    for b in blocks:
                        if isinstance(b, dict):
                            b["translated_text"] = ""
                        elif hasattr(b, "translated_text"):
                            b.translated_text = ""

                blocks = mgr.translate(
                    blocks=blocks,
                    mode="text",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    progress_callback=trans_cb,
                    onomatopoeia_mode=onoma_mode,
                )
                if auto_save:
                    cache_mgr.save_page_cache(self.image_path, blocks=blocks)
                self.sig_step_done.emit("translate", blocks)

            # 3. Inpainting Stage
            erased_saved = False
            if erased_img is None and cache_status["erased"] and self.mode not in ("full", "inpaint_only"):
                self.sig_progress.emit(72, "从本地缓存直接载入无字底图(免重复擦除)...")
                erased_img = cache_mgr.load_page_cache(self.image_path, load_images=True)["erased_img"]
                erased_saved = True
                self.sig_step_done.emit("inpaint", erased_img)
            elif erased_img is None or self.mode in ["full", "inpaint_only"]:
                if self._is_cancelled:
                    return
                self.sig_progress.emit(70, "正在执行图像背景文字清除与智能修复...")
                inp_cfg = self.config.get("inpaint", {}) if isinstance(self.config.get("inpaint"), dict) else {}
                inpaint_eng = InpaintEngine(mode=inp_cfg.get("engine", self.config.get("inpaint_engine", "auto")))

                def inpaint_cb(pct, msg):
                    self.sig_progress.emit(int(70 + pct * 0.20), msg)

                erased_img = inpaint_eng.inpaint(
                    original_img, blocks,
                    bubble_dilation=self.config.get("bubble_dilation", 3),
                    onomatopoeia_dilation=self.config.get("onomatopoeia_dilation", 6),
                    feather_radius=self.config.get("feather_radius", 4),
                    progress_callback=inpaint_cb,
                    manual_mask=self.manual_mask,
                    onomatopoeia_mode=onoma_mode
                )
                if auto_save:
                    cache_mgr.save_page_cache(self.image_path, erased_img=erased_img, blocks=blocks)
                    erased_saved = True
                cleanup_gpu_memory(trim_ram=False)
                self.sig_step_done.emit("inpaint", erased_img)

                if self.mode == "inpaint_only":
                    self.sig_finished.emit({
                        "original_img": original_img,
                        "blocks": blocks,
                        "erased_img": erased_img,
                        "translated_img": None
                    })
                    return

            # 4. Typography Rendering Stage
            if self._is_cancelled:
                return
            self.sig_progress.emit(92, "正在生成高保真排版与字体渲染...")
            typo_eng = TypographyEngine()
            base_bg = erased_img if erased_img is not None else original_img
            render_blocks, ignored_blocks = partition_render_blocks(blocks, onoma_mode)
            render_base = prepare_render_base(base_bg, original_img, render_blocks, ignored_blocks)
            translated_img = typo_eng.render_translations(render_base, render_blocks, self.config)
            if auto_save:
                cache_mgr.save_page_cache(
                    self.image_path,
                    erased_img=None if erased_saved else erased_img,
                    blocks=blocks,
                    rendered_img=translated_img
                )
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
        finally:
            try:
                cleanup_gpu_memory(trim_ram=True)
            except Exception:
                pass
