"""
app/core/pipeline/batch_worker.py
Batch processing QThread worker executing sequential chapter translation.
Maintains shared engine instances, granular batch signals, and automatic file export.
"""
import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import List, Dict, Any, Optional

from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine
from app.core.typography.engine import TypographyEngine
from app.core.translation import TranslationManager, ProviderConfig
from app.core.models import TranslationBlock
from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imread, safe_cv2_imwrite
from app.core.pipeline.exporter import MangaExporter
import gc


class BatchWorker(QThread):
    sig_batch_progress = pyqtSignal(int, int, str, int, str)  # current, total, filename, pct, msg
    sig_item_completed = pyqtSignal(str, dict)  # image_id, result
    sig_batch_finished = pyqtSignal(int, int)  # success_count, fail_count
    sig_item_failed = pyqtSignal(str, str)  # image_id, error_msg

    def __init__(
        self,
        queue_items: List[Dict[str, Any]],
        config: Dict[str, Any],
        export_dir: str = "",
        root_dir: Optional[str] = None,
        force_retranslate: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.queue_items = queue_items
        self.config = config
        self.export_dir = export_dir
        self.root_dir = root_dir
        self.force_retranslate = force_retranslate
        self._is_cancelled = False

        if not self.root_dir and self.queue_items:
            valid_roots = list(dict.fromkeys(it.get("root_dir") for it in self.queue_items if it.get("root_dir")))
            if len(valid_roots) == 1:
                self.root_dir = valid_roots[0]
            else:
                valid_paths = [it["path"] for it in self.queue_items if it.get("path")]
                if len(valid_paths) > 1:
                    try:
                        common = os.path.commonpath([os.path.normpath(os.path.abspath(p)) for p in valid_paths])
                        if os.path.isdir(common):
                            self.root_dir = common
                        else:
                            self.root_dir = os.path.dirname(common)
                    except Exception:
                        self.root_dir = None

    def cancel(self):
        """Signals cooperative cancellation."""
        self._is_cancelled = True

    def resolve_export_path(self, item: Dict[str, Any]) -> Optional[str]:
        """Resolves target export destination path preserving relative subfolder structure."""
        if not self.export_dir:
            return None
        img_path = item.get("path", "")
        if not img_path:
            return None
        return MangaExporter.compute_export_path(
            image_path=img_path,
            export_dir=self.export_dir,
            rel_path=item.get("rel_path"),
            root_dir=self.root_dir or item.get("root_dir")
        )

    def run(self):
        total = len(self.queue_items)
        if total == 0:
            self.sig_batch_finished.emit(0, 0)
            return

        # Shared engines for batch reuse
        ocr_cfg = self.config.get("ocr", {}) if isinstance(self.config.get("ocr"), dict) else {}
        ocr_eng = OCREngine(
            engine_type=ocr_cfg.get("engine", self.config.get("ocr_engine", "easyocr")),
            use_gpu=not ocr_cfg.get("force_cpu", False) if "force_cpu" in ocr_cfg else self.config.get("use_gpu", True),
            lang=ocr_cfg.get("lang", self.config.get("ocr_lang", "japan"))
        )
        inp_cfg = self.config.get("inpaint", {}) if isinstance(self.config.get("inpaint"), dict) else {}
        inpaint_eng = InpaintEngine(mode=inp_cfg.get("engine", self.config.get("inpaint_engine", "auto")))

        llm_cfg = self.config.get("llm", {}) if isinstance(self.config.get("llm"), dict) else {}
        provider_name = llm_cfg.get("provider", self.config.get("provider", "openai"))
        api_key = llm_cfg.get("api_key", self.config.get("api_key", ""))
        model = llm_cfg.get("model", self.config.get("model", "gpt-4o-mini"))
        endpoint = llm_cfg.get("endpoint", self.config.get("custom_endpoint", ""))
        temperature = float(llm_cfg.get("temperature", self.config.get("temperature", 0.3)))
        timeout = float(llm_cfg.get("timeout_seconds", self.config.get("timeout_seconds", 60.0)))
        max_retries = int(llm_cfg.get("max_retries", self.config.get("max_retries", 3)))
        proxy_url = llm_cfg.get("proxy_url", self.config.get("proxy_url", ""))

        trans_mgr = TranslationManager.get_instance()
        trans_mgr.set_active_provider(
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
        source_lang = self.config.get("source_lang", "自动识别")
        target_lang = self.config.get("target_lang", "简体中文")
        typo_eng = TypographyEngine()

        success_count = 0
        fail_count = 0

        cache_mgr = get_cache_manager()

        for idx, item in enumerate(self.queue_items):
            if self._is_cancelled:
                break

            img_id = item.get("id", str(idx))
            img_path = item.get("path", "")
            filename = os.path.basename(img_path)

            try:
                # 1. Check for full cache breakpoint resumption (skipped when force_retranslate is True)
                if not self.force_retranslate and cache_mgr.is_fully_translated(img_path):
                    self.sig_batch_progress.emit(idx + 1, total, filename, 90, "已命中本地缓存，正在检查导出...")
                    cached_data = cache_mgr.load_page_cache(img_path, load_images=False)
                    export_path = self.resolve_export_path(item)
                    if export_path:
                        if not os.path.exists(export_path):
                            full_c = cache_mgr.load_page_cache(img_path, load_images=True)
                            if full_c["rendered_img"] is not None:
                                MangaExporter.export_hierarchical_image(
                                    full_c["rendered_img"],
                                    export_path,
                                    source_path=img_path
                                )

                    self.sig_batch_progress.emit(idx + 1, total, filename, 100, "秒速恢复(缓存)")
                    self.sig_item_completed.emit(img_id, {
                        "has_cache": True,
                        "blocks_count": len(cached_data.get("blocks", [])),
                        "export_path": export_path
                    })
                    success_count += 1
                    continue

                # 2. Need processing: Load original image safely
                self.sig_batch_progress.emit(idx + 1, total, filename, 10, "正在读取图像...")
                original_img = safe_cv2_imread(img_path)
                if original_img is None:
                    raise RuntimeError("无法解码图像文件")

                cache_status = cache_mgr.has_cache(img_path)

                # 3. OCR Stage (Check cache first)
                if cache_status["blocks"]:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 25, "正在读取已识别文本缓存...")
                    cached_blocks = cache_mgr.load_page_cache(img_path, load_images=False)["blocks"]
                    blocks = cached_blocks
                else:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 30, "正在识别文字...")
                    blocks = ocr_eng.detect_and_recognize(original_img)
                    cache_mgr.save_page_cache(img_path, blocks=blocks)

                # 4. Inpaint Stage (Check cache first)
                erased_img = None
                if cache_status["erased"]:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 50, "正在载入已消除底图缓存...")
                    erased_img = cache_mgr.load_page_cache(img_path, load_images=True)["erased_img"]
                else:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 55, "正在消除背景...")
                    erased_img = inpaint_eng.inpaint(
                        original_img, blocks,
                        bubble_dilation=self.config.get("bubble_dilation", 3),
                        onomatopoeia_dilation=self.config.get("onomatopoeia_dilation", 6),
                        feather_radius=self.config.get("feather_radius", 4)
                    )
                    cache_mgr.save_page_cache(img_path, erased_img=erased_img, blocks=blocks)

                # 5. Translate Stage (Check if blocks already have translations)
                if self.force_retranslate:
                    for b in blocks:
                        if isinstance(b, dict):
                            b["translated_text"] = ""
                        elif hasattr(b, "translated_text"):
                            b.translated_text = ""
                    has_translations = False
                else:
                    has_translations = any(
                        bool(getattr(b, "translated_text", "") if hasattr(b, "translated_text") else b.get("translated_text", ""))
                        for b in blocks
                    )
                if not has_translations:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 75, "正在大模型翻译...")
                    blocks = trans_mgr.translate(blocks=blocks, mode="text", source_lang=source_lang, target_lang=target_lang)
                    cache_mgr.save_page_cache(img_path, blocks=blocks)

                # 6. Render Stage
                self.sig_batch_progress.emit(idx + 1, total, filename, 90, "正在生成排版...")
                base_bg = erased_img if erased_img is not None else original_img
                translated_img = typo_eng.render_translations(base_bg, blocks, self.config)

                # 7. Persist complete cache to disk (.amt_cache/)
                cache_mgr.save_page_cache(
                    img_path,
                    erased_img=erased_img,
                    blocks=blocks,
                    rendered_img=translated_img
                )

                # 8. Auto export if export_dir specified
                export_path = self.resolve_export_path(item)
                if export_path:
                    compressed = False
                    if hasattr(self.config, "style"):
                        compressed = getattr(self.config.style, "export_compressed", False)
                    elif isinstance(self.config, dict):
                        style_cfg = self.config.get("style", {})
                        if isinstance(style_cfg, dict):
                            compressed = style_cfg.get("export_compressed", False)
                    MangaExporter.export_hierarchical_image(
                        translated_img,
                        export_path,
                        source_path=img_path,
                        compressed=compressed
                    )

                self.sig_batch_progress.emit(idx + 1, total, filename, 100, "完成并已存盘")
                self.sig_item_completed.emit(img_id, {
                    "has_cache": True,
                    "blocks_count": len(blocks),
                    "export_path": export_path
                })
                success_count += 1

                # 9. Free heavy arrays from RAM immediately!
                del original_img
                del erased_img
                del translated_img
                if (idx + 1) % 4 == 0:
                    gc.collect()

            except Exception as e:
                print(f"[-] Batch item failed: {filename}: {e}")
                self.sig_item_failed.emit(img_id, str(e))
                fail_count += 1

        self.sig_batch_finished.emit(success_count, fail_count)
