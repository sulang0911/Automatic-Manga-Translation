"""
app/core/pipeline/batch_worker.py
Batch processing QThread worker executing sequential chapter translation.
Maintains shared engine instances, granular batch signals, automatic file export,
and aggressive inter-stage RAM/VRAM reclamation for 8GB RAM + GTX 1080 Ti systems.
"""
import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import List, Dict, Any, Optional, Tuple

from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine
from app.core.typography.engine import TypographyEngine
from app.core.inpaint.color_analyzer import refresh_blocks_text_colors
from app.core.translation import TranslationManager, ProviderConfig
from app.core.models import TranslationBlock
from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imread, safe_cv2_imwrite
from app.core.pipeline.exporter import MangaExporter
from app.core.hardware import cleanup_gpu_memory, configure_low_resource_runtime, lower_worker_priority
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
        retranslate_from_cache: bool = False,
        clear_cache_dirs: bool = True,
        parent=None
    ):
        super().__init__(parent)
        self.queue_items = queue_items
        self.config = config
        self.export_dir = export_dir
        self.root_dir = root_dir
        self.force_retranslate = force_retranslate
        self.retranslate_from_cache = retranslate_from_cache
        self.clear_cache_dirs = clear_cache_dirs
        self._is_cancelled = False
        self._is_paused = False

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

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

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

    def _export_cached_item(
        self,
        item: Dict[str, Any],
        export_path: str,
        cache_mgr,
        typo_eng: TypographyEngine,
    ) -> int:
        """Exports a single cached page (rendering on-the-fly if needed) and returns block count."""
        img_path = item.get("path", "")
        full_c = cache_mgr.load_page_cache(img_path, load_images=True)
        rendered_to_export = full_c.get("rendered_img")
        if rendered_to_export is None:
            base_bg = full_c.get("erased_img")
            if base_bg is None:
                base_bg = safe_cv2_imread(img_path)
            if base_bg is not None:
                c_blocks = full_c.get("blocks", [])
                eff_cfg = item.get("style") or full_c.get("page_style") or self.config
                onoma_mode = "normal"
                if hasattr(eff_cfg, "onomatopoeia_mode"):
                    onoma_mode = getattr(eff_cfg, "onomatopoeia_mode", "normal")
                elif hasattr(eff_cfg, "style"):
                    onoma_mode = getattr(eff_cfg.style, "onomatopoeia_mode", "normal")
                elif isinstance(eff_cfg, dict):
                    style_cfg = eff_cfg.get("style", eff_cfg)
                    if isinstance(style_cfg, dict):
                        onoma_mode = style_cfg.get("onomatopoeia_mode", "normal")

                render_blocks = [
                    b for b in c_blocks
                    if (b.type if hasattr(b, "type") else b.get("type", "")) != "onomatopoeia"
                    or onoma_mode != "ignore"
                    or (b.force_erase if hasattr(b, "force_erase") else b.get("force_erase", False))
                ]
                rendered_to_export = typo_eng.render_translations(base_bg, render_blocks, eff_cfg)
                if rendered_to_export is not None:
                    cache_mgr.save_page_cache(img_path, rendered_img=rendered_to_export)
            del base_bg

        if rendered_to_export is not None:
            compressed = False
            if hasattr(self.config, "style"):
                compressed = getattr(self.config.style, "export_compressed", False)
            elif isinstance(self.config, dict):
                style_cfg = self.config.get("style", {})
                if isinstance(style_cfg, dict):
                    compressed = style_cfg.get("export_compressed", False)
            MangaExporter.export_hierarchical_image(
                rendered_to_export,
                export_path,
                source_path=img_path,
                compressed=compressed,
            )
        blocks_cnt = len(full_c.get("blocks", []))
        del rendered_to_export, full_c
        return blocks_cnt

    def run(self):
        lower_worker_priority()
        configure_low_resource_runtime()

        total = len(self.queue_items)
        if total == 0:
            self.sig_batch_finished.emit(0, 0)
            return

        # Lazy shared engines: only instantiate heavy OCR/Inpaint/LLM engines on first cache miss
        ocr_cfg = self.config.get("ocr", {}) if isinstance(self.config.get("ocr"), dict) else {}
        from app.core.pipeline.utils import normalize_source_lang, is_auto_source_lang, source_lang_to_ocr_lang
        effective_source = self.config.get("source_lang") or ocr_cfg.get("lang") or self.config.get("ocr_lang")
        ocr_lang = source_lang_to_ocr_lang(effective_source)
        is_manual = not is_auto_source_lang(effective_source)

        inp_cfg = self.config.get("inpaint", {}) if isinstance(self.config.get("inpaint"), dict) else {}
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

        source_lang = self.config.get("source_lang", "自动识别")
        target_lang = self.config.get("target_lang", "简体中文")
        typo_eng = TypographyEngine()

        _engines: Dict[str, Any] = {"ocr": None, "inpaint": None, "trans": None}

        def _get_ocr_eng():
            if _engines["ocr"] is None:
                from app.core.pipeline.utils import build_ocr_engine_kwargs
                _engines["ocr"] = OCREngine(**build_ocr_engine_kwargs(
                    self.config, lang=ocr_lang, is_manual=is_manual, source_lang=effective_source
                ))
            return _engines["ocr"]

        def _get_inpaint_eng():
            if _engines["inpaint"] is None:
                _engines["inpaint"] = InpaintEngine(
                    mode=inp_cfg.get("engine", self.config.get("inpaint_engine", "auto"))
                )
            return _engines["inpaint"]

        def _get_trans_mgr():
            if _engines["trans"] is None:
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
                        system_prompt=system_prompt,
                    ),
                )
                _engines["trans"] = mgr
            return _engines["trans"]

        success_count = 0
        fail_count = 0

        cache_mgr = get_cache_manager()

        # When force_retranslate is True, clear software-generated cache before reprocessing
        if self.force_retranslate and self.queue_items:
            self.sig_batch_progress.emit(1, total, "", 0, "正在清理原有缓存...")
            if self.clear_cache_dirs:
                cache_mgr.clear_caches_for_items(self.queue_items)
            else:
                for it in self.queue_items:
                    p = it.get("path") if isinstance(it, dict) else str(it)
                    if p:
                        cache_mgr.clear_cache(p)

        import time
        handled_indices = set()

        # Parallel fast-path for pure cache-hit exports (keeps bounded RAM & avoids ML startup)
        if self.export_dir and not self.force_retranslate and not self.retranslate_from_cache:
            cache_export_tasks = []
            for idx, item in enumerate(self.queue_items):
                img_path = item.get("path", "")
                export_path = self.resolve_export_path(item)
                is_done_status = str(item.get("status", "")).lower() in ("completed", "done")
                is_cache_done = is_done_status or cache_mgr.is_fully_translated(img_path)
                if is_cache_done and export_path and not os.path.exists(export_path):
                    cache_export_tasks.append((idx, item, export_path))

            if len(cache_export_tasks) >= 2:
                import concurrent.futures
                max_workers = min(3, max(2, os.cpu_count() or 2))
                task_iter = iter(cache_export_tasks)
                pending_futs: Dict[Any, Tuple[int, Dict[str, Any], str]] = {}
                completed_export_n = 0

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="AMT-CacheExport"
                ) as pool:
                    def _fill_pool():
                        while len(pending_futs) < max_workers and not self._is_cancelled and not self._is_paused:
                            nxt = next(task_iter, None)
                            if nxt is None:
                                break
                            fut = pool.submit(self._export_cached_item, nxt[1], nxt[2], cache_mgr, typo_eng)
                            pending_futs[fut] = nxt

                    _fill_pool()
                    while pending_futs:
                        while self._is_paused and not self._is_cancelled:
                            time.sleep(0.1)
                        if self._is_cancelled:
                            break
                        done_futs, _ = concurrent.futures.wait(
                            pending_futs.keys(),
                            timeout=0.25,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for fut in done_futs:
                            idx_t, item_t, export_p_t = pending_futs.pop(fut)
                            handled_indices.add(idx_t)
                            img_id_t = item_t.get("id", str(idx_t))
                            fname_t = os.path.basename(item_t.get("path", ""))
                            completed_export_n += 1
                            try:
                                blocks_cnt = fut.result()
                                self.sig_batch_progress.emit(
                                    completed_export_n, total, fname_t, 100, "已导出(缓存)"
                                )
                                self.sig_item_completed.emit(img_id_t, {
                                    "has_cache": True,
                                    "blocks_count": blocks_cnt,
                                    "export_path": export_p_t,
                                })
                                success_count += 1
                            except Exception as exc:
                                print(f"[-] Batch cache export failed: {fname_t}: {exc}")
                                self.sig_item_failed.emit(img_id_t, str(exc))
                                fail_count += 1
                        _fill_pool()

        for idx, item in enumerate(self.queue_items):
            if idx in handled_indices:
                continue

            while self._is_paused and not self._is_cancelled:
                time.sleep(0.1)

            if self._is_cancelled:
                break

            img_id = item.get("id", str(idx))
            img_path = item.get("path", "")
            filename = os.path.basename(img_path)

            original_img = None
            erased_img = None
            translated_img = None

            try:
                export_path = self.resolve_export_path(item)

                # 1. Fast skip for already completed pages when running normal Batch Translate (skip completed directly without checking render/export)
                if not self.force_retranslate and not self.retranslate_from_cache:
                    is_done_status = str(item.get("status", "")).lower() in ("completed", "done")
                    is_cache_done = is_done_status or cache_mgr.is_fully_translated(img_path)
                    if is_cache_done:
                        need_export_file = bool(export_path and not os.path.exists(export_path))
                        if not need_export_file:
                            self.sig_batch_progress.emit(idx + 1, total, filename, 100, "已跳过(已完成)")
                            self.sig_item_completed.emit(img_id, {
                                "has_cache": True,
                                "skipped": True,
                                "blocks_count": len(item.get("blocks") or []),
                                "export_path": export_path
                            })
                            success_count += 1
                            continue
                        else:
                            # Batch export mode: single-item export from cache
                            self.sig_batch_progress.emit(idx + 1, total, filename, 90, "正在从缓存导出页面...")
                            blocks_cnt = self._export_cached_item(item, export_path, cache_mgr, typo_eng)
                            cleanup_gpu_memory(trim_ram=False)
                            self.sig_batch_progress.emit(idx + 1, total, filename, 100, "已导出(缓存)")
                            self.sig_item_completed.emit(img_id, {
                                "has_cache": True,
                                "blocks_count": blocks_cnt,
                                "export_path": export_path
                            })
                            success_count += 1
                            continue

                # 2. Need processing: Load original image safely
                self.sig_batch_progress.emit(idx + 1, total, filename, 10, "正在读取图像...")
                original_img = safe_cv2_imread(img_path)
                if original_img is None:
                    raise RuntimeError("无法解码图像文件")

                cache_status = {"blocks": False, "erased": False, "rendered": False} if self.force_retranslate else cache_mgr.has_cache(img_path)

                # 3. OCR Stage (Check cache first)
                reuse_ocr_cache = False
                if cache_status["blocks"]:
                    cached_blocks = cache_mgr.load_page_cache(img_path, load_images=False)["blocks"]
                    if self.retranslate_from_cache or is_auto_source_lang(effective_source):
                        reuse_ocr_cache = True
                    else:
                        norm_src = normalize_source_lang(effective_source)
                        sample_text = " ".join(str(b.get("original_text", "")) if isinstance(b, dict) else str(getattr(b, "original_text", "")) for b in cached_blocks[:5])
                        has_kana = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in sample_text)
                        has_hangul = any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in sample_text)
                        if norm_src == "en" and (has_kana or has_hangul):
                            reuse_ocr_cache = False
                        elif norm_src == "ko" and has_kana and not has_hangul:
                            reuse_ocr_cache = False
                        elif norm_src == "ja" and has_hangul and not has_kana:
                            reuse_ocr_cache = False
                        else:
                            reuse_ocr_cache = True

                if reuse_ocr_cache:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 25, "正在读取已识别文本缓存...")
                    blocks = cached_blocks
                    refresh_blocks_text_colors(original_img, blocks)
                else:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 30, "正在识别文字...")
                    blocks = _get_ocr_eng().detect_and_recognize(original_img)
                    cache_mgr.save_page_cache(img_path, blocks=blocks)
                    # Release OCR GPU activation buffers before Inpaint stage
                    cleanup_gpu_memory(trim_ram=False)

                # 4. Inpaint Stage (Check cache first)
                erased_already_saved = False
                if cache_status["erased"]:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 50, "正在载入已消除底图缓存...")
                    erased_already_saved = True
                    erased_img = cache_mgr.load_page_cache(img_path, load_images=True, load_rendered=False)["erased_img"]
                else:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 55, "正在消除背景...")
                    onoma_mode = "normal"
                    if hasattr(self.config, "style"):
                        onoma_mode = getattr(self.config.style, "onomatopoeia_mode", "normal")
                    elif isinstance(self.config, dict):
                        style_cfg = self.config.get("style", {})
                        if isinstance(style_cfg, dict):
                            onoma_mode = style_cfg.get("onomatopoeia_mode", "normal")

                    erased_img = _get_inpaint_eng().inpaint(
                        original_img, blocks,
                        bubble_dilation=self.config.get("bubble_dilation", 3),
                        onomatopoeia_dilation=self.config.get("onomatopoeia_dilation", 6),
                        feather_radius=self.config.get("feather_radius", 4),
                        onomatopoeia_mode=onoma_mode
                    )
                    cache_mgr.save_page_cache(img_path, erased_img=erased_img, blocks=blocks)
                    erased_already_saved = True

                # Free original_img and GPU VRAM BEFORE calling LLM Translation while keeping erased_img ready for rendering
                if erased_img is not None:
                    original_img = None
                cleanup_gpu_memory(trim_ram=False)

                # 5. Translate Stage (Check if blocks already have translations)
                if self.force_retranslate or self.retranslate_from_cache:
                    for b in blocks:
                        if isinstance(b, dict):
                            b["translated_text"] = ""
                        elif hasattr(b, "translated_text"):
                            b.translated_text = ""
                    needs_translation = bool(blocks)
                else:
                    needs_translation = any(
                        bool((getattr(b, "original_text", "") if hasattr(b, "original_text") else b.get("original_text", "")).strip()) and
                        not (getattr(b, "translated_text", "") if hasattr(b, "translated_text") else b.get("translated_text", ""))
                        for b in blocks
                    ) if blocks else False

                if needs_translation:
                    self.sig_batch_progress.emit(idx + 1, total, filename, 75, "正在大模型翻译...")
                    blocks = _get_trans_mgr().translate(blocks=blocks, mode="text", source_lang=source_lang, target_lang=target_lang)
                    cache_mgr.save_page_cache(img_path, blocks=blocks)

                # 6. Render & Persist Stage (Always pre-bake rendered_img into .amt_cache for zero-lag page switching)
                self.sig_batch_progress.emit(idx + 1, total, filename, 90, "正在生成排版并写入缓存..." if not export_path else "正在生成排版并导出...")
                base_bg = erased_img if erased_img is not None else original_img
                if base_bg is None:
                    base_bg = safe_cv2_imread(img_path)
                onoma_mode = "normal"
                if hasattr(self.config, "style"):
                    onoma_mode = getattr(self.config.style, "onomatopoeia_mode", "normal")
                elif isinstance(self.config, dict):
                    style_cfg = self.config.get("style", {})
                    if isinstance(style_cfg, dict):
                        onoma_mode = style_cfg.get("onomatopoeia_mode", "normal")

                render_blocks = [
                    b for b in blocks 
                    if (getattr(b, "type", "") if hasattr(b, "type") else b.get("type", "")) != "onomatopoeia" or onoma_mode != "ignore" or (getattr(b, "force_erase", False) if hasattr(b, "force_erase") else b.get("force_erase", False))
                ]
                eff_render_cfg = item.get("style") or cache_mgr.load_page_cache(img_path, load_images=False).get("page_style") or self.config
                translated_img = typo_eng.render_translations(base_bg, render_blocks, eff_render_cfg)
                del base_bg

                if export_path and translated_img is not None:
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

                cache_mgr.save_page_cache(
                    img_path,
                    erased_img=None if erased_already_saved else erased_img,
                    blocks=blocks,
                    rendered_img=translated_img
                )

                self.sig_batch_progress.emit(idx + 1, total, filename, 100, "完成并已存盘")
                self.sig_item_completed.emit(img_id, {
                    "has_cache": True,
                    "blocks_count": len(blocks),
                    "export_path": export_path
                })
                success_count += 1

            except Exception as e:
                print(f"[-] Batch item failed: {filename}: {e}")
                self.sig_item_failed.emit(img_id, str(e))
                fail_count += 1
            finally:
                # Free heavy arrays from RAM and CUDA VRAM immediately after EVERY page
                original_img = None
                erased_img = None
                translated_img = None
                try:
                    cleanup_gpu_memory(trim_ram=True)
                except Exception:
                    pass
                # Brief cooperative yield so Windows DWM and Qt event loop remain responsive
                QThread.msleep(15)

        try:
            cleanup_gpu_memory(trim_ram=True)
        except Exception:
            pass
        self.sig_batch_finished.emit(success_count, fail_count)
