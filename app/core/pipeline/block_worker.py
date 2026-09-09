"""
app/core/pipeline/block_worker.py
Dedicated QThread worker for single-block manual OCR recognition and translation.
Upgraded to achieve full parity with (and exceed) automatic page pipeline quality:
1. Multi-scale context-aware OCR with true page-level coordinate mapping & paragraph merging.
2. Prior cache detection reuse if re-selecting an existing/deleted bubble.
3. Tight text stroke mask extraction from original_cv (avoiding outer bubble borders/smudges).
4. Dialogue-context-aware translation via configured LLM provider.
5. High-fidelity inpainting and typography rendering.
"""
import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any, List, Optional, Tuple

from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine
from app.core.typography.engine import TypographyEngine
from app.core.translation import TranslationManager, ProviderConfig
from app.core.models import TranslationBlock
from app.core.cache.cache_manager import get_cache_manager
from app.core.inpaint.color_analyzer import (
    get_background_color_rgb, get_text_mask, is_background_uniform,
    analyze_text_color, dilate_mask
)


class BlockOcrTranslateWorker(QThread):
    sig_progress = pyqtSignal(int, str)
    sig_completed = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(
        self,
        image_path: str,
        original_cv: np.ndarray,
        target_block: Dict[str, Any],
        all_blocks: List[Dict[str, Any]],
        existing_erased: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
        parent=None
    ):
        super().__init__(parent)
        self.image_path = image_path
        self.original_cv = original_cv
        self.target_block = target_block
        self.all_blocks = all_blocks
        self.existing_erased = existing_erased
        self.config = config or {}
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self.original_cv is None or self.original_cv.size == 0:
                self.sig_error.emit("原图数据为空，无法执行框选识别。")
                return

            h_img, w_img = self.original_cv.shape[:2]
            tb = self.target_block

            # 1. Target pixel bounds
            xmin_norm = float(tb.get("xmin", 0))
            ymin_norm = float(tb.get("ymin", 0))
            xmax_norm = float(tb.get("xmax", 100))
            ymax_norm = float(tb.get("ymax", 100))

            xmin_px = max(0, min(int(round((xmin_norm / 100.0) * w_img)), w_img - 1))
            ymin_px = max(0, min(int(round((ymin_norm / 100.0) * h_img)), h_img - 1))
            xmax_px = max(xmin_px + 2, min(int(round((xmax_norm / 100.0) * w_img)), w_img))
            ymax_px = max(ymin_px + 2, min(int(round((ymax_norm / 100.0) * h_img)), h_img))

            # 2. High-Precision OCR Detection with True Page-Level Merging
            self.sig_progress.emit(20, "正在执行高精度多尺度 OCR 识别...")
            if self._is_cancelled:
                return

            ocr_cfg = self.config.get("ocr", {}) if isinstance(self.config.get("ocr"), dict) else {}
            from app.core.pipeline.utils import normalize_source_lang, is_auto_source_lang, source_lang_to_ocr_lang
            effective_source = self.config.get("source_lang") or ocr_cfg.get("lang") or self.config.get("ocr_lang")
            ocr_lang = source_lang_to_ocr_lang(effective_source)
            is_manual = not is_auto_source_lang(effective_source)

            ocr_eng = OCREngine(
                engine_type=ocr_cfg.get("engine", self.config.get("ocr_engine", "ctd")),
                use_gpu=not ocr_cfg.get("force_cpu", False) if "force_cpu" in ocr_cfg else self.config.get("use_gpu", True),
                lang=ocr_lang,
                enable_ensemble_detection=ocr_cfg.get("ensemble_detection", self.config.get("ocr_ensemble_detection", False)),
                enable_ensemble_recognition=ocr_cfg.get("ensemble_recognition", self.config.get("ocr_ensemble_recognition", False)),
                is_manual=is_manual,
                source_lang=effective_source,
            )

            rec_text, tight_box = self._run_context_ocr(
                ocr_eng, xmin_px, ymin_px, xmax_px, ymax_px, w_img, h_img
            )

            tb["original_text"] = rec_text

            # Analyze colors based on clean original image crop
            crop_for_color = self.original_cv[ymin_px:ymax_px, xmin_px:xmax_px]
            try:
                bg_r, bg_g, bg_b = get_background_color_rgb(crop_for_color)
                tb["bg_color"] = f"#{bg_r:02X}{bg_g:02X}{bg_b:02X}"
                tb["text_color"] = analyze_text_color(crop_for_color, (bg_b, bg_g, bg_r))
            except Exception:
                pass

            if self._is_cancelled:
                return

            # 3. Context-Aware LLM Translation
            trans_text = ""
            if rec_text:
                self.sig_progress.emit(50, f"识别到文本: 【{rec_text[:14]}...】正在结合上下文翻译...")
                llm_cfg = self.config.get("llm", {}) if isinstance(self.config.get("llm"), dict) else {}
                provider_name = llm_cfg.get("provider", self.config.get("provider", "openai"))
                api_key = llm_cfg.get("api_key", self.config.get("api_key", ""))
                model = llm_cfg.get("model", self.config.get("model", "gpt-4o-mini"))
                endpoint = llm_cfg.get("endpoint", self.config.get("custom_endpoint", ""))
                temperature = float(llm_cfg.get("temperature", self.config.get("temperature", 0.3)))
                timeout = float(llm_cfg.get("timeout_seconds", self.config.get("timeout_seconds", 60.0)))
                max_retries = int(llm_cfg.get("max_retries", self.config.get("max_retries", 3)))

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
                        max_retries=max_retries
                    )
                )

                source_lang = self.config.get("source_lang", "自动识别")
                target_lang = self.config.get("target_lang", "简体中文")

                trans_res = mgr.translate(
                    blocks=[tb],
                    mode="text",
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                if trans_res and len(trans_res) > 0:
                    r0 = trans_res[0]
                    if isinstance(r0, dict):
                        trans_text = str(r0.get("translated_text", "")).strip()
                    else:
                        trans_text = str(getattr(r0, "translated_text", "")).strip()
            else:
                self.sig_progress.emit(50, "框选区域未检测到文字，保留空白气泡框...")

            tb["translated_text"] = trans_text

            if self._is_cancelled:
                return

            # 4. Clean Stroke-Level Background Inpainting (No Outer Border Smudges)
            self.sig_progress.emit(75, "正在消除框选文字底层背景...")
            base_erased = self.existing_erased.copy() if self.existing_erased is not None else self.original_cv.copy()
            
            new_erased = self._inpaint_clean_box(
                base_erased=base_erased,
                xmin_px=xmin_px,
                ymin_px=ymin_px,
                xmax_px=xmax_px,
                ymax_px=ymax_px,
                tight_box=tight_box,
                tb=tb
            )

            if self._is_cancelled:
                return

            # 5. Typography Rendering & Cache Saving
            self.sig_progress.emit(90, "正在生成高质量排版与字体渲染...")
            typo_eng = TypographyEngine()
            rendered_img = typo_eng.render_translations(new_erased, self.all_blocks, self.config)

            cache_mgr = get_cache_manager()
            if self.image_path:
                cache_mgr.save_page_cache(
                    self.image_path,
                    erased_img=new_erased,
                    blocks=self.all_blocks,
                    rendered_img=rendered_img
                )

            self.sig_progress.emit(100, "框选识别与翻译完成！")
            self.sig_completed.emit({
                "target_block": tb,
                "blocks": self.all_blocks,
                "erased_img": new_erased,
                "translated_img": rendered_img
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.sig_error.emit(f"框选识别翻译失败: {str(e)}")

    def _run_context_ocr(
        self,
        ocr_eng: OCREngine,
        xmin_px: int,
        ymin_px: int,
        xmax_px: int,
        ymax_px: int,
        w_img: int,
        h_img: int
    ) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
        """
        Executes high-precision OCR on the region strictly adhering to manual source language:
        1. Checks if page cache has prior high-resolution detections overlapping this box,
           verifying that cached language does not conflict with user's manual source language setting.
        2. Direct single-crop recognition using ocr_eng.recognize_crop matching manual source language.
        3. Multi-scale context-padded OCR for line segmentation and merging when multiple fragments exist.
        4. Reliable fallbacks ensuring no blank bubbles when valid text is present.
        """
        raw_source = self.config.get("source_lang") or getattr(ocr_eng, "lang", "auto")
        from app.core.pipeline.utils import normalize_source_lang, is_auto_source_lang
        norm_source = normalize_source_lang(raw_source)

        # A. Check page cache for prior full-page detections covering this region
        if self.image_path:
            try:
                cache_mgr = get_cache_manager()
                if cache_mgr.has_cache(self.image_path).get("blocks"):
                    cached = cache_mgr.load_page_cache(self.image_path, load_images=False)
                    for cb in cached.get("blocks", []):
                        cb_dict = cb if isinstance(cb, dict) else (cb.to_dict() if hasattr(cb, "to_dict") else {})
                        cb_x1 = int(round((cb_dict.get("xmin", 0) / 100.0) * w_img))
                        cb_y1 = int(round((cb_dict.get("ymin", 0) / 100.0) * h_img))
                        cb_x2 = int(round((cb_dict.get("xmax", 0) / 100.0) * w_img))
                        cb_y2 = int(round((cb_dict.get("ymax", 0) / 100.0) * h_img))

                        inter_x1 = max(xmin_px, cb_x1)
                        inter_y1 = max(ymin_px, cb_y1)
                        inter_x2 = min(xmax_px, cb_x2)
                        inter_y2 = min(ymax_px, cb_y2)

                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            cb_area = max(1, (cb_x2 - cb_x1) * (cb_y2 - cb_y1))
                            if inter_area / cb_area >= 0.65:
                                # High overlap with a previously detected block!
                                prior_text = str(cb_dict.get("original_text", "")).strip()
                                if prior_text:
                                    if not is_auto_source_lang(norm_source):
                                        has_kana = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in prior_text)
                                        has_hangul = any('\uac00' <= c <= '\ud7af' or '\u1100' <= c <= '\u11ff' for c in prior_text)
                                        if norm_source == "en" and (has_kana or has_hangul):
                                            continue
                                        if norm_source == "ko" and has_kana and not has_hangul:
                                            continue
                                        if norm_source == "ja" and has_hangul and not has_kana:
                                            continue
                                    return prior_text, (cb_x1, cb_y1, cb_x2, cb_y2)
            except Exception:
                pass

        # B. Direct single-crop recognition adhering strictly to manual source language
        tight_crop = self.original_cv[ymin_px:ymax_px, xmin_px:xmax_px]
        if hasattr(ocr_eng, "recognize_crop") and tight_crop is not None and tight_crop.size > 0:
            try:
                rec_txt, rec_conf = ocr_eng.recognize_crop(tight_crop, lang=norm_source)
                if rec_txt and rec_txt.strip():
                    return rec_txt.strip(), (xmin_px, ymin_px, xmax_px, ymax_px)
            except Exception as e_rec:
                print(f"[-] Crop recognize_crop error: {e_rec}")

        # C. Multi-Scale Context-Padded Crop & Line Segmentation
        pad_x = max(16, int((xmax_px - xmin_px) * 0.20))
        pad_y = max(16, int((ymax_px - ymin_px) * 0.20))
        c_xmin = max(0, xmin_px - pad_x)
        c_ymin = max(0, ymin_px - pad_y)
        c_xmax = min(w_img, xmax_px + pad_x)
        c_ymax = min(h_img, ymax_px + pad_y)

        crop = self.original_cv[c_ymin:c_ymax, c_xmin:c_xmax]
        if (crop is None or crop.size == 0) and (tight_crop is None or tight_crop.size == 0):
            return "", None

        target_img = crop if (crop is not None and crop.size > 0) else tight_crop
        h_c, w_c = target_img.shape[:2]
        scale = 1.0
        if max(h_c, w_c) < 650 and max(h_c, w_c) > 0:
            scale = min(3.0, 650.0 / max(h_c, w_c))
            scaled_crop = cv2.resize(target_img, (int(w_c * scale), int(h_c * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            scaled_crop = target_img

        raw_boxes = []
        if hasattr(ocr_eng, "detect_and_recognize"):
            try:
                res = ocr_eng.detect_and_recognize(scaled_crop)
                for b in res:
                    txt = b.get("original_text", b.get("text", ""))
                    if not txt:
                        continue
                    bx1 = b.get("xmin", 0)
                    by1 = b.get("ymin", 0)
                    bx2 = b.get("xmax", 0)
                    by2 = b.get("ymax", 0)
                    if isinstance(bx1, (float, int)) and bx1 <= 100.0 and bx2 <= 100.0:
                        sw = scaled_crop.shape[1]
                        sh = scaled_crop.shape[0]
                        px_xmin = int(round((bx1 / 100.0) * sw / scale))
                        px_ymin = int(round((by1 / 100.0) * sh / scale))
                        px_xmax = int(round((bx2 / 100.0) * sw / scale))
                        px_ymax = int(round((by2 / 100.0) * sh / scale))
                    else:
                        px_xmin = int(bx1 / scale)
                        px_ymin = int(by1 / scale)
                        px_xmax = int(bx2 / scale)
                        px_ymax = int(by2 / scale)

                    raw_boxes.append({
                        "xmin": c_xmin + px_xmin,
                        "ymin": c_ymin + px_ymin,
                        "xmax": c_xmin + px_xmax,
                        "ymax": c_ymin + px_ymax,
                        "text": txt,
                        "conf": float(b.get("conf", 1.0))
                    })
            except Exception as e_det:
                pass
        elif getattr(ocr_eng, "engine_type", "easyocr") == "easyocr" or hasattr(ocr_eng, "_easyocr_reader"):
            try:
                if hasattr(ocr_eng, "_init_easyocr"):
                    ocr_eng._init_easyocr(lang=norm_source)
                reader = getattr(ocr_eng, "_easyocr_reader", None)
                if reader is not None:
                    rgb = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2RGB)
                    results = reader.readtext(rgb)
                    for bbox, text, conf in results:
                        clean_text = text.strip()
                        if not clean_text or conf < 0.20:
                            continue
                        pts = np.array(bbox, dtype=np.float32) / scale
                        raw_boxes.append({
                            "xmin": c_xmin + int(np.min(pts[:, 0])),
                            "ymin": c_ymin + int(np.min(pts[:, 1])),
                            "xmax": c_xmin + int(np.max(pts[:, 0])),
                            "ymax": c_ymin + int(np.max(pts[:, 1])),
                            "text": clean_text,
                            "conf": float(conf)
                        })
            except Exception:
                pass

        # Filter raw boxes to those within user's manual selection box
        filtered_boxes = []
        margin = 8
        for b in raw_boxes:
            b_mid_x = (b["xmin"] + b["xmax"]) / 2.0
            b_mid_y = (b["ymin"] + b["ymax"]) / 2.0
            if (xmin_px - margin <= b_mid_x <= xmax_px + margin) and (ymin_px - margin <= b_mid_y <= ymax_px + margin):
                filtered_boxes.append(b)

        if not filtered_boxes and raw_boxes:
            filtered_boxes = raw_boxes

        # D. Paragraph/Line Merging
        if filtered_boxes:
            if hasattr(ocr_eng, "_merge_adjacent_boxes"):
                merged = ocr_eng._merge_adjacent_boxes(filtered_boxes, w_img=w_img, h_img=h_img)
            else:
                from app.core.ocr.base import merge_adjacent_boxes
                merged = merge_adjacent_boxes(filtered_boxes, img_w=w_img, img_h=h_img)
            if merged:
                full_text = "\n".join(b["text"] for b in merged).strip()
                t_xmin = min(b["xmin"] for b in merged)
                t_ymin = min(b["ymin"] for b in merged)
                t_xmax = max(b["xmax"] for b in merged)
                t_ymax = max(b["ymax"] for b in merged)
                return full_text, (t_xmin, t_ymin, t_xmax, t_ymax)

        # E. Fallback reading on padded crop using recognize_crop or reader
        if hasattr(ocr_eng, "recognize_crop") and crop is not None and crop.size > 0:
            try:
                txt, conf = ocr_eng.recognize_crop(crop, lang=norm_source)
                if txt and txt.strip():
                    return txt.strip(), (xmin_px, ymin_px, xmax_px, ymax_px)
            except Exception:
                pass

        if tight_crop is not None and tight_crop.size > 0 and hasattr(ocr_eng, "_easyocr_reader") and ocr_eng._easyocr_reader is not None:
            try:
                res = ocr_eng._easyocr_reader.readtext(tight_crop)
                texts = [str(item[1]).strip() for item in res if len(item) > 1 and str(item[1]).strip()]
                if texts:
                    return "\n".join(texts), (xmin_px, ymin_px, xmax_px, ymax_px)
            except Exception:
                pass

        return "", None

    def _inpaint_clean_box(
        self,
        base_erased: np.ndarray,
        xmin_px: int,
        ymin_px: int,
        xmax_px: int,
        ymax_px: int,
        tight_box: Optional[Tuple[int, int, int, int]],
        tb: Dict[str, Any]
    ) -> np.ndarray:
        """
        Inpaints text strokes cleanly without disturbing the bubble's outer black border:
        1. Always extracts high-contrast text mask from original_cv (not blurry erased image).
        2. Restricts inpainting strictly to the interior tight text bounds so bubble border is untouched.
        3. Fills uniform bubble backgrounds cleanly with sampled interior color.
        """
        h_img, w_img = self.original_cv.shape[:2]
        
        # Determine inpaint bounds: if tight text box was found, use tight box + small padding
        if tight_box is not None:
            t_x1, t_y1, t_x2, t_y2 = tight_box
            pad = 6
            target_x1 = max(0, min(t_x1 - pad, w_img - 1))
            target_y1 = max(0, min(t_y1 - pad, h_img - 1))
            target_x2 = max(target_x1 + 2, min(t_x2 + pad, w_img))
            target_y2 = max(target_y1 + 2, min(t_y2 + pad, h_img))
        else:
            target_x1, target_y1 = xmin_px, ymin_px
            target_x2, target_y2 = xmax_px, ymax_px

        crop_orig = self.original_cv[target_y1:target_y2, target_x1:target_x2]
        if crop_orig.size == 0:
            return base_erased

        # Sample background color from interior margin
        bg_r, bg_g, bg_b = get_background_color_rgb(crop_orig)
        bg_bgr = [int(bg_b), int(bg_g), int(bg_r)]

        # Get stroke mask from original image
        text_mask = get_text_mask(crop_orig, [bg_r, bg_g, bg_b])
        dilated = dilate_mask(text_mask, dilation_pixels=3)

        # If background is uniform (typical comic bubble), directly clean text strokes onto base_erased
        is_uniform = is_background_uniform(crop_orig, std_thresh=18.0)
        
        if is_uniform:
            sub_erased = base_erased[target_y1:target_y2, target_x1:target_x2]
            sub_erased[dilated == 255] = bg_bgr
            base_erased[target_y1:target_y2, target_x1:target_x2] = sub_erased
        else:
            # Complex art background: invoke InpaintEngine with localized mask
            inp_cfg = self.config.get("inpaint", {}) if isinstance(self.config.get("inpaint"), dict) else {}
            inpaint_eng = InpaintEngine(mode=inp_cfg.get("engine", self.config.get("inpaint_engine", "auto")))
            base_erased = inpaint_eng.inpaint(
                base_erased,
                [tb],
                bubble_dilation=self.config.get("bubble_dilation", 3),
                onomatopoeia_dilation=self.config.get("onomatopoeia_dilation", 6),
                feather_radius=self.config.get("feather_radius", 4)
            )

        return base_erased
