"""
tests/unit/test_manual_source_lang_routing.py
Unit tests verifying strict enforcement of user-configured manual source languages (JA, EN, KO, CHS, CHT)
across OCR engine recognition, canvas manual box-selection OCR, reading order mode, and translation prompts.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from app.core.pipeline.utils import (
    normalize_source_lang,
    is_auto_source_lang,
    source_lang_to_ocr_lang,
)
from app.core.translation.prompt_templates import PromptTemplates
from app.core.models import TranslationBlock, ReadingOrderMode
from desktop.core.ocr_engine import OCREngine
from app.core.pipeline.block_worker import BlockOcrTranslateWorker


class TestLanguageNormalization:
    def test_canonical_mappings(self):
        assert normalize_source_lang("自动识别") == "auto"
        assert normalize_source_lang("auto") == "auto"
        assert normalize_source_lang("") == "auto"
        assert normalize_source_lang(None) == "auto"

        assert normalize_source_lang("日语") == "ja"
        assert normalize_source_lang("Japanese") == "ja"
        assert normalize_source_lang("japan") == "ja"
        assert normalize_source_lang("ja") == "ja"

        assert normalize_source_lang("英语") == "en"
        assert normalize_source_lang("English") == "en"
        assert normalize_source_lang("en") == "en"

        assert normalize_source_lang("韩语") == "ko"
        assert normalize_source_lang("Korean") == "ko"
        assert normalize_source_lang("ko") == "ko"

        assert normalize_source_lang("简体中文") == "chs"
        assert normalize_source_lang("chinese") == "chs"
        assert normalize_source_lang("chs") == "chs"

        assert normalize_source_lang("繁体中文") == "cht"
        assert normalize_source_lang("chinese_cht") == "cht"
        assert normalize_source_lang("cht") == "cht"

    def test_is_auto(self):
        assert is_auto_source_lang("自动识别") is True
        assert is_auto_source_lang("auto") is True
        assert is_auto_source_lang(None) is True
        assert is_auto_source_lang("英语") is False
        assert is_auto_source_lang("韩语") is False
        assert is_auto_source_lang("日语") is False

    def test_source_lang_to_ocr_lang(self):
        assert source_lang_to_ocr_lang("英语") == "en"
        assert source_lang_to_ocr_lang("en") == "en"
        assert source_lang_to_ocr_lang("韩语") == "korean"
        assert source_lang_to_ocr_lang("ko") == "korean"
        assert source_lang_to_ocr_lang("简体中文") == "ch"
        assert source_lang_to_ocr_lang("繁体中文") == "chinese_cht"
        assert source_lang_to_ocr_lang("日语") == "japan"
        assert source_lang_to_ocr_lang("ja") == "japan"
        assert source_lang_to_ocr_lang("自动识别") == "auto"


class TestPromptTemplatesManualSourceLang:
    def test_english_manual_prompt_rules(self):
        prompt = PromptTemplates.build_text_system_prompt("英语", "简体中文")
        assert "translating from 英语 to 简体中文" in prompt
        assert "The source language is strictly specified by the user as 英语" in prompt
        assert "Preserve English comic dialogue style" in prompt
        assert "Western comic narrative reading order" in prompt

    def test_korean_manual_prompt_rules(self):
        prompt = PromptTemplates.build_text_system_prompt("韩语", "简体中文")
        assert "translating from 韩语 to 简体中文" in prompt
        assert "The source language is strictly specified by the user as 韩语" in prompt
        assert "Preserve Korean speech hierarchy" in prompt
        assert "Webtoon continuous vertical reading order" in prompt

    def test_chinese_manual_prompt_rules(self):
        prompt = PromptTemplates.build_text_system_prompt("繁体中文", "简体中文")
        assert "The source language is strictly specified by the user as 繁体中文" in prompt
        assert "Preserve Chinese comic dialogue tone" in prompt

    def test_auto_detect_prompt_rules(self):
        prompt = PromptTemplates.build_text_system_prompt("自动识别", "简体中文")
        assert "automatically detecting the source language" in prompt
        assert "Language Auto-Detection" in prompt


class TestOCREngineManualRouting:
    def test_manual_non_english_does_not_override_language(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="korean")
        dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        is_en, detected = eng.detect_page_language(dummy_img)
        assert is_en is False
        assert detected == "ko"

    def test_manual_english_directly_returns_english(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="en")
        dummy_img = np.full((100, 100, 3), 255, dtype=np.uint8)
        is_en, detected = eng.detect_page_language(dummy_img)
        assert is_en is True
        assert detected == "en"

    def test_recognize_crop_routes_to_english_reader(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="en")
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "Comic Text", 0.92)
        ]
        eng._easyocr_en_reader = mock_reader
        eng._easyocr_readers["en"] = mock_reader

        crop = np.full((30, 80, 3), 255, dtype=np.uint8)
        txt, conf = eng.recognize_crop(crop, lang="en")
        assert "Comic Text" in txt
        assert conf >= 0.90

    def test_recognize_crop_routes_to_manga_ocr_for_japanese(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="japan")
        mock_manga = MagicMock()
        mock_manga.recognize_crop.return_value = "こんにちは"
        eng._manga_ocr = mock_manga

        crop = np.full((30, 80, 3), 255, dtype=np.uint8)
        txt, conf = eng.recognize_crop(crop, lang="ja")
        assert txt == "こんにちは"
        assert conf >= 0.90

    def test_reading_order_resolution_respects_manual_language(self):
        eng_en = OCREngine(engine_type="ctd", use_gpu=False, lang="en")
        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([
            {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 200, "text": "Box Left", "conf": 0.9},
            {"xmin": 700, "ymin": 100, "xmax": 900, "ymax": 200, "text": "Box Right", "conf": 0.9},
        ], None)
        eng_en._ctd_detector = mock_ctd

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [([[0, 0], [10, 0], [10, 10], [0, 10]], "Text", 0.9)]
        eng_en._easyocr_en_reader = mock_reader
        eng_en._easyocr_readers["en"] = mock_reader

        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        res = eng_en.detect_and_recognize(img)
        assert len(res) == 2
        box_left = next(b for b in res if b["xmin"] == 10.0)
        box_right = next(b for b in res if b["xmin"] == 70.0)
        # In Western LTR, Left is read before Right
        assert box_left["reading_order"] < box_right["reading_order"]


class TestCanvasBoxSelectionManualLanguage:
    def test_manual_box_selection_uses_user_source_language(self, monkeypatch):
        class MockTranslationManager:
            last_source_lang = None
            @classmethod
            def get_instance(cls):
                return MockTranslationManager()
            def set_active_provider(self, *args, **kwargs):
                pass
            def translate(self, blocks, **kwargs):
                MockTranslationManager.last_source_lang = kwargs.get("source_lang")
                for b in blocks:
                    b["translated_text"] = "Translated Success"
                return blocks

        monkeypatch.setattr("app.core.pipeline.block_worker.TranslationManager", MockTranslationManager)

        last_recognized_lang = []
        class MockOCREng:
            def __init__(self, *args, **kwargs):
                self.lang = kwargs.get("lang")
            def recognize_crop(self, crop, lang=None, **kwargs):
                last_recognized_lang.append(lang or self.lang)
                return "Recognized Text", 0.95

        monkeypatch.setattr("app.core.pipeline.block_worker.OCREngine", MockOCREng)

        dummy_img = np.full((500, 500, 3), 255, dtype=np.uint8)
        target_block = {
            "id": "box_1",
            "xmin": 10.0, "ymin": 10.0, "xmax": 40.0, "ymax": 30.0,
            "original_text": "", "translated_text": ""
        }

        # 1. Test Korean source language
        worker_ko = BlockOcrTranslateWorker(
            image_path="",
            original_cv=dummy_img,
            target_block=dict(target_block),
            all_blocks=[target_block],
            config={"source_lang": "韩语", "target_lang": "简体中文"}
        )
        worker_ko.run()
        assert "ko" in last_recognized_lang or "korean" in last_recognized_lang
        assert MockTranslationManager.last_source_lang == "韩语"

        # 2. Test English source language
        last_recognized_lang.clear()
        worker_en = BlockOcrTranslateWorker(
            image_path="",
            original_cv=dummy_img,
            target_block=dict(target_block),
            all_blocks=[target_block],
            config={"source_lang": "英语", "target_lang": "简体中文"}
        )
        worker_en.run()
        assert "en" in last_recognized_lang
        assert MockTranslationManager.last_source_lang == "英语"

    def test_cache_conflict_rejected_when_manual_language_differs(self, monkeypatch):
        class MockCacheManager:
            def has_cache(self, path):
                return {"blocks": True, "erased": False, "rendered": False}
            def load_page_cache(self, path, load_images=False):
                return {
                    "blocks": [
                        {
                            "id": "cached_ja",
                            "xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 30.0,
                            "original_text": "こんにちは！",
                            "translated_text": "你好！"
                        }
                    ]
                }
            def save_page_cache(self, *args, **kwargs):
                pass

        monkeypatch.setattr("app.core.pipeline.block_worker.get_cache_manager", lambda: MockCacheManager())

        class MockOCREng:
            def __init__(self, *args, **kwargs):
                pass
            def recognize_crop(self, crop, lang=None, **kwargs):
                return "Fresh English Recognition", 0.95

        monkeypatch.setattr("app.core.pipeline.block_worker.OCREngine", MockOCREng)

        dummy_img = np.full((500, 500, 3), 255, dtype=np.uint8)
        target_block = {
            "id": "box_en",
            "xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 30.0,
            "original_text": "", "translated_text": ""
        }

        worker = BlockOcrTranslateWorker(
            image_path="test.jpg",
            original_cv=dummy_img,
            target_block=target_block,
            all_blocks=[target_block],
            config={"source_lang": "英语", "target_lang": "简体中文"}
        )
        worker.run()
        assert target_block["original_text"] == "Fresh English Recognition"

    def test_canvas_box_selection_chinese_routing(self, monkeypatch):
        last_recognized_lang = []
        class MockOCREng:
            def __init__(self, *args, **kwargs):
                self.lang = kwargs.get("lang")
            def recognize_crop(self, crop, lang=None, **kwargs):
                last_recognized_lang.append(lang or self.lang)
                return "中文识别测试", 0.95

        monkeypatch.setattr("app.core.pipeline.block_worker.OCREngine", MockOCREng)
        class MockTransMgr:
            @classmethod
            def get_instance(cls):
                return MockTransMgr()
            def set_active_provider(self, *args, **kwargs):
                pass
            def translate(self, blocks, **kwargs):
                for b in blocks:
                    b["translated_text"] = "Chinese Trans Result"
                return blocks
        monkeypatch.setattr("app.core.pipeline.block_worker.TranslationManager", MockTransMgr)

        dummy_img = np.full((500, 500, 3), 255, dtype=np.uint8)
        target_block = {
            "id": "box_chs",
            "xmin": 10.0, "ymin": 10.0, "xmax": 40.0, "ymax": 30.0,
            "original_text": "", "translated_text": ""
        }
        worker = BlockOcrTranslateWorker(
            image_path="",
            original_cv=dummy_img,
            target_block=dict(target_block),
            all_blocks=[target_block],
            config={"source_lang": "简体中文", "target_lang": "英语"}
        )
        worker.run()
        assert "chs" in last_recognized_lang or "ch" in last_recognized_lang


class TestOCREngineManualVersusAuto:
    def test_manual_japanese_strictly_preserves_japanese_when_english_text_present(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="japan", is_manual=True)

        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([{
            "xmin": 30, "ymin": 40, "xmax": 180, "ymax": 120,
            "text": "", "conf": 0.98, "angle": 0.0,
            "polygon": [[30, 40], [180, 40], [180, 120], [30, 120]],
            "line_count": 3
        }], None)
        eng._ctd_detector = mock_ctd

        mock_easy = MagicMock()
        mock_easy.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Wait, don't go!", 0.95),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "It is dangerous.", 0.96)
        ]
        eng._easyocr_reader = mock_easy

        mock_mocr = MagicMock()
        mock_mocr.recognize_crop.return_value = "待って、行かないで！"
        eng._manga_ocr = mock_mocr

        dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        assert len(results) == 1
        # Manga-OCR MUST have been called because user set manual Japanese
        assert mock_mocr.recognize_crop.called
        assert results[0]["original_text"] == "待って、行かないで！"

    def test_auto_mode_with_source_lang_auto_allows_english_switch(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="auto", source_lang="自动识别")

        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([{
            "xmin": 30, "ymin": 40, "xmax": 180, "ymax": 120,
            "text": "", "conf": 0.98, "angle": 0.0,
            "polygon": [[30, 40], [180, 40], [180, 120], [30, 120]],
            "line_count": 3
        }], None)
        eng._ctd_detector = mock_ctd

        mock_easy = MagicMock()
        mock_easy.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Wait, don't go!", 0.95),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "It is dangerous.", 0.96)
        ]
        eng._easyocr_reader = mock_easy

        mock_mocr = MagicMock()
        eng._manga_ocr = mock_mocr

        dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        assert len(results) == 1
        assert "Wait, don't go!" in results[0]["original_text"]
        # Manga-OCR should NOT have been called
        assert not mock_mocr.recognize_crop.called

