"""
tests/unit/test_manga_ocr_pipeline.py
Unit tests for Scheme 1: Decoupled Manga-OCR recognition pipeline.
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from app.core.ocr.manga_ocr_wrapper import MangaOCRRecognizer, get_manga_ocr
from app.core.ocr.paddle_engine import PaddleOCREngine
from desktop.core.ocr_engine import OCREngine


class TestMangaOCRRecognizer:
    def test_availability(self):
        # Should return boolean without raising
        avail = MangaOCRRecognizer.is_available()
        assert isinstance(avail, bool)

    def test_empty_crop_handling(self):
        recognizer = MangaOCRRecognizer(force_cpu=True)
        assert recognizer.recognize_crop(None) == ""
        assert recognizer.recognize_crop(np.zeros((0, 0, 3), dtype=np.uint8)) == ""
        assert recognizer.recognize_crop(np.zeros((2, 2, 3), dtype=np.uint8)) == ""  # < 4px

    def test_clean_text(self):
        assert MangaOCRRecognizer._clean_text("  こんにちは   世界  ") == "こんにちは 世界"
        assert MangaOCRRecognizer._clean_text("") == ""

    def test_rotation_correction_on_slanted_crop(self):
        recognizer = MangaOCRRecognizer(force_cpu=True)
        mock_mocr = MagicMock(return_value="テスト")
        recognizer._mocr = mock_mocr

        # 50x50 white crop
        crop = np.full((50, 50, 3), 255, dtype=np.uint8)

        # Call with angle = -25.0 degrees
        res = recognizer.recognize_crop(crop, angle=-25.0)
        assert res == "テスト"
        assert mock_mocr.called
        passed_img = mock_mocr.call_args[0][0]
        assert isinstance(passed_img, Image.Image)
        # Rotated image with expand=True should be larger than 50x50
        assert passed_img.width > 50 or passed_img.height > 50

    def test_recognize_crops_batch(self):
        recognizer = MangaOCRRecognizer(force_cpu=True)
        recognizer._mocr = MagicMock(side_effect=["セリフ一", "セリフ二"])

        crop1 = np.full((30, 30, 3), 255, dtype=np.uint8)
        crop2 = np.full((30, 30, 3), 255, dtype=np.uint8)
        crops = [(crop1, 0.0), (crop2, 18.0)]

        results = recognizer.recognize_crops(crops)
        assert results == ["セリフ一", "セリフ二"]


class TestDecoupledOCREngine:
    def test_init_manga_ocr_engine(self):
        eng = OCREngine(engine_type="manga_ocr", use_gpu=False, lang="japan")
        assert eng.engine_type == "manga_ocr"
        assert eng.lang == "japan"
        assert not eng.use_gpu

    def test_decoupled_flow_japanese(self):
        eng = OCREngine(engine_type="paddle_manga", use_gpu=False, lang="japan")

        # Mock Paddle detector to return 1 box with placeholder text
        mock_paddle = MagicMock()
        mock_paddle.ocr.return_value = [[
            [[[10, 10], [100, 10], [100, 40], [10, 40]], ("raw_ocr_txt", 0.5)]
        ]]
        eng._paddle_ocr = mock_paddle

        # Mock MangaOCR to return high precision Japanese text
        mock_manga_recognizer = MagicMock()
        mock_manga_recognizer.recognize_crop.return_value = "高精度な日本語"
        eng._manga_ocr = mock_manga_recognizer

        dummy_img = np.full((200, 200, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        assert len(results) == 1
        assert results[0]["original_text"] == "高精度な日本語"
        assert results[0]["confidence"] >= 0.95
        assert mock_manga_recognizer.recognize_crop.called

    def test_decoupled_fallback_non_japanese(self):
        # When language is English, Manga-OCR should NOT overwrite native detector's text
        eng = OCREngine(engine_type="paddle_manga", use_gpu=False, lang="en")

        mock_paddle = MagicMock()
        mock_paddle.ocr.return_value = [[
            [[[10, 10], [100, 10], [100, 40], [10, 40]], ("English Text", 0.88)]
        ]]
        eng._paddle_ocr = mock_paddle

        mock_manga_recognizer = MagicMock()
        eng._manga_ocr = mock_manga_recognizer

        dummy_img = np.full((200, 200, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        assert len(results) == 1
        assert results[0]["original_text"] == "English Text"
        # Manga-OCR should NOT be called for English
        assert not mock_manga_recognizer.recognize_crop.called


class TestPaddleOCREngineWithMangaOCR:
    def test_paddle_manga_ocr_flag(self):
        eng = PaddleOCREngine(lang="japan", force_cpu=True, use_manga_ocr=True)
        assert eng.use_manga_ocr is True
        assert eng.force_cpu is True
