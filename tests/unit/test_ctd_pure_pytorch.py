"""
tests/unit/test_ctd_pure_pytorch.py
Unit tests for Pure PyTorch Comic-Text-Detector (CTD) and smart language routing pipeline.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.ocr.ctd_engine import ComicTextDetectorEngine
from desktop.core.ocr_engine import OCREngine


class TestComicTextDetectorEngine:
    def test_availability(self):
        # Should return boolean without raising
        avail = ComicTextDetectorEngine.is_available()
        assert isinstance(avail, bool)

    def test_empty_image_handling(self):
        engine = ComicTextDetectorEngine(use_gpu=False)
        boxes, mask = engine.detect(None)
        assert boxes == []
        assert mask is None

        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        boxes, mask = engine.detect(empty_img)
        assert boxes == []
        assert mask is None

    def test_detection_and_angle_extraction(self):
        engine = ComicTextDetectorEngine(use_gpu=False)

        # Mock internal TextDetector
        mock_detector = MagicMock()
        dummy_block_normal = MagicMock()
        dummy_block_normal.xyxy = [10, 20, 100, 80]
        dummy_block_normal.angle = 3  # < 15 deg -> should clamp to 0.0
        dummy_block_normal.prob = 0.95
        dummy_block_normal.lines = [[[10, 20], [100, 20], [100, 45], [10, 45]]]

        dummy_block_tilted = MagicMock()
        dummy_block_tilted.xyxy = [120, 30, 200, 150]
        dummy_block_tilted.angle = -18.5  # >= 15 deg -> preserve -18.5
        dummy_block_tilted.prob = 0.92
        dummy_block_tilted.lines = [[[120, 30], [200, 50], [180, 150], [100, 130]]]

        mock_detector.return_value = (None, None, [dummy_block_normal, dummy_block_tilted])
        engine._detector = mock_detector

        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        boxes, _ = engine.detect(img)

        assert len(boxes) == 2
        # First box clamped to 0.0
        assert boxes[0]["angle"] == 0.0
        assert boxes[0]["xmin"] == 10
        assert boxes[0]["ymin"] == 20
        assert boxes[0]["xmax"] == 100
        assert boxes[0]["ymax"] == 80

        # Second box tilted at -18.5
        assert boxes[1]["angle"] == -18.5
        assert boxes[1]["xmin"] == 120
        assert boxes[1]["ymin"] == 30

    def test_slanted_off_bubble_line_splitting(self):
        """
        Critical bugfix verification: When CTD YOLO groups a slanted line (e.g. 20.0 deg)
        together with horizontal speech bubble lines (0.0 deg), detect() must automatically
        split them into separate text boxes instead of bundling them together.
        """
        engine = ComicTextDetectorEngine(use_gpu=False)
        mock_detector = MagicMock()

        dummy_mixed_block = MagicMock()
        dummy_mixed_block.xyxy = [100, 50, 400, 300]
        dummy_mixed_block.angle = 0.0
        dummy_mixed_block.prob = 0.96
        # 2 horizontal dialogue lines + 1 slanted off-bubble line (at ~20 deg)
        dummy_mixed_block.lines = [
            # Line 1: Horizontal dialogue
            [[120, 60], [350, 60], [350, 95], [120, 95]],
            # Line 2: Horizontal dialogue
            [[120, 105], [360, 105], [360, 140], [120, 140]],
            # Line 3: Slanted off-bubble text (dx=100, dy=36.4 -> ~20 deg)
            [[250, 200], [350, 236], [340, 260], [240, 224]],
        ]

        mock_detector.return_value = (None, None, [dummy_mixed_block])
        engine._detector = mock_detector

        img = np.full((500, 500, 3), 255, dtype=np.uint8)
        boxes, _ = engine.detect(img)

        # Must be split into 2 separate boxes
        assert len(boxes) == 2

        # One box is the horizontal bubble dialogue
        bubble_box = next(b for b in boxes if b["angle"] == 0.0)
        assert bubble_box["line_count"] == 2
        assert bubble_box["ymin"] >= 60 and bubble_box["ymax"] <= 150

        # One box is the slanted off-bubble text
        slanted_box = next(b for b in boxes if b["angle"] != 0.0)
        assert slanted_box["line_count"] == 1
        assert abs(slanted_box["angle"] - 20.0) <= 2.0

    def test_moderate_slant_angle_preserved(self):
        """Verifies that moderate slant angles (e.g. -14.0 deg) are not wiped out by the deadband."""
        engine = ComicTextDetectorEngine(use_gpu=False)
        mock_detector = MagicMock()

        dummy_tilted_14 = MagicMock()
        dummy_tilted_14.xyxy = [100, 100, 300, 180]
        dummy_tilted_14.angle = -14.0
        dummy_tilted_14.prob = 0.90
        dummy_tilted_14.lines = [
            # dx = 150, dy = -37.5 -> -14.0 deg
            [[100, 140], [250, 103], [255, 125], [105, 162]]
        ]

        mock_detector.return_value = (None, None, [dummy_tilted_14])
        engine._detector = mock_detector

        img = np.full((400, 400, 3), 255, dtype=np.uint8)
        boxes, _ = engine.detect(img)

        assert len(boxes) == 1
        assert abs(boxes[0]["angle"] - (-14.0)) <= 1.0

    def test_can_merge_pair_rejects_slanted_adjacent_to_bubble(self):
        """Verifies can_merge_pair strictly rejects merging slanted off-bubble text into horizontal dialogue."""
        from app.core.ocr.base import can_merge_pair

        # Horizontal speech bubble dialogue
        bubble_box = {
            "xmin": 100, "ymin": 100, "xmax": 300, "ymax": 180,
            "text": "Yeah, man. Whatever.", "angle": 0.0
        }
        # Adjacent slanted off-bubble text (-14 deg)
        slanted_box = {
            "xmin": 120, "ymin": 190, "xmax": 310, "ymax": 240,
            "text": "It's her BF's or something", "angle": -14.0
        }

        # Should be strictly rejected
        assert can_merge_pair(bubble_box, slanted_box, img_w=1000, img_h=1000) is False


class TestPurePyTorchOCREngine:
    def test_ctd_engine_init(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="japan")
        assert eng.engine_type == "ctd"
        assert not eng.use_gpu

    def test_english_comic_auto_routes_to_easyocr(self):
        """
        Critical test: Verifies that when an English comic is loaded,
        the system automatically switches to EasyOCR and extracts clean English,
        preventing Manga-OCR from turning English into Japanese kana gibberish.
        """
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="japan")

        # Mock CTD detector returning 1 dialogue bubble
        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([{
            "xmin": 30, "ymin": 40, "xmax": 180, "ymax": 120,
            "text": "", "conf": 0.98, "angle": 0.0,
            "polygon": [[30, 40], [180, 40], [180, 120], [30, 120]],
            "line_count": 3
        }], None)
        eng._ctd_detector = mock_ctd

        # Mock EasyOCR returning English text
        mock_easy = MagicMock()
        mock_easy.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Wait, don't go!", 0.95),
            ([[0, 25], [100, 25], [100, 45], [0, 45]], "It is dangerous.", 0.96)
        ]
        eng._easyocr_reader = mock_easy

        # Mock MangaOCR - should NOT be called for English
        mock_mocr = MagicMock()
        eng._manga_ocr = mock_mocr

        dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        assert len(results) == 1
        assert "Wait, don't go!" in results[0]["original_text"]
        assert "It is dangerous." in results[0]["original_text"]
        # Manga-OCR should NOT have been called
        assert not mock_mocr.recognize_crop.called

    def test_japanese_comic_routes_to_manga_ocr(self):
        """
        Verifies that when Japanese dialogue is detected,
        Manga-OCR is invoked to provide high precision Japanese text.
        """
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="japan")

        # Mock CTD detector returning 1 dialogue bubble
        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([{
            "xmin": 30, "ymin": 40, "xmax": 180, "ymax": 120,
            "text": "", "conf": 0.98, "angle": 0.0,
            "polygon": [[30, 40], [180, 40], [180, 120], [30, 120]],
            "line_count": 3
        }], None)
        eng._ctd_detector = mock_ctd

        # Mock EasyOCR probe returning Japanese kana (0 english words >= 3 letters)
        mock_easy = MagicMock()
        mock_easy.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "こんにちは", 0.95)
        ]
        eng._easyocr_reader = mock_easy

        # Mock MangaOCR returning high-precision Japanese
        mock_mocr = MagicMock()
        mock_mocr.recognize_crop.return_value = "こんにちは、世界！"
        eng._manga_ocr = mock_mocr

        dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        assert len(results) == 1
        assert results[0]["original_text"] == "こんにちは、世界！"
        assert mock_mocr.recognize_crop.called
