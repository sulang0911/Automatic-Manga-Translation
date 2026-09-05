import pytest
import numpy as np
from desktop.core.ocr_engine import (
    compute_box_iou,
    fuse_detected_boxes,
    ensemble_recognize_text,
    OCREngine
)
from desktop.core.config_manager import ConfigManager, DEFAULT_CONFIG
from app.core.config import AppConfig, OCRConfig


class TestBoxIoUAndCoverage:
    def test_non_overlapping_boxes(self):
        b1 = {"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50}
        b2 = {"xmin": 100, "ymin": 100, "xmax": 150, "ymax": 150}
        iou, cov = compute_box_iou(b1, b2)
        assert iou == 0.0
        assert cov == 0.0

    def test_identical_boxes(self):
        b1 = {"xmin": 10, "ymin": 20, "xmax": 80, "ymax": 90}
        b2 = {"xmin": 10, "ymin": 20, "xmax": 80, "ymax": 90}
        iou, cov = compute_box_iou(b1, b2)
        assert pytest.approx(iou, 0.01) == 1.0
        assert pytest.approx(cov, 0.01) == 1.0

    def test_contained_box_high_coverage(self):
        # b2 is a smaller box inside b1
        b1 = {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}
        b2 = {"xmin": 20, "ymin": 20, "xmax": 60, "ymax": 60}
        iou, cov = compute_box_iou(b1, b2)
        assert iou < 0.35  # IoU is modest because b1 is much larger
        assert pytest.approx(cov, 0.01) == 1.0  # Coverage of b2 is 100%


class TestFuseDetectedBoxes:
    def test_overlapping_box_deduplication(self):
        p_boxes = [
            {
                "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200,
                "text": "こんにちは", "conf": 0.95,
                "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "angle": 0.0
            }
        ]
        s_boxes = [
            {
                "xmin": 105, "ymin": 105, "xmax": 195, "ymax": 195,
                "text": "konnichiwa", "conf": 0.70
            }
        ]
        fused = fuse_detected_boxes(p_boxes, s_boxes)
        assert len(fused) == 1
        assert fused[0]["text"] == "こんにちは"
        assert fused[0]["sec_text"] == "konnichiwa"
        assert fused[0]["sec_conf"] == 0.70

    def test_missed_rectangular_box_recovered(self):
        p_boxes = [
            {
                "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150,
                "text": "氣泡文字", "conf": 0.90,
                "polygon": [[50, 50], [150, 50], [150, 150], [50, 150]],
                "angle": 0.0
            }
        ]
        s_boxes = [
            {
                "xmin": 400, "ymin": 500, "xmax": 550, "ymax": 580,
                "text": "CHAPTER 1", "conf": 0.88
            }
        ]
        fused = fuse_detected_boxes(p_boxes, s_boxes)
        assert len(fused) == 2
        recovered = fused[1]
        assert recovered["text"] == "CHAPTER 1"
        assert recovered["xmin"] == 400
        assert "polygon" in recovered
        assert len(recovered["polygon"]) == 4

    def test_tiny_noise_boxes_discarded(self):
        p_boxes = []
        s_boxes = [
            {"xmin": 10, "ymin": 10, "xmax": 15, "ymax": 15, "text": ".", "conf": 0.5},  # 5x5: too small
            {"xmin": 50, "ymin": 50, "xmax": 120, "ymax": 80, "text": "SIGN", "conf": 0.8}
        ]
        fused = fuse_detected_boxes(p_boxes, s_boxes, min_w=12, min_h=10)
        assert len(fused) == 1
        assert fused[0]["text"] == "SIGN"


class TestEnsembleRecognizeText:
    def test_alphanumeric_anti_hallucination(self):
        # Case 1: Manga-OCR hallucinating Japanese Kana on English name / age
        text_pri = "アリス(19)"
        conf_pri = 0.95
        text_sec = "Chris (19)"
        conf_sec = 0.88
        res_t, res_c = ensemble_recognize_text(text_pri, conf_pri, text_sec, conf_sec, target_lang="japan")
        assert res_t == "Chris (19)"
        assert res_c == 0.88

    def test_english_sfx_anti_hallucination(self):
        text_pri = "ハハ"
        conf_pri = 0.92
        text_sec = "Haha"
        conf_sec = 0.85
        res_t, res_c = ensemble_recognize_text(text_pri, conf_pri, text_sec, conf_sec, target_lang="japan")
        assert res_t == "Haha"

    def test_authentic_cjk_preservation(self):
        # Manga-OCR recognizes authentic Japanese dialogue, secondary is garbled English
        text_pri = "待ってくれ！"
        conf_pri = 0.95
        text_sec = "mattekure"
        conf_sec = 0.60
        res_t, res_c = ensemble_recognize_text(text_pri, conf_pri, text_sec, conf_sec, target_lang="japan")
        assert res_t == "待ってくれ！"
        assert res_c == 0.95

    def test_agreement_confidence_boost(self):
        text_pri = "CHAPTER 5"
        conf_pri = 0.90
        text_sec = "CHAPTER 5"
        conf_sec = 0.85
        res_t, res_c = ensemble_recognize_text(text_pri, conf_pri, text_sec, conf_sec)
        assert res_t == "CHAPTER 5"
        assert res_c >= 0.95

    def test_primary_empty_fallback_to_secondary(self):
        text_pri = ""
        conf_pri = 0.0
        text_sec = "WARNING"
        conf_sec = 0.78
        res_t, res_c = ensemble_recognize_text(text_pri, conf_pri, text_sec, conf_sec)
        assert res_t == "WARNING"
        assert res_c == 0.78


class TestEnsembleConfigIntegration:
    def test_default_config_has_ensemble_keys(self):
        assert "ocr_ensemble_detection" in DEFAULT_CONFIG
        assert "ocr_ensemble_recognition" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["ocr_ensemble_detection"] is False
        assert DEFAULT_CONFIG["ocr_ensemble_recognition"] is False

    def test_app_config_serialization(self):
        cfg = AppConfig()
        cfg.ocr.ensemble_detection = True
        cfg.ocr.ensemble_recognition = True
        d = cfg.to_dict()
        assert d["ocr"]["ensemble_detection"] is True
        assert d["ocr"]["ensemble_recognition"] is True

        loaded = AppConfig.from_dict(d)
        assert loaded.ocr.ensemble_detection is True
        assert loaded.ocr.ensemble_recognition is True

    def test_ocr_engine_initialization_with_ensemble(self):
        eng = OCREngine(
            engine_type="ctd",
            use_gpu=False,
            enable_ensemble_detection=True,
            enable_ensemble_recognition=True
        )
        assert eng.enable_ensemble_detection is True
        assert eng.enable_ensemble_recognition is True


class TestOCREngineEnsemblePipeline:
    def test_ctd_ensemble_detection_fuses_secondary_box(self):
        from unittest.mock import MagicMock
        eng = OCREngine(
            engine_type="ctd",
            use_gpu=False,
            lang="japan",
            enable_ensemble_detection=True,
            enable_ensemble_recognition=False
        )

        # Mock CTD detector returning 1 bubble
        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([
            {
                "xmin": 20, "ymin": 20, "xmax": 80, "ymax": 80,
                "text": "こんにちは", "conf": 0.95,
                "polygon": [[20, 20], [80, 20], [80, 80], [20, 80]],
                "angle": 0.0
            }
        ], None)
        eng._ctd_detector = mock_ctd

        # Mock EasyOCR returning a secondary narration box missed by CTD
        mock_easyocr = MagicMock()
        # EasyOCR readtext returns [(bbox, text, conf)]
        mock_easyocr.readtext.return_value = [
            ([[120, 120], [180, 120], [180, 160], [120, 160]], "CHAPTER 1", 0.90)
        ]
        eng._easyocr_reader = mock_easyocr

        # Mock Manga-OCR
        mock_manga = MagicMock()
        mock_manga.recognize_crop.side_effect = ["こんにちは", "CHAPTER 1"]
        eng._manga_ocr = mock_manga

        dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        # Should contain both the CTD bubble and the recovered EasyOCR box
        assert len(results) == 2
        texts = [b["original_text"] for b in results]
        assert "こんにちは" in texts
        assert "CHAPTER 1" in texts

    def test_ctd_ensemble_recognition_arbitrates_text(self):
        from unittest.mock import MagicMock
        eng = OCREngine(
            engine_type="ctd",
            use_gpu=False,
            lang="japan",
            enable_ensemble_detection=False,
            enable_ensemble_recognition=True
        )

        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([
            {
                "xmin": 30, "ymin": 30, "xmax": 120, "ymax": 80,
                "text": "temp", "conf": 0.90,
                "polygon": [[30, 30], [120, 30], [120, 80], [30, 80]],
                "angle": 0.0
            }
        ], None)
        eng._ctd_detector = mock_ctd

        # Manga-OCR hallucinates Kana on English name
        mock_manga = MagicMock()
        mock_manga.recognize_crop.return_value = "アリス(19)"
        eng._manga_ocr = mock_manga

        # EasyOCR correctly recognizes English name
        mock_easyocr = MagicMock()
        mock_easyocr.readtext.return_value = [
            ([[0, 0], [90, 0], [90, 50], [0, 50]], "Chris (19)", 0.88)
        ]
        eng._easyocr_reader = mock_easyocr

        dummy_img = np.full((200, 200, 3), 255, dtype=np.uint8)
        results = eng.detect_and_recognize(dummy_img)

        assert len(results) == 1
        # Arbitrated to Chris (19) instead of Kana hallucination
        assert results[0]["original_text"] == "Chris (19)"

