"""
tests/unit/test_smart_ocr_routing.py
Unit tests for SOTA multi-language OCR routing, line cropping, VLM fallback,
8GB-RAM LRU caching, config path resolution, and SettingsDialog persistence.
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.config import APP_ROOT, AppConfig, OCRConfig, resolve_app_path
from app.core.ocr.transformers_rec_wrapper import (
    _GLOBAL_REC_CACHE,
    _MAX_CACHED_MODELS,
    TransformersOCRRecognizer,
    extract_line_crops_for_box,
    get_transformers_ocr,
    join_recognized_lines,
)
from app.core.ocr.vlm_ocr import recognize_crop_with_vlm
from app.core.pipeline.utils import build_ocr_engine_kwargs
from desktop.core.ocr_engine import OCREngine


class TestAppRootResolution:
    def test_resolve_app_path_anchors_bare_filenames_to_app_root(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert os.path.normcase(resolve_app_path("desktop_config.json")) == os.path.normcase(
            os.path.join(APP_ROOT, "desktop_config.json")
        )
        assert os.path.normcase(resolve_app_path(".amt_last_session.json")) == os.path.normcase(
            os.path.join(APP_ROOT, ".amt_last_session.json")
        )
        assert os.path.isfile(os.path.join(APP_ROOT, "run_desktop.py"))
        assert os.path.isdir(os.path.join(APP_ROOT, "app"))

        explicit = str(tmp_path / "custom_config.json")
        assert resolve_app_path(explicit) == explicit

    def test_build_ocr_engine_kwargs_extracts_nested_and_flat_settings(self):
        cfg = {
            "ocr": {
                "engine": "ctd",
                "force_cpu": True,
                "en_recognizer": "ppocr_v6",
                "zh_recognizer": "ppocr_v5_server",
                "ja_recognizer": "manga_ocr",
                "paddle_ocr_version": "PP-OCRv5",
                "vlm_fallback": True,
                "vlm_conf_threshold": 0.68,
            },
            "llm": {
                "provider": "openai",
                "api_key": "test-key",
                "model": "gpt-4o",
                "endpoint": "https://api.example.com/v1",
            },
        }
        kw = build_ocr_engine_kwargs(cfg, lang="en", is_manual=True, source_lang="英语")
        assert kw["use_gpu"] is False
        assert kw["en_recognizer"] == "ppocr_v6"
        assert kw["zh_recognizer"] == "ppocr_v5_server"
        assert kw["ja_recognizer"] == "manga_ocr"
        assert kw["paddle_ocr_version"] == "PP-OCRv5"
        assert kw["vlm_fallback"] is True
        assert kw["vlm_conf_threshold"] == pytest.approx(0.68)
        assert kw["vlm_llm_config"].model == "gpt-4o"


class TestLineExtractionAndCache:
    def test_extract_line_crops_orders_horizontal_and_vertical(self):
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        # Put distinct pixel intensities in upper line vs lower line to verify ordering
        img[20:45, 20:180] = 50
        img[60:85, 20:180] = 150
        box_h = {
            "xmin": 15, "ymin": 15, "xmax": 185, "ymax": 90,
            "lines": [
                [[20, 60], [180, 60], [180, 85], [20, 85]],  # Lower line listed out-of-order first
                [[20, 20], [180, 20], [180, 45], [20, 45]],  # Upper line
            ],
        }
        strips_h = extract_line_crops_for_box(img, box=box_h)
        assert len(strips_h) == 2
        assert strips_h[0].mean() < strips_h[1].mean()  # Upper dark line comes first

        # Vertical CJK box: right-to-left column ordering & rotated 90 deg CCW (width > height)
        img_v = np.full((300, 300, 3), 255, dtype=np.uint8)
        img_v[20:200, 220:250] = 40   # Right column
        img_v[20:200, 140:170] = 160  # Left column
        box_v = {
            "xmin": 135, "ymin": 15, "xmax": 255, "ymax": 205,
            "lines": [
                [[140, 20], [170, 20], [170, 200], [140, 200]],
                [[220, 20], [250, 20], [250, 200], [220, 200]],
            ],
        }
        strips_v = extract_line_crops_for_box(img_v, box=box_v)
        assert len(strips_v) == 2
        assert strips_v[0].shape[1] > strips_v[0].shape[0]
        assert strips_v[0].mean() < strips_v[1].mean()  # Right column comes first

    def test_extract_line_crops_avoids_90deg_rotation_and_merges_same_row(self):
        img = np.full((120, 700, 3), 255, dtype=np.uint8)
        # Same-row horizontally split segments listed right-before-left with slight y jitter
        box = {
            "xmin": 50, "ymin": 30, "xmax": 600, "ymax": 60,
            "lines": [
                [[310, 34], [580, 34], [580, 56], [310, 56]],
                [[60, 35], [300, 35], [300, 57], [60, 57]],
            ],
        }
        strips = extract_line_crops_for_box(img, box=box)
        # Adjacent fragments on the same row merge cleanly into 1 wide horizontal strip (not rotated!)
        assert len(strips) == 1
        assert strips[0].shape[1] > 500
        assert strips[0].shape[0] < 40

    def test_join_recognized_lines_handles_en_cjk_and_speakers(self):
        assert join_recognized_lines(["Very well.", "Take off everything"]) == "Very well. Take off everything"
        assert join_recognized_lines(["今天天气", "真好啊"]) == "今天天气真好啊"
        assert join_recognized_lines(["(Podo) Go now.", "(Yerin) Wait!"]) == "(Podo) Go now.\n(Yerin) Wait!"

    def test_lru_cache_evicts_oldest_model_to_protect_ram(self):
        _GLOBAL_REC_CACHE.clear()
        unload_calls = []

        r1 = get_transformers_ocr("ppocr_v6", use_gpu=False)
        r2 = get_transformers_ocr("ppocr_v5_server", use_gpu=False)
        r1.unload = lambda: unload_calls.append("r1")
        r2.unload = lambda: unload_calls.append("r2")

        # Touch r1 so r2 becomes LRU
        get_transformers_ocr("ppocr_v6", use_gpu=False)
        r3 = get_transformers_ocr("korean_ppocr_v5", use_gpu=False)

        assert len(_GLOBAL_REC_CACHE) == _MAX_CACHED_MODELS
        assert unload_calls == ["r2"]
        _GLOBAL_REC_CACHE.clear()


class TestSmartLanguageRoutingAndVLM:
    @pytest.mark.parametrize(
        "lang,expected_key,sample_text",
        [
            ("en", "ppocr_v6", '(Podo) "Very well."'),
            ("chs", "ppocr_v5_server", "我们现在就带你去牢房。"),
            ("cht", "ppocr_v5_server", "這是繁體中文漫畫測試！"),
            ("korean", "korean_ppocr_v5", "잠깐만요!"),
        ],
    )
    def test_ctd_routes_each_language_to_dedicated_best_recognizer(self, lang, expected_key, sample_text):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang=lang)
        eng.enable_ensemble_detection = False
        eng.enable_ensemble_recognition = False

        mock_ctd = MagicMock()
        mock_ctd.detect.return_value = ([
            {
                "xmin": 10, "ymin": 10, "xmax": 210, "ymax": 60,
                "text": "", "conf": 0.92,
                "polygon": [[10, 10], [210, 10], [210, 60], [10, 60]],
                "angle": 0.0,
                "lines": [[[12, 12], [208, 12], [208, 58], [12, 58]]],
            }
        ], None)
        eng._ctd_detector = mock_ctd

        mock_rec = MagicMock()
        mock_rec.recognize_box.return_value = (sample_text, 0.96)
        eng._transformer_recognizers[expected_key] = mock_rec

        img = np.full((120, 260, 3), 255, dtype=np.uint8)
        blocks = eng.detect_and_recognize(img)
        assert len(blocks) == 1
        assert blocks[0]["original_text"] == sample_text
        mock_rec.recognize_box.assert_called_once()

    def test_japanese_custom_recognizer_option_can_override_manga_ocr(self):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="japan", ja_recognizer="ppocr_v5_server")
        mock_rec = MagicMock()
        mock_rec.recognize_box.return_value = ("こんにちは", 0.93)
        eng._transformer_recognizers["ppocr_v5_server"] = mock_rec

        crop = np.full((40, 120, 3), 255, dtype=np.uint8)
        txt, conf = eng.recognize_crop(crop, lang="ja")
        assert txt == "こんにちは"
        assert conf == pytest.approx(0.93)

    def test_transformer_failure_falls_back_gracefully_to_easyocr(self, monkeypatch):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="en")
        monkeypatch.setattr(
            eng, "_get_transformer_rec", lambda _key: (_ for _ in ()).throw(RuntimeError("offline cache miss"))
        )
        fallback_reader = MagicMock()
        fallback_reader.readtext.return_value = [([[0, 0], [80, 0], [80, 20], [0, 20]], "Fallback Line", 0.88)]
        eng._easyocr_en_reader = fallback_reader
        eng._easyocr_readers["en"] = fallback_reader
        eng._owned_easyocr_ids.add(id(fallback_reader))  # Pretend EasyOCR is live runtime reader

        crop = np.full((40, 100, 3), 255, dtype=np.uint8)
        txt, conf = eng.recognize_crop(crop, lang="en")
        assert txt == "Fallback Line"
        assert conf == pytest.approx(0.88)

    def test_empty_transformer_result_does_not_initialize_easyocr(self, monkeypatch):
        eng = OCREngine(engine_type="ctd", use_gpu=False, lang="en")
        mock_rec = MagicMock()
        mock_rec.recognize_box.return_value = ("", 0.0)
        eng._transformer_recognizers["ppocr_v6"] = mock_rec

        monkeypatch.setattr(
            eng, "_read_block_easyocr", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("EasyOCR should not run"))
        )
        crop = np.full((40, 100, 3), 255, dtype=np.uint8)
        txt, conf = eng.recognize_crop(crop, lang="en")
        assert txt == ""
        assert conf == 0.0

    def test_vlm_fallback_triggers_only_below_confidence_threshold(self, monkeypatch):
        calls = []

        def fake_vlm(crop, llm_cfg, source_lang="auto", timeout_seconds=15.0):
            calls.append((source_lang, llm_cfg.model))
            return "Clarified Bubble Text", 0.92

        monkeypatch.setattr("app.core.ocr.vlm_ocr.recognize_crop_with_vlm", fake_vlm)
        llm_obj = SimpleNamespace(provider="openai", api_key="k", model="gpt-4o", endpoint="https://api.example.com/v1", proxy_url="")

        eng = OCREngine(
            engine_type="ctd", use_gpu=False, lang="en",
            vlm_fallback=True, vlm_conf_threshold=0.75, vlm_llm_config=llm_obj,
        )
        mock_rec = MagicMock()
        mock_rec.recognize_box.return_value = ("garbled text", 0.42)
        eng._transformer_recognizers["ppocr_v6"] = mock_rec

        crop = np.full((40, 120, 3), 255, dtype=np.uint8)
        txt, conf = eng.recognize_crop(crop, lang="en")
        assert txt == "Clarified Bubble Text"
        assert conf == pytest.approx(0.92)
        assert len(calls) == 1

        # High confidence should skip VLM API completely
        mock_rec.recognize_box.return_value = ("Crystal Clear Text", 0.96)
        txt2, conf2 = eng.recognize_crop(crop, lang="en")
        assert txt2 == "Crystal Clear Text"
        assert conf2 == pytest.approx(0.96)
        assert len(calls) == 1

    def test_vlm_http_openai_and_failure_safety(self, monkeypatch):
        crop = np.full((40, 120, 3), 255, dtype=np.uint8)

        class DummyResp:
            def __init__(self, code, payload):
                self.status_code = code
                self._payload = payload
                self.text = str(payload)

            def json(self):
                return self._payload

        monkeypatch.setattr(
            "app.core.ocr.vlm_ocr.requests.post",
            lambda *a, **kw: DummyResp(200, {"choices": [{"message": {"content": '(Yerin) "Really?"'}}]}),
        )
        cfg = SimpleNamespace(provider="openai", api_key="secret", model="gpt-4o", endpoint="https://api.openai.com/v1", proxy_url="")
        txt, conf = recognize_crop_with_vlm(crop, cfg, source_lang="en")
        assert txt == '(Yerin) "Really?"'
        assert conf >= 0.90

        monkeypatch.setattr("app.core.ocr.vlm_ocr.requests.post", lambda *a, **kw: DummyResp(500, {"error": "boom"}))
        txt_err, conf_err = recognize_crop_with_vlm(crop, cfg, source_lang="en")
        assert txt_err == ""
        assert conf_err == 0.0


class TestSettingsDialogSmartOCRPersistence:
    def test_settings_dialog_saves_per_language_ocr_and_vlm_config(self, qapp, monkeypatch, tmp_path):
        from app.ui.settings.settings_dialog import SettingsDialog

        cfg = AppConfig()
        saved_path = tmp_path / "desktop_config.json"
        orig_save = cfg.save
        monkeypatch.setattr(cfg, "save", lambda _p: orig_save(str(saved_path)))

        dlg = SettingsDialog(config=cfg)
        dlg.en_rec_combo.setCurrentIndex(dlg.en_rec_combo.findData("ppocr_v6"))
        dlg.zh_rec_combo.setCurrentIndex(dlg.zh_rec_combo.findData("ppocr_v6"))
        dlg.ja_rec_combo.setCurrentIndex(dlg.ja_rec_combo.findData("ppocr_v5_server"))
        dlg.paddle_ver_combo.setCurrentIndex(dlg.paddle_ver_combo.findData("PP-OCRv3"))
        dlg.vlm_fallback_cb.setChecked(True)
        dlg.vlm_thresh_spin.setValue(0.65)
        dlg._save_to_config()

        reloaded = AppConfig.load(str(saved_path))
        assert reloaded.ocr.en_recognizer == "ppocr_v6"
        assert reloaded.ocr.zh_recognizer == "ppocr_v6"
        assert reloaded.ocr.ja_recognizer == "ppocr_v5_server"
        assert reloaded.ocr.paddle_ocr_version == "PP-OCRv3"
        assert reloaded.ocr.vlm_fallback is True
        assert reloaded.ocr.vlm_conf_threshold == pytest.approx(0.65)
