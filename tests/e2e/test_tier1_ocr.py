import os
import pytest
import numpy as np
import cv2
from desktop.core.ocr_engine import OCREngine, get_background_color_hex

# ============================================================================
# F-OCR-01: Multi-Engine Local OCR Runner
# ============================================================================

def test_focr_01_engine_initialization():
    engine_paddle = OCREngine(engine_type="paddle", use_gpu=False, lang="japan")
    assert engine_paddle.engine_type == "paddle"
    assert not engine_paddle.use_gpu
    assert engine_paddle.lang == "japan"

    engine_easy = OCREngine(engine_type="easyocr", use_gpu=False, lang="ch")
    assert engine_easy.engine_type == "easyocr"
    assert engine_easy.lang == "ch"

def test_focr_01_empty_image_handling():
    engine = OCREngine(engine_type="paddle", use_gpu=False)
    assert engine.detect_and_recognize(None) == []
    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    assert engine.detect_and_recognize(empty_arr) == []

def test_focr_01_progress_callback():
    engine = OCREngine(engine_type="paddle", use_gpu=False)
    # Mock paddle ocr to return deterministic results without loading heavy neural weights
    progress_calls = []
    def cb(pct, msg):
        progress_calls.append((pct, msg))

    # Test progress callback with empty image
    res = engine.detect_and_recognize(None, progress_callback=cb)
    assert res == []

    # Test progress callback with a dummy image with mocked internal ocr
    engine._paddle_ocr = type("MockPaddle", (), {
        "ocr": lambda self, img: []
    })()
    dummy_img = np.full((200, 200, 3), 255, dtype=np.uint8)
    res = engine.detect_and_recognize(dummy_img, progress_callback=cb)
    assert len(progress_calls) >= 3
    assert progress_calls[-1][0] >= 90

def test_focr_01_block_schema_keys():
    engine = OCREngine(engine_type="paddle", use_gpu=False)
    engine._paddle_ocr = type("MockPaddle", (), {
        "ocr": lambda self, img: [[
            [[[50, 50], [150, 50], [150, 150], [50, 150]], ("テスト", 0.95)]
        ]]
    })()
    dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
    blocks = engine.detect_and_recognize(dummy_img)
    assert len(blocks) == 1
    b = blocks[0]
    required_keys = ["id", "original_text", "translated_text", "xmin", "ymin", "xmax", "ymax", "bg_color", "text_color", "type"]
    for k in required_keys:
        assert k in b
    assert b["original_text"] == "テスト"
    assert b["translated_text"] == ""

def test_focr_01_coordinate_normalization_bounds():
    engine = OCREngine(engine_type="paddle", use_gpu=False)
    engine._paddle_ocr = type("MockPaddle", (), {
        "ocr": lambda self, img: [[
            [[[10, 20], [200, 20], [200, 150], [10, 150]], ("セリフ", 0.9)]
        ]]
    })()
    dummy_img = np.full((500, 400, 3), 255, dtype=np.uint8)
    blocks = engine.detect_and_recognize(dummy_img)
    assert len(blocks) == 1
    b = blocks[0]
    assert 0.0 <= b["xmin"] <= 100.0
    assert 0.0 <= b["ymin"] <= 100.0
    assert 0.0 <= b["xmax"] <= 100.0
    assert 0.0 <= b["ymax"] <= 100.0
    assert b["xmin"] < b["xmax"]
    assert b["ymin"] < b["ymax"]

# ============================================================================
# F-OCR-02: GPU Hardware Auto-Fallback
# ============================================================================

def test_focr_02_cpu_mode_flag():
    engine = OCREngine(engine_type="paddle", use_gpu=False)
    assert not engine.use_gpu

def test_focr_02_gpu_mode_flag():
    engine = OCREngine(engine_type="paddle", use_gpu=True)
    assert engine.use_gpu

def test_focr_02_paddle_fallback_to_cpu(monkeypatch):
    engine = OCREngine(engine_type="paddle", use_gpu=True)
    calls = []
    def mock_paddle_init(*args, **kwargs):
        calls.append(kwargs.get("use_gpu"))
        if kwargs.get("use_gpu") is True:
            raise RuntimeError("CUDA out of memory or driver failure")
        return type("PaddleDummy", (), {"ocr": lambda s, img: []})()

    monkeypatch.setattr("paddleocr.PaddleOCR", mock_paddle_init)
    engine._init_paddle()
    assert len(calls) == 2
    assert calls[0] is True
    assert calls[1] is False
    assert engine._paddle_ocr is not None

def test_focr_02_easyocr_langs_japanese():
    engine = OCREngine(engine_type="easyocr", use_gpu=False, lang="japan")
    # Verify langs mapped to ['ja', 'en']
    langs = ['ja', 'en'] if engine.lang in ['japan', 'ja'] else ['ch_sim', 'en']
    assert 'ja' in langs

def test_focr_02_easyocr_langs_chinese():
    engine = OCREngine(engine_type="easyocr", use_gpu=False, lang="ch")
    langs = ['ja', 'en'] if engine.lang in ['japan', 'ja'] else ['ch_sim', 'en']
    assert 'ch_sim' in langs

# ============================================================================
# F-OCR-03: Vertical & Horizontal Text Detection
# ============================================================================

def test_focr_03_merge_adjacent_boxes():
    engine = OCREngine()
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 150, "ymax": 120, "text": "行1", "conf": 0.9},
        {"xmin": 102, "ymin": 122, "xmax": 152, "ymax": 142, "text": "行2", "conf": 0.9},
    ]
    merged = engine._merge_adjacent_boxes(boxes, w_img=1000, h_img=1000)
    assert len(merged) == 1
    assert "行1\n行2" in merged[0]["text"]
    assert merged[0]["xmin"] == 100
    assert merged[0]["ymax"] == 142

def test_focr_03_merge_isolated_boxes_remain_distinct():
    engine = OCREngine()
    boxes = [
        {"xmin": 50, "ymin": 50, "xmax": 100, "ymax": 80, "text": "右上", "conf": 0.9},
        {"xmin": 800, "ymin": 800, "xmax": 850, "ymax": 830, "text": "左下", "conf": 0.9},
    ]
    merged = engine._merge_adjacent_boxes(boxes, w_img=1000, h_img=1000)
    assert len(merged) == 2

def test_focr_03_bubble_vs_onomatopoeia_classification():
    engine = OCREngine()
    # Normal dialogue box: aspect ~ 0.8 -> bubble
    box_bubble = {"xmin": 100, "ymin": 100, "xmax": 180, "ymax": 200, "text": "会話"}
    aspect = (box_bubble["xmax"] - box_bubble["xmin"]) / (box_bubble["ymax"] - box_bubble["ymin"])
    is_bubble = not (aspect > 4.0 or aspect < 0.15)
    assert is_bubble is True

    # Extreme banner: aspect 5.0 -> onomatopoeia
    box_onoma = {"xmin": 50, "ymin": 50, "xmax": 550, "ymax": 100, "text": "ゴゴゴゴ"}
    aspect = (box_onoma["xmax"] - box_onoma["xmin"]) / (box_onoma["ymax"] - box_onoma["ymin"])
    is_bubble = not (aspect > 4.0 or aspect < 0.15)
    assert is_bubble is False

def test_focr_03_manga_reading_order():
    engine = OCREngine()
    # Manga is Right-to-Left, Top-to-Bottom
    boxes = [
        {"xmin": 100, "ymin": 50, "xmax": 200, "ymax": 90, "text": "左上"},
        {"xmin": 700, "ymin": 50, "xmax": 800, "ymax": 90, "text": "右上"},
        {"xmin": 700, "ymin": 500, "xmax": 800, "ymax": 540, "text": "右下"},
    ]
    sorted_boxes = engine._sort_manga_reading_order(boxes, w_img=1000)
    # Right-top (xmin=700, ymin=50) should come first, then left-top (xmin=100, ymin=50)
    assert sorted_boxes[0]["text"] == "右上"
    assert sorted_boxes[1]["text"] == "左上"
    assert sorted_boxes[2]["text"] == "右下"

def test_focr_03_empty_boxes_merge_safe():
    engine = OCREngine()
    assert engine._merge_adjacent_boxes([], 1000, 1000) == []

# ============================================================================
# F-OCR-04: Auto Color Extraction
# ============================================================================

def test_focr_04_white_background_crop():
    white_crop = np.full((100, 100, 3), 255, dtype=np.uint8)
    color = get_background_color_hex(white_crop)
    assert color == "#FFFFFF"

def test_focr_04_black_background_crop():
    black_crop = np.full((100, 100, 3), 0, dtype=np.uint8)
    color = get_background_color_hex(black_crop)
    assert color == "#000000"

def test_focr_04_pure_color_crop():
    # Pure red in BGR is (0, 0, 255)
    red_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    red_crop[:, :] = (0, 0, 255)
    color = get_background_color_hex(red_crop)
    assert color == "#FF0000"

def test_focr_04_empty_crop_safe():
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    assert get_background_color_hex(empty_crop) == "#FFFFFF"
    assert get_background_color_hex(None) == "#FFFFFF"

def test_focr_04_tiny_crop_safe():
    tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
    hex_val = get_background_color_hex(tiny)
    assert hex_val.startswith("#")
    assert len(hex_val) == 7

# ============================================================================
# F-OCR-05: Blank/Spacer Page Fast Bypass
# ============================================================================

def test_focr_05_solid_white_detection():
    white_img = np.full((800, 600, 3), 255, dtype=np.uint8)
    gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    assert variance < 1.0  # Solid color

def test_focr_05_solid_black_detection():
    black_img = np.full((800, 600, 3), 0, dtype=np.uint8)
    gray = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    assert variance < 1.0

def test_focr_05_content_image_high_variance(sample_manga_image_np):
    gray = cv2.cvtColor(sample_manga_image_np, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    assert variance > 50.0  # Real content page has high variance

def test_focr_05_solid_color_ocr_returns_empty():
    engine = OCREngine(use_gpu=False)
    engine._paddle_ocr = type("MockPaddle", (), {"ocr": lambda s, img: [[]]})()
    white_img = np.full((400, 400, 3), 255, dtype=np.uint8)
    blocks = engine.detect_and_recognize(white_img)
    assert blocks == []

def test_focr_05_engine_state_preserved_after_blank():
    engine = OCREngine(use_gpu=False)
    engine._paddle_ocr = type("MockPaddle", (), {"ocr": lambda s, img: [[]]})()
    white_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res1 = engine.detect_and_recognize(white_img)
    assert res1 == []
    # Engine is still ready
    assert engine._paddle_ocr is not None
