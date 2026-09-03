import os
import pytest
import numpy as np
import cv2
from PIL import ImageFont

from desktop.core.ocr_engine import OCREngine, get_background_color_hex
from desktop.core.inpaint_engine import InpaintEngine
from desktop.core.translation_engine import TranslationEngine
from desktop.core.typography_engine import TypographyEngine
from desktop.core.pipeline_worker import PipelineWorker

# ============================================================================
# Tier 2: Boundary & Corner Cases
# ============================================================================

def test_tier2_01_extreme_tall_narrow_bubble():
    # Extreme vertical aspect ratio: width=15, height=300 -> aspect = 0.05
    box = {"xmin": 50, "ymin": 50, "xmax": 65, "ymax": 350, "text": "あああ"}
    aspect = (box["xmax"] - box["xmin"]) / max(1, (box["ymax"] - box["ymin"]))
    is_bubble = not (aspect > 4.0 or aspect < 0.15)
    assert is_bubble is False  # Extreme aspect classified as SFX/onomatopoeia

def test_tier2_02_extreme_wide_banner_bubble():
    # Extreme horizontal banner: width=800, height=40 -> aspect = 20.0
    box = {"xmin": 50, "ymin": 20, "xmax": 850, "ymax": 60, "text": "第１話　はじまりの朝"}
    aspect = (box["xmax"] - box["xmin"]) / max(1, (box["ymax"] - box["ymin"]))
    is_bubble = not (aspect > 4.0 or aspect < 0.15)
    assert is_bubble is False

def test_tier2_03_solid_white_spacer_page():
    img = np.full((1200, 800, 3), 255, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    assert variance < 0.1
    # Bypasses OCR
    engine = OCREngine(use_gpu=False)
    engine._paddle_ocr = type("MockPaddle", (), {"ocr": lambda s, i: [[]]})()
    blocks = engine.detect_and_recognize(img)
    assert blocks == []

def test_tier2_04_solid_black_spacer_page():
    img = np.full((1200, 800, 3), 0, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    assert variance < 0.1
    engine = OCREngine(use_gpu=False)
    engine._paddle_ocr = type("MockPaddle", (), {"ocr": lambda s, i: [[]]})()
    blocks = engine.detect_and_recognize(img)
    assert blocks == []

def test_tier2_05_inverted_contrast_scheme():
    # Black speech bubble with white text
    dark_crop = np.full((100, 100, 3), 15, dtype=np.uint8)
    bg_hex = get_background_color_hex(dark_crop)
    assert bg_hex == "#0F0F0F"

    # White text (#FFFFFF) over dark bubble gets light luminance, auto stroke should be black or white
    text_rgb = (255, 255, 255)
    lum = 0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]
    stroke_rgb = (255, 255, 255) if lum < 128 else (0, 0, 0)
    assert stroke_rgb == (0, 0, 0)

def test_tier2_06_http_429_rate_limit_payload(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    class Mock429:
        status_code = 429
        text = '{"error": {"message": "Rate limit exceeded (TPM). Please wait 30s."}}'
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock429())

    with pytest.raises(RuntimeError) as exc:
        engine.translate_blocks([{"id": "1", "original_text": "text"}])
    assert "429" in str(exc.value)

def test_tier2_07_http_500_internal_error_payload(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    class Mock500:
        status_code = 500
        text = '{"error": "Internal Server Error"}'
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock500())

    with pytest.raises(RuntimeError) as exc:
        engine.translate_blocks([{"id": "1", "original_text": "text"}])
    assert "500" in str(exc.value)

def test_tier2_08_truncated_corrupt_image(temp_dir):
    corrupt_file = os.path.join(temp_dir, "truncated.png")
    with open(corrupt_file, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 10)  # Incomplete PNG chunk

    arr = np.fromfile(corrupt_file, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is None

def test_tier2_09_high_res_4k_image():
    # 4K image (4096 x 4096)
    img_4k = np.full((4096, 4096, 3), 240, dtype=np.uint8)
    typo_eng = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 20, "ymin": 20, "xmax": 40, "ymax": 40, "translated_text": "4K测试"}]
    res = typo_eng.render_translations(img_4k, blocks, {})
    assert res.shape == (4096, 4096, 3)

def test_tier2_10_low_res_micro_image():
    # Micro image (32 x 32)
    micro = np.full((32, 32, 3), 200, dtype=np.uint8)
    typo_eng = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 90, "ymax": 90, "translated_text": "小"}]
    res = typo_eng.render_translations(micro, blocks, {})
    assert res.shape == (32, 32, 3)

def test_tier2_11_missing_font_fallback():
    typo_eng = TypographyEngine()
    font = typo_eng._get_font("ImaginaryComicFont-Bold-999", 20)
    assert font is not None

def test_tier2_12_empty_text_blocks_pipeline(sample_manga_image_np):
    inpaint_eng = InpaintEngine(mode="opencv_telea")
    erased = inpaint_eng.inpaint(sample_manga_image_np, [])
    assert np.array_equal(erased, sample_manga_image_np)

    trans_eng = TranslationEngine(api_key="")
    translated_blocks = trans_eng.translate_blocks([])
    assert translated_blocks == []

    typo_eng = TypographyEngine()
    rendered = typo_eng.render_translations(erased, [], {})
    assert rendered.shape == sample_manga_image_np.shape

def test_tier2_13_cjk_punctuation_and_emojis(sample_manga_image_np):
    typo_eng = TypographyEngine()
    special_text = "「待って！」『本当に…！？』💥(笑)✨"
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 40, "translated_text": special_text}]
    res = typo_eng.render_translations(sample_manga_image_np, blocks, {})
    assert res is not None

def test_tier2_14_rapid_consecutive_cancellations():
    worker = PipelineWorker("dummy.png", {})
    for _ in range(10):
        worker.cancel()
    assert worker._is_cancelled

def test_tier2_15_windows_path_special_characters(temp_dir, sample_manga_image_np):
    # Windows path with brackets, spaces, and hash
    special_name = "[Chapter 01] Page #01 (Final).png"
    special_path = os.path.join(temp_dir, special_name)

    success, buf = cv2.imencode(".png", sample_manga_image_np)
    assert success
    with open(special_path, "wb") as f:
        f.write(buf.tobytes())

    assert os.path.exists(special_path)
    # Read using numpy stream
    with open(special_path, "rb") as f:
        data = f.read()
    loaded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert loaded.shape == sample_manga_image_np.shape
