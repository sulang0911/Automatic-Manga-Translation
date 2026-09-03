import os
import pytest
import numpy as np
import cv2
from PIL import ImageFont

from desktop.ui.toast import Toast
from desktop.ui.main_window import MainWindow
from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine
from desktop.core.translation_engine import TranslationEngine
from desktop.core.typography_engine import TypographyEngine
from desktop.core.pipeline_worker import PipelineWorker
from desktop.core.batch_worker import BatchWorker

# ============================================================================
# F-ERR-01: Non-Blocking Floating Toast Banners
# ============================================================================

def test_ferr_01_toast_initialization(qapp):
    win = MainWindow()
    toast = Toast(win)
    assert toast.parent() is win
    assert toast.opacity_effect is not None
    assert toast.hide_timer is not None
    win.close()

def test_ferr_01_toast_message_types(qapp):
    win = MainWindow()
    toast = Toast(win)

    # Info
    toast.show_message("信息提示", "info")
    assert toast.icon_label.text() == "ℹ"
    assert toast.msg_label.text() == "信息提示"

    # Success
    toast.show_message("成功提示", "success")
    assert toast.icon_label.text() == "✓"

    # Warning
    toast.show_message("警告提示", "warning")
    assert toast.icon_label.text() == "⚠"

    # Error
    toast.show_message("错误提示", "error")
    assert toast.icon_label.text() == "✕"
    win.close()

def test_ferr_01_toast_positioning(qapp):
    win = MainWindow()
    win.resize(1000, 700)
    toast = Toast(win)
    toast.show_message("定位测试", "info")
    # Position should be centered horizontally near bottom
    assert toast.pos().y() > 0
    win.close()

def test_ferr_01_toast_timer_configured(qapp):
    win = MainWindow()
    toast = Toast(win)
    toast.show_message("定时器测试", "info", duration_ms=1500)
    assert toast.hide_timer.isActive()
    win.close()

def test_ferr_01_toast_fade_out(qapp):
    win = MainWindow()
    toast = Toast(win)
    toast.show_message("淡出测试", "info")
    toast._fade_out()
    assert toast.anim_out is not None
    win.close()

# ============================================================================
# F-ERR-02: API Quota & Rate Limit Diagnostics
# ============================================================================

def test_ferr_02_http_429_rate_limit_detection(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    class Mock429:
        status_code = 429
        text = '{"error": {"message": "You exceeded your current quota, please check your plan and billing details."}}'
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock429())

    with pytest.raises(RuntimeError) as exc_info:
        engine.translate_blocks([{"id": "1", "original_text": "text"}])
    assert "429" in str(exc_info.value)
    assert "quota" in str(exc_info.value)

def test_ferr_02_missing_key_demo_fallback():
    engine = TranslationEngine(api_key="")
    res = engine.translate_blocks([{"id": "b1", "original_text": "demo text"}])
    assert "【译】demo text" in res[0]["translated_text"]

def test_ferr_02_error_preserves_pipeline_state(qapp):
    win = MainWindow()
    win.current_image_data = {"id": "item1", "path": "p.png"}
    win._on_worker_error("API Quota Exceeded")
    assert win.run_btn.isEnabled()
    assert win.status_label.text() == "处理出错"
    win.close()

def test_ferr_02_failed_queue_status_update(qapp, sample_manga_image_file):
    win = MainWindow()
    win.queue_panel.add_paths([sample_manga_image_file])
    item_id = win.queue_panel.items_data[0]["id"]
    win.queue_panel.update_item_status(item_id, "failed", "配额超限")
    assert win.queue_panel.items_data[0]["status"] == "failed"
    win.close()

def test_ferr_02_error_text_tagged_in_blocks(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rate limit")))

    blocks = [{"id": "1", "original_text": "Hello", "translated_text": ""}]
    with pytest.raises(RuntimeError):
        engine.translate_blocks(blocks)
    assert "翻译错误" in blocks[0]["translated_text"]

# ============================================================================
# F-ERR-03: Network Fault Tolerance & Retries
# ============================================================================

def test_ferr_03_connection_refused(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    def mock_conn_err(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")
    monkeypatch.setattr(requests, "post", mock_conn_err)

    with pytest.raises(requests.ConnectionError):
        engine.translate_blocks([{"id": "1", "original_text": "text"}])

def test_ferr_03_request_timeout(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    def mock_timeout(*args, **kwargs):
        raise requests.Timeout("Read timed out")
    monkeypatch.setattr(requests, "post", mock_timeout)

    with pytest.raises(requests.Timeout):
        engine.translate_blocks([{"id": "1", "original_text": "text"}])

def test_ferr_03_http_502_bad_gateway(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    class Mock502:
        status_code = 502
        text = "Bad Gateway"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock502())

    with pytest.raises(RuntimeError) as exc_info:
        engine.translate_blocks([{"id": "1", "original_text": "text"}])
    assert "502" in str(exc_info.value)

def test_ferr_03_corrupt_json_handled_by_regex(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    import requests
    class MockResp:
        status_code = 200
        def json(self):
            # Broken JSON that regex fallback can rescue
            return {"choices": [{"message": {"content": 'prefix "id": "1", "translated_text": "成功拯救" suffix'}}]}
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResp())

    blocks = [{"id": "1", "original_text": "test", "translated_text": ""}]
    res = engine.translate_blocks(blocks)
    assert res[0]["translated_text"] == "成功拯救"

def test_ferr_03_pipeline_worker_catches_network_error(sample_manga_image_file, monkeypatch):
    worker = PipelineWorker(sample_manga_image_file, {"api_key": "sk-test"}, mode="translate_only", existing_blocks=[{"id": "1", "original_text": "a"}])
    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("Network Down")))

    errors = []
    worker.sig_error.connect(lambda e: errors.append(e))
    worker.run()
    assert len(errors) == 1
    assert "处理失败" in errors[0]

# ============================================================================
# F-ERR-04: Model & Dependency Safeguards
# ============================================================================

def test_ferr_04_paddle_ocr_cpu_fallback():
    engine = OCREngine(use_gpu=False)
    assert not engine.use_gpu

def test_ferr_04_lama_inpainter_not_crashing_on_absence():
    engine = InpaintEngine(mode="opencv_telea")
    assert engine.mode == "opencv_telea"

def test_ferr_04_bilateral_filter_safety(sample_manga_image_np):
    filtered = cv2.bilateralFilter(sample_manga_image_np, 5, 50, 50)
    assert filtered.shape == sample_manga_image_np.shape

def test_ferr_04_telea_inpaint_fallback(sample_manga_image_np):
    mask = np.zeros(sample_manga_image_np.shape[:2], dtype=np.uint8)
    mask[50:100, 50:100] = 255
    res = cv2.inpaint(sample_manga_image_np, mask, 3, cv2.INPAINT_TELEA)
    assert res.shape == sample_manga_image_np.shape

def test_ferr_04_empty_input_for_ocr():
    engine = OCREngine()
    assert engine.detect_and_recognize(np.zeros((0, 0, 3), dtype=np.uint8)) == []

# ============================================================================
# F-ERR-05: Corrupted Image File Isolation
# ============================================================================

def test_ferr_05_imdecode_corrupt_bytes():
    corrupt_bytes = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    res = cv2.imdecode(corrupt_bytes, cv2.IMREAD_COLOR)
    assert res is None

def test_ferr_05_pipeline_worker_corrupt_file(temp_dir):
    corrupt_path = os.path.join(temp_dir, "corrupt.png")
    with open(corrupt_path, "wb") as f:
        f.write(b"not a valid image content")

    worker = PipelineWorker(corrupt_path, {})
    errors = []
    worker.sig_error.connect(lambda e: errors.append(e))
    worker.run()
    assert len(errors) == 1
    assert "无法解码" in errors[0]

def test_ferr_05_batch_worker_continues_after_corrupted_file(temp_dir, sample_manga_image_file, monkeypatch):
    corrupt_path = os.path.join(temp_dir, "corrupt_item.png")
    with open(corrupt_path, "wb") as f:
        f.write(b"broken bytes")

    monkeypatch.setattr(OCREngine, "detect_and_recognize", lambda *args, **kwargs: [])
    monkeypatch.setattr(InpaintEngine, "inpaint", lambda self, img, blocks, *args, **kwargs: img)

    items = [
        {"id": "bad1", "path": corrupt_path},
        {"id": "good2", "path": sample_manga_image_file}
    ]
    worker = BatchWorker(items, {"api_key": ""})  # demo mode
    failed_items = []
    completed_items = []
    finished_counts = []

    worker.sig_item_failed.connect(lambda i, e: failed_items.append(i))
    worker.sig_item_completed.connect(lambda i, r: completed_items.append(i))
    worker.sig_batch_finished.connect(lambda s, f: finished_counts.append((s, f)))

    worker.run()
    assert "bad1" in failed_items
    assert len(finished_counts) == 1
    assert finished_counts[0][1] >= 1

def test_ferr_05_zero_byte_file(temp_dir):
    zero_path = os.path.join(temp_dir, "zero.png")
    with open(zero_path, "wb") as f:
        pass
    worker = PipelineWorker(zero_path, {})
    errors = []
    worker.sig_error.connect(lambda e: errors.append(e))
    worker.run()
    assert len(errors) == 1

def test_ferr_05_corrupt_file_does_not_crash_main_window(qapp, temp_dir):
    corrupt_path = os.path.join(temp_dir, "corrupt.png")
    with open(corrupt_path, "wb") as f:
        f.write(b"corrupt")

    win = MainWindow()
    # Selecting corrupt file does not raise unhandled exception
    win._on_image_selected({"id": "1", "path": corrupt_path})
    win.close()

# ============================================================================
# F-ERR-06: Font Fallback Hierarchy
# ============================================================================

def test_ferr_06_non_existent_font_fallback():
    engine = TypographyEngine()
    font = engine._get_font("CompletelyFakeFontName9999", 24)
    assert font is not None

def test_ferr_06_windows_fonts_list_populated():
    from desktop.core.typography_engine import WINDOWS_FONTS
    assert len(WINDOWS_FONTS) > 0
    assert "msyh.ttc" in WINDOWS_FONTS

def test_ferr_06_font_load_default_success():
    default_font = ImageFont.load_default()
    assert default_font is not None

def test_ferr_06_font_cache_integrity():
    engine = TypographyEngine()
    f1 = engine._get_font("Segoe UI", 14)
    f2 = engine._get_font("Segoe UI", 14)
    assert f1 is f2

def test_ferr_06_negative_font_size_safe():
    engine = TypographyEngine()
    lines, font, sz = engine._fit_text_to_box("测试", 100, 100, "Arial", font_size_scale=0.01, auto_fit=True, bold=False)
    # Must be clamped to minimum >= 9
    assert sz >= 9
