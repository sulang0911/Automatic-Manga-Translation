"""
tests/unit/test_manual_box_ocr.py
Unit tests verifying the manual box-selection OCR recognition & translation feature:
1. Canvas HUD has "🔍 框选识别 (O)" tool button and signals.
2. MangaCanvasView handles 'draw_ocr' mode, shortcut O, and emits sig_bubble_ocr_requested.
3. InspectorPanel has 'ocr_translate_block_btn' and emits sig_ocr_translate_block_requested.
4. BlockOcrTranslateWorker runs OCR, translation, inpaint, and typography on the selected patch.
"""
import numpy as np
import pytest
from PyQt6.QtCore import Qt, QPointF, QPoint
from PyQt6.QtGui import QKeyEvent, QMouseEvent

from app.ui.canvas.view import MangaCanvasView, CanvasZoomHud
from app.ui.inspector.inspector_panel import InspectorPanel
from app.core.pipeline.block_worker import BlockOcrTranslateWorker


def test_canvas_hud_ocr_draw_button(qapp):
    hud = CanvasZoomHud()
    assert hasattr(hud, "btn_ocr_draw")
    assert hasattr(hud, "btn_draw")
    assert "O" in hud.btn_ocr_draw.text()
    assert "R" in hud.btn_draw.text()

    toggled_states = []
    hud.sig_tool_ocr_draw_toggled.connect(lambda state: toggled_states.append(state))

    hud.btn_ocr_draw.setChecked(True)
    assert len(toggled_states) == 1
    assert toggled_states[0] is True


def test_canvas_view_ocr_tool_mode_and_shortcut(qapp):
    view = MangaCanvasView()
    assert hasattr(view, "sig_bubble_ocr_requested")
    assert view.tool_mode == "select"

    # Test toggle_ocr_draw_tool
    view.toggle_ocr_draw_tool()
    assert view.tool_mode == "draw_ocr"
    assert view.hud.btn_ocr_draw.isChecked()

    # Toggle back to select
    view.toggle_ocr_draw_tool()
    assert view.tool_mode == "select"
    assert not view.hud.btn_ocr_draw.isChecked()

    # Test key shortcut O
    event_o = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_O, Qt.KeyboardModifier.NoModifier)
    view.keyPressEvent(event_o)
    assert view.tool_mode == "draw_ocr"

    # Test Escape to reset
    event_esc = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    view.keyPressEvent(event_esc)
    assert view.tool_mode == "select"


def test_canvas_view_ocr_drawing_emits_request(qapp):
    view = MangaCanvasView()
    # Provide a dummy 1000x1000 image
    dummy_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    view.set_data(original_cv=dummy_img)

    view.set_tool_mode("draw_ocr")
    assert view.tool_mode == "draw_ocr"

    ocr_requests = []
    view.sig_bubble_ocr_requested.connect(lambda b: ocr_requests.append(b))

    # Simulate mouse drag
    press_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(100, 100),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    view.mousePressEvent(press_event)
    assert view._is_drawing_rect is True

    move_event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(300, 250),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    view.mouseMoveEvent(move_event)

    release_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(300, 250),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier
    )
    view.mouseReleaseEvent(release_event)

    assert len(ocr_requests) == 1
    req = ocr_requests[0]
    assert "id" in req
    assert req["xmin"] >= 0
    assert req["ymin"] >= 0
    assert req["xmax"] > req["xmin"]
    assert req["ymax"] > req["ymin"]
    assert view.tool_mode == "select"


def test_inspector_panel_ocr_translate_button(qapp):
    panel = InspectorPanel()
    assert hasattr(panel, "ocr_translate_block_btn")
    assert hasattr(panel, "sig_ocr_translate_block_requested")

    emitted = []
    panel.sig_ocr_translate_block_requested.connect(lambda b: emitted.append(b))

    block = {
        "id": "test_block_1",
        "xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 50.0,
        "original_text": "Hello", "translated_text": "你好"
    }
    panel.set_blocks([block])
    panel.select_block_by_id("test_block_1")

    panel.ocr_translate_block_btn.click()
    assert len(emitted) == 1
    assert emitted[0]["id"] == "test_block_1"


def test_block_ocr_translate_worker_execution(qapp, monkeypatch):
    # Mock TranslationManager and OCR for fast, isolated unit test
    class MockTranslationManager:
        @classmethod
        def get_instance(cls, *args, **kwargs):
            return MockTranslationManager()
        def set_active_provider(self, *args, **kwargs):
            pass
        def translate(self, blocks, **kwargs):
            for b in blocks:
                b["translated_text"] = "【测试译文】"
            return blocks

    class MockOCREngine:
        def __init__(self, *args, **kwargs):
            pass
        def detect_and_recognize(self, crop, **kwargs):
            return [{"original_text": "Sample Manga Text", "xmin": 0, "ymin": 0, "xmax": 50, "ymax": 20}]

    monkeypatch.setattr("app.core.pipeline.block_worker.TranslationManager", MockTranslationManager)
    monkeypatch.setattr("app.core.pipeline.block_worker.OCREngine", MockOCREngine)

    dummy_img = np.full((800, 800, 3), 255, dtype=np.uint8)
    target_block = {
        "id": "target_1",
        "xmin": 20.0, "ymin": 20.0, "xmax": 60.0, "ymax": 40.0,
        "original_text": "",
        "translated_text": "",
        "type": "bubble"
    }
    all_blocks = [target_block]

    worker = BlockOcrTranslateWorker(
        image_path="",
        original_cv=dummy_img,
        target_block=target_block,
        all_blocks=all_blocks,
        existing_erased=None,
        config={}
    )

    completed_payload = []
    worker.sig_completed.connect(lambda res: completed_payload.append(res))

    worker.run()

    assert len(completed_payload) == 1
    res = completed_payload[0]
    assert res["target_block"]["original_text"] == "Sample Manga Text"
    assert res["target_block"]["translated_text"] == "【测试译文】"
    assert res["erased_img"] is not None
    assert res["translated_img"] is not None


def test_block_ocr_reuses_prior_fullpage_cache_when_reselecting(qapp, monkeypatch):
    """
    Verifies that when a user deletes a problematic bubble and manually box-selects it,
    if the page cache had prior full-page high-resolution OCR detections in that area,
    it reuses the high-res text cleanly.
    """
    class MockCacheManager:
        def has_cache(self, path):
            return {"blocks": True, "erased": False, "rendered": False}
        def load_page_cache(self, path, load_images=False):
            return {
                "blocks": [
                    {
                        "id": "prior_1",
                        "xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 30.0,
                        "original_text": "Prior Full Page High Res Text",
                        "translated_text": "旧译文"
                    }
                ]
            }
        def save_page_cache(self, *args, **kwargs):
            pass

    monkeypatch.setattr("app.core.pipeline.block_worker.get_cache_manager", lambda: MockCacheManager())

    class MockTranslationManager:
        @classmethod
        def get_instance(cls, *args, **kwargs):
            return MockTranslationManager()
        def set_active_provider(self, *args, **kwargs):
            pass
        def translate(self, blocks, **kwargs):
            for b in blocks:
                b["translated_text"] = "重新翻译成功"
            return blocks

    monkeypatch.setattr("app.core.pipeline.block_worker.TranslationManager", MockTranslationManager)

    dummy_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    target_block = {
        "id": "new_manual_1",
        "xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 30.0,
        "original_text": "",
        "translated_text": "",
        "type": "bubble"
    }

    worker = BlockOcrTranslateWorker(
        image_path="dummy.jpg",
        original_cv=dummy_img,
        target_block=target_block,
        all_blocks=[target_block],
        existing_erased=None,
        config={}
    )

    completed = []
    worker.sig_completed.connect(lambda res: completed.append(res))
    worker.run()

    assert len(completed) == 1
    assert completed[0]["target_block"]["original_text"] == "Prior Full Page High Res Text"
    assert completed[0]["target_block"]["translated_text"] == "重新翻译成功"

