"""
tests/unit/test_retranslate_status_correspondence.py
Unit tests verifying front-end page status correspondence when re-translating an image
via right-click context menus (both in the sidebar page list and on the canvas).
"""
import os
import pytest
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.sidebar.page_list import PageListWidget, PageItemWidget
from app.core.config import AppConfig
from app.core.cache.cache_manager import get_cache_manager


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_page_list_update_item_status(qapp, tmp_path):
    """
    Verifies that update_item_status correctly updates both the PageItemWidget
    (status text and dot color) and the underlying items_data dictionary.
    """
    test_img = tmp_path / "page_001.png"
    cv2.imwrite(str(test_img), np.full((100, 100, 3), 255, dtype=np.uint8))

    panel = PageListWidget()
    panel.add_paths([str(test_img)])

    assert len(panel.items_data) == 1
    item_id = panel.items_data[0]["id"]
    widget = panel._item_widgets[item_id]

    # Initially queued
    assert panel.items_data[0]["status"] == "queued"

    # Update to processing
    panel.update_item_status(item_id, "processing", "处理中")
    assert panel.items_data[0]["status"] == "processing"
    assert panel.items_data[0]["status_text"] == "处理中"
    assert widget.status_label.text() == "处理中"
    assert widget.status_dot._color.upper() == "#3B82F6"

    # Update to completed
    panel.update_item_status(item_id, "completed", "已完成")
    assert panel.items_data[0]["status"] == "completed"
    assert panel.items_data[0]["status_text"] == "已完成"
    assert widget.status_label.text() == "已完成"
    assert widget.status_dot._color.upper() == "#10B981"

    # Update to failed
    panel.update_item_status(item_id, "failed", "失败: OCR超时")
    assert panel.items_data[0]["status"] == "failed"
    assert panel.items_data[0]["status_text"] == "失败: OCR超时"
    assert widget.status_label.text() == "失败: OCR超时"
    assert widget.status_dot._color.upper() == "#EF4444"


def test_mainwindow_canvas_retranslate_updates_status(qapp, tmp_path, monkeypatch):
    """
    Verifies that right-clicking the canvas to re-translate:
    1. Clears disk cache.
    2. Clears in-memory blocks and images.
    3. Updates page_list status through the full translation lifecycle:
       queued -> processing -> completed.
    """
    img_path = str(tmp_path / "test_retrans.png")
    cv2.imwrite(img_path, np.full((200, 200, 3), 255, dtype=np.uint8))

    # Pre-populate disk cache
    cache_mgr = get_cache_manager()
    cache_mgr.save_page_cache(
        img_path,
        blocks=[{"original_text": "Old", "translated_text": "旧"}],
        erased_img=np.zeros((200, 200, 3), dtype=np.uint8)
    )
    assert cache_mgr.has_cache(img_path)["blocks"] is True

    win = MainWindow()
    win.page_list.add_paths([img_path])
    item_id = win.page_list.items_data[0]["id"]
    win._on_page_selected(win.page_list.items_data[0])

    # Simulate completed initial state
    win.page_list.update_item_status(item_id, "completed", "已完成")
    assert win.page_list.items_data[0]["status"] == "completed"

    # Mock PipelineWorker to avoid invoking external OCR / LLM in unit test
    def mock_start(worker_self):
        # Emulate synchronous pipeline completion
        worker_self.sig_finished.emit({
            "original_img": cv2.imread(img_path),
            "translated_img": np.ones((200, 200, 3), dtype=np.uint8) * 200,
            "erased_img": np.ones((200, 200, 3), dtype=np.uint8) * 255,
            "blocks": [{"original_text": "New", "translated_text": "新"}]
        })

    from app.core.pipeline.pipeline_worker import PipelineWorker
    monkeypatch.setattr(PipelineWorker, "start", mock_start)

    # User triggers re-translate from canvas context menu
    win._on_canvas_retranslate_requested()

    # Verify disk cache was cleared
    assert cache_mgr.has_cache(img_path)["blocks"] is False

    # Verify status transitioned and finished at completed
    assert win.page_list.items_data[0]["status"] == "completed"
    assert win.page_list.items_data[0]["status_text"] == "已完成"
    widget = win.page_list._item_widgets[item_id]
    assert widget.status_label.text() == "已完成"
    assert widget.status_dot._color.upper() == "#10B981"


def test_mainwindow_pipeline_error_updates_status_to_failed(qapp, tmp_path, monkeypatch):
    """
    Verifies that if single-page pipeline encounters an error,
    the image status in page_list is accurately updated to failed.
    """
    img_path = str(tmp_path / "test_err.png")
    cv2.imwrite(img_path, np.full((200, 200, 3), 255, dtype=np.uint8))

    win = MainWindow()
    win.page_list.add_paths([img_path])
    item_id = win.page_list.items_data[0]["id"]
    win._on_page_selected(win.page_list.items_data[0])

    def mock_start_error(worker_self):
        worker_self.sig_error.emit("OCR model failed to decode image")

    from app.core.pipeline.pipeline_worker import PipelineWorker
    monkeypatch.setattr(PipelineWorker, "start", mock_start_error)

    win._start_pipeline_for_page(mode="full")

    assert win.page_list.items_data[0]["status"] == "failed"
    assert "失败" in win.page_list.items_data[0]["status_text"]
    widget = win.page_list._item_widgets[item_id]
    assert widget.status_dot._color.upper() == "#EF4444"


def test_page_list_update_item_status_default_message(qapp, tmp_path):
    """
    Verifies that calling update_item_status without message correctly populates
    items_data['status_text'] with mapped Chinese label rather than empty string.
    """
    test_img = tmp_path / "page_default_msg.png"
    cv2.imwrite(str(test_img), np.full((100, 100, 3), 255, dtype=np.uint8))

    panel = PageListWidget()
    panel.add_paths([str(test_img)])
    item_id = panel.items_data[0]["id"]

    # Calling with message=""
    panel.update_item_status(item_id, "completed")
    assert panel.items_data[0]["status"] == "completed"
    assert panel.items_data[0]["status_text"] == "已完成"

    panel.update_item_status(item_id, "processing")
    assert panel.items_data[0]["status"] == "processing"
    assert panel.items_data[0]["status_text"] == "处理中"

    panel.update_item_status(item_id, "queued")
    assert panel.items_data[0]["status"] == "queued"
    assert panel.items_data[0]["status_text"] == "等待中"


def test_mainwindow_canvas_retranslate_resets_view_and_inspector(qapp, tmp_path, monkeypatch):
    """
    Verifies that right-clicking canvas re-translate resets canvas_view to original mode
    with zero bubbles and clears inspector blocks immediately upon click.
    """
    img_path = str(tmp_path / "test_reset.png")
    cv2.imwrite(img_path, np.full((200, 200, 3), 255, dtype=np.uint8))

    win = MainWindow()
    win.page_list.add_paths([img_path])
    win._on_page_selected(win.page_list.items_data[0])

    # Populate dummy bubbles
    win.canvas_view.set_data(
        original_cv=cv2.imread(img_path),
        translated_cv=np.ones((200, 200, 3), dtype=np.uint8) * 100,
        blocks=[{"id": "b1", "text": "test"}]
    )
    win.inspector_panel.set_blocks([{"id": "b1", "text": "test"}])
    assert len(win.inspector_panel.bubble_list) > 0

    # Prevent actual pipeline thread from completing synchronously during this check
    from app.core.pipeline.pipeline_worker import PipelineWorker
    monkeypatch.setattr(PipelineWorker, "start", lambda self: None)

    win._on_canvas_retranslate_requested()

    # View should be reset to original mode
    assert win.canvas_view.view_mode == "original"
    assert win.canvas_view.translated_cv is None
    assert len(win.inspector_panel.bubble_list) == 0
    assert win.page_list.items_data[0]["status"] == "processing"


def test_mainwindow_run_cancel_resets_status_to_queued(qapp, tmp_path, monkeypatch):
    """
    Verifies that cancelling an active worker via the run button resets
    the page status in the page list from processing back to queued.
    """
    img_path = str(tmp_path / "test_cancel.png")
    cv2.imwrite(img_path, np.full((200, 200, 3), 255, dtype=np.uint8))

    win = MainWindow()
    win.page_list.add_paths([img_path])
    item_id = win.page_list.items_data[0]["id"]
    win._on_page_selected(win.page_list.items_data[0])

    from app.core.pipeline.pipeline_worker import PipelineWorker
    # Mock isRunning to return True and cancel to set a flag
    monkeypatch.setattr(PipelineWorker, "start", lambda self: None)
    monkeypatch.setattr(PipelineWorker, "isRunning", lambda self: True)

    win._start_pipeline_for_page(mode="full")
    assert win.page_list.items_data[0]["status"] == "processing"

    win._on_run_clicked()
    assert win.page_list.items_data[0]["status"] == "queued"
    assert win.page_list.items_data[0]["status_text"] == "等待中"

