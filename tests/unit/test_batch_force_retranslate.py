"""
tests/unit/test_batch_force_retranslate.py
Unit tests verifying the batch full re-translation feature:
- BatchWorker with force_retranslate=True bypasses cache skipping and re-executes translation & rendering
- PageListWidget exposes both normal batch translate and force re-translate all buttons
- MainWindow toolbar exposes normal batch translate and force re-translate all buttons
"""
import os
import tempfile
import shutil
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.core.models import TranslationBlock
from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imwrite
from app.core.pipeline.batch_worker import BatchWorker
from app.ui.sidebar.page_list import PageListWidget
from desktop.ui.queue_panel import QueuePanel


@pytest.fixture
def temp_manga_env():
    tmp_dir = tempfile.mkdtemp(prefix="amt_force_test_")
    page_paths = []
    for i in range(1, 3):
        p = os.path.join(tmp_dir, f"test_page_{i:02d}.png")
        img = np.ones((60, 60, 3), dtype=np.uint8) * 200
        safe_cv2_imwrite(p, img, ext=".png")
        page_paths.append(p)

    export_dir = os.path.join(tmp_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    yield tmp_dir, page_paths, export_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


@patch("app.core.pipeline.batch_worker.OCREngine")
@patch("app.core.pipeline.batch_worker.InpaintEngine")
@patch("app.core.pipeline.batch_worker.TranslationManager")
@patch("app.core.pipeline.batch_worker.TypographyEngine")
def test_batch_worker_force_retranslate_bypasses_cache(
    mock_typo_cls, mock_trans_cls, mock_inpaint_cls, mock_ocr_cls, temp_manga_env
):
    tmp_dir, page_paths, export_dir = temp_manga_env

    mock_ocr = MagicMock()
    mock_ocr.detect_and_recognize.return_value = [
        {"id": "b1", "original_text": "オハヨウ", "translated_text": "", "xmin": 5, "ymin": 5, "xmax": 25, "ymax": 25}
    ]
    mock_ocr_cls.return_value = mock_ocr

    mock_inpaint = MagicMock()
    mock_inpaint.inpaint.return_value = np.zeros((60, 60, 3), dtype=np.uint8)
    mock_inpaint_cls.return_value = mock_inpaint

    # Translation counter
    call_records = []
    mock_trans_mgr = MagicMock()
    def mock_translate(blocks, **kwargs):
        call_records.append(len(blocks))
        tb = TranslationBlock(id="b1", original_text="オハヨウ", translated_text=f"早安_{len(call_records)}", xmin=5, ymin=5, xmax=25, ymax=25)
        return [tb]
    mock_trans_mgr.translate.side_effect = mock_translate
    mock_trans_cls.get_instance.return_value = mock_trans_mgr

    mock_typo = MagicMock()
    mock_typo.render_translations.return_value = np.ones((60, 60, 3), dtype=np.uint8) * 128
    mock_typo_cls.return_value = mock_typo

    items = [{"id": f"it_{i}", "path": p} for i, p in enumerate(page_paths)]
    cfg = {"ocr_engine": "easyocr", "provider": "deepseek"}

    # 1. First run: normal batch translation
    w1 = BatchWorker(items, cfg, export_dir=export_dir, force_retranslate=False)
    w1.run()
    assert len(call_records) == 2

    # 2. Second run with force_retranslate=False: should hit cache and skip
    call_records.clear()
    w2 = BatchWorker(items, cfg, export_dir=export_dir, force_retranslate=False)
    w2.run()
    assert len(call_records) == 0, "Normal batch must skip already translated items"

    # 3. Third run with force_retranslate=True: must NOT skip, must re-translate all pages!
    call_records.clear()
    w3 = BatchWorker(items, cfg, export_dir=export_dir, force_retranslate=True)
    w3.run()
    assert len(call_records) == 2, "force_retranslate=True must re-translate all items regardless of cache"


def test_page_list_widget_has_both_batch_buttons(qapp):
    panel = PageListWidget()
    assert hasattr(panel, "batch_btn")
    assert hasattr(panel, "retranslate_all_btn")
    assert "跳过" in panel.batch_btn.text() or "批量翻译" in panel.batch_btn.text()
    assert "重新翻译" in panel.retranslate_all_btn.text()

    # Enable buttons for click test
    panel.batch_btn.setEnabled(True)
    panel.retranslate_all_btn.setEnabled(True)

    # Signals
    emitted = []
    panel.sig_start_batch.connect(lambda: emitted.append("batch"))
    panel.sig_start_retranslate_all.connect(lambda: emitted.append("retranslate"))

    panel.batch_btn.click()
    assert emitted == ["batch"]

    panel.retranslate_all_btn.click()
    assert emitted == ["batch", "retranslate"]


def test_desktop_queue_panel_has_both_batch_buttons(qapp):
    panel = QueuePanel()
    assert hasattr(panel, "batch_btn")
    assert hasattr(panel, "retranslate_all_btn")

    emitted = []
    panel.items_data = [{"id": "1", "path": "dummy.png"}]
    panel.sig_start_batch.connect(lambda it: emitted.append("batch"))
    panel.sig_start_retranslate_all.connect(lambda it: emitted.append("retranslate"))

    panel.batch_btn.click()
    assert emitted == ["batch"]

    panel.retranslate_all_btn.click()
    assert emitted == ["batch", "retranslate"]


def test_clear_caches_for_items_deletes_cache_directories(temp_manga_env):
    """
    Verifies that cache_mgr.clear_caches_for_items() removes .amt_cache folders on disk.
    """
    tmp_dir, page_paths, _ = temp_manga_env
    cache_mgr = get_cache_manager()

    # Create dummy cache files for the pages
    for p in page_paths:
        cache_mgr.save_page_cache(
            p,
            blocks=[{"id": "b1", "original_text": "text"}],
            erased_img=np.zeros((10, 10, 3), dtype=np.uint8),
            rendered_img=np.ones((10, 10, 3), dtype=np.uint8)
        )
        paths = cache_mgr.get_cache_paths(p)
        assert os.path.exists(paths["dir"])
        assert os.path.exists(paths["blocks_json"])

    # Clear cache for all items
    items = [{"id": str(i), "path": p} for i, p in enumerate(page_paths)]
    cache_mgr.clear_caches_for_items(items)

    # Verify that .amt_cache directories are completely gone
    for p in page_paths:
        paths = cache_mgr.get_cache_paths(p)
        assert not os.path.exists(paths["dir"])


def test_mainwindow_start_batch_force_retranslate_clears_cache_and_state(qapp, temp_manga_env):
    """
    Verifies that MainWindow._start_batch(force_retranslate=True) clears disk cache folders
    and resets in-memory item blocks, erased, and rendered states.
    """
    from app.ui.main_window import MainWindow
    tmp_dir, page_paths, _ = temp_manga_env
    cache_mgr = get_cache_manager()

    # Pre-populate cache
    for p in page_paths:
        cache_mgr.save_page_cache(
            p,
            blocks=[{"id": "b1", "original_text": "text"}],
            erased_img=np.zeros((10, 10, 3), dtype=np.uint8),
            rendered_img=np.ones((10, 10, 3), dtype=np.uint8)
        )

    win = MainWindow()
    items = [
        {
            "id": str(i),
            "path": p,
            "filename": os.path.basename(p),
            "blocks": [{"id": "b1"}],
            "erased_img": np.zeros((10, 10, 3), dtype=np.uint8),
            "translated_img": np.ones((10, 10, 3), dtype=np.uint8)
        }
        for i, p in enumerate(page_paths)
    ]
    win.page_list.items_data = items
    win.current_image_data = items[0]

    # Trigger force retranslate batch
    win._start_batch(force_retranslate=True)

    # All in-memory states must be cleared
    for it in items:
        assert it["blocks"] == []
        assert it["erased_img"] is None
        assert it["translated_img"] is None

    # Disk cache folder must have been removed
    for p in page_paths:
        paths = cache_mgr.get_cache_paths(p)
        assert not os.path.exists(paths["dir"])

    if win.active_batch_worker:
        win.active_batch_worker.cancel()
        win.active_batch_worker.wait(1000)
    win.close()

