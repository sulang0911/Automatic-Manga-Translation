"""
tests/unit/test_batch_cache_streaming.py
Unit tests verifying BatchWorker disk cache generation, low-memory streaming,
and instant breakpoint resumption.
"""
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imwrite
from app.core.pipeline.batch_worker import BatchWorker
from app.core.models import TranslationBlock


@pytest.fixture
def temp_batch_env():
    tmp_dir = tempfile.mkdtemp(prefix="amt_batch_test_")
    # Generate 3 dummy manga page images
    page_paths = []
    for i in range(1, 4):
        p = os.path.join(tmp_dir, f"page_{i:03d}.png")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
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
def test_batch_worker_cache_creation_and_memory_isolation(
    mock_typo_cls, mock_trans_cls, mock_inpaint_cls, mock_ocr_cls, temp_batch_env
):
    tmp_dir, page_paths, export_dir = temp_batch_env

    # Setup mocks
    mock_ocr = MagicMock()
    mock_ocr.detect_and_recognize.return_value = [
        {"id": "b1", "original_text": "hello", "translated_text": "", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30}
    ]
    mock_ocr_cls.return_value = mock_ocr

    mock_inpaint = MagicMock()
    mock_inpaint.inpaint.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_inpaint_cls.return_value = mock_inpaint

    mock_trans_mgr = MagicMock()
    def mock_translate(blocks, **kwargs):
        tb = TranslationBlock(id="b1", original_text="hello", translated_text="你好", xmin=10, ymin=10, xmax=30, ymax=30)
        return [tb]
    mock_trans_mgr.translate.side_effect = mock_translate
    mock_trans_cls.get_instance.return_value = mock_trans_mgr

    mock_typo = MagicMock()
    mock_typo.render_translations.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 200
    mock_typo_cls.return_value = mock_typo

    queue_items = [{"id": f"id_{i}", "path": p} for i, p in enumerate(page_paths)]
    cfg = {"ocr_engine": "easyocr", "provider": "deepseek", "model": "deepseek-chat"}

    worker = BatchWorker(queue_items, cfg, export_dir=export_dir)

    completed_payloads = []
    worker.sig_item_completed.connect(lambda iid, res: completed_payloads.append((iid, res)))

    # Run batch
    worker.run()

    cache_mgr = get_cache_manager()

    # 1. Verify all 3 pages generated disk cache in .amt_cache
    for p in page_paths:
        has = cache_mgr.has_cache(p)
        assert has["erased"] is True
        assert has["blocks"] is True
        assert has["rendered"] is True
        assert cache_mgr.is_fully_translated(p) is True

    # 2. Verify payload does NOT pin heavy numpy arrays in RAM
    assert len(completed_payloads) == 3
    for iid, res in completed_payloads:
        assert res.get("has_cache") is True
        assert "original_img" not in res
        assert "erased_img" not in res
        assert "translated_img" not in res

    # 3. Verify breakpoint resumption on second run
    mock_ocr.detect_and_recognize.reset_mock()
    mock_inpaint.inpaint.reset_mock()
    mock_trans_mgr.translate.reset_mock()

    worker2 = BatchWorker(queue_items, cfg, export_dir=export_dir)
    worker2.run()

    # Neither OCR, nor Inpaint, nor LLM should have been called! Everything was loaded from cache!
    assert mock_ocr.detect_and_recognize.call_count == 0
    assert mock_inpaint.inpaint.call_count == 0
    assert mock_trans_mgr.translate.call_count == 0

    # 4. Verify force_retranslate=True re-translates all pages regardless of existing cache
    mock_trans_mgr.translate.reset_mock()
    mock_typo.render_translations.reset_mock()

    worker3 = BatchWorker(queue_items, cfg, export_dir=export_dir, force_retranslate=True)
    worker3.run()

    # LLM translate and typography render must be re-executed for all 3 pages
    assert mock_trans_mgr.translate.call_count == 3
    assert mock_typo.render_translations.call_count == 3