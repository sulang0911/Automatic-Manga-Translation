"""
tests/unit/test_pipeline_m4.py
Unit tests for Milestone M4: PipelineWorker, BatchWorker, MangaExporter, SettingsDialog, and Toast.
Tests F-EXP-01, F-EXP-02, F-EXP-03, F-EXP-04, F-ASY-01, F-ASY-02, F-ASY-03, and F-ERR-01.
"""
import os
import zipfile
import pytest
import numpy as np
import cv2
from PIL import Image

from app.core.pipeline.pipeline_worker import PipelineWorker
from app.core.pipeline.batch_worker import BatchWorker
from app.core.pipeline.exporter import MangaExporter
from app.ui.widgets.toast import Toast
from app.ui.settings.settings_dialog import SettingsDialog
from app.ui.main_window import MainWindow
from app.core.config import AppConfig


# =========================================================================
# F-ASY-01, F-ASY-02, F-ASY-03: PipelineWorker & BatchWorker
# =========================================================================

def test_pipeline_worker_initialization():
    worker = PipelineWorker("dummy.png", {}, mode="full")
    assert worker.mode == "full"
    assert not worker._is_cancelled
    assert hasattr(worker, "sig_progress")
    assert hasattr(worker, "sig_step_done")
    assert hasattr(worker, "sig_finished")
    assert hasattr(worker, "sig_error")


def test_pipeline_worker_cancellation():
    worker = PipelineWorker("dummy.png", {})
    assert not worker._is_cancelled
    worker.cancel()
    assert worker._is_cancelled


def test_pipeline_worker_missing_file_error():
    worker = PipelineWorker("non_existent_image_12345.png", {})
    errors = []
    worker.sig_error.connect(lambda e: errors.append(e))
    worker.run()
    assert len(errors) == 1
    assert "不存在" in errors[0]


def test_batch_worker_initialization():
    items = [{"id": "1", "path": "p1.png"}]
    worker = BatchWorker(items, {}, export_dir="")
    assert len(worker.queue_items) == 1
    assert not worker._is_cancelled
    assert hasattr(worker, "sig_batch_progress")
    assert hasattr(worker, "sig_item_completed")
    assert hasattr(worker, "sig_batch_finished")
    assert hasattr(worker, "sig_item_failed")


def test_batch_worker_empty_queue():
    worker = BatchWorker([], {}, export_dir="")
    emitted = []
    worker.sig_batch_finished.connect(lambda s, f: emitted.append((s, f)))
    worker.run()
    assert len(emitted) == 1
    assert emitted[0] == (0, 0)


def test_batch_worker_cancellation():
    worker = BatchWorker([{"id": "1", "path": "test.png"}], {})
    worker.cancel()
    assert worker._is_cancelled


# =========================================================================
# F-EXP-02, F-EXP-03, F-EXP-04: Multi-Format Exporter
# =========================================================================

def test_exporter_single_image_png_jpg_webp(tmp_path):
    img = np.ones((200, 300, 3), dtype=np.uint8) * 180

    png_path = str(tmp_path / "out.png")
    assert MangaExporter.export_single_image(img, png_path, fmt="PNG")
    assert os.path.exists(png_path)
    assert os.path.getsize(png_path) > 0

    jpg_path = str(tmp_path / "out.jpg")
    assert MangaExporter.export_single_image(img, jpg_path, fmt="JPG", compressed=True)
    assert os.path.exists(jpg_path)
    assert os.path.getsize(jpg_path) > 0

    webp_path = str(tmp_path / "out.webp")
    assert MangaExporter.export_single_image(img, webp_path, fmt="WEBP", compressed=True)
    assert os.path.exists(webp_path)
    assert os.path.getsize(webp_path) > 0


def test_exporter_compile_chapter_pdf(tmp_path):
    # Create 3 synthetic manga images with natural names
    paths = []
    for name in ["p1.png", "p2.png", "p10.png"]:
        p = str(tmp_path / name)
        img = np.ones((400, 300, 3), dtype=np.uint8) * 200
        cv2.imwrite(p, img)
        paths.append(p)

    pdf_out = str(tmp_path / "chapter.pdf")
    success = MangaExporter.compile_chapter_pdf(paths, pdf_out)
    assert success
    assert os.path.exists(pdf_out)
    assert os.path.getsize(pdf_out) > 500


def test_exporter_package_chapter_zip(tmp_path):
    paths = []
    for name in ["page_01.png", "page_02.png"]:
        p = str(tmp_path / name)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(p, img)
        paths.append(p)

    zip_out = str(tmp_path / "chapter.zip")
    success = MangaExporter.package_chapter_zip(paths, zip_out)
    assert success
    assert os.path.exists(zip_out)

    with zipfile.ZipFile(zip_out, "r") as zf:
        names = zf.namelist()
        assert len(names) == 2
        assert "page_01.png" in names
        assert "page_02.png" in names


# =========================================================================
# F-ERR-01 & GUI Preferences
# =========================================================================

def test_toast_widget(qapp):
    toast = Toast()
    toast.show_message("测试通知", "success", duration_ms=500)
    assert toast.msg_label.text() == "测试通知"
    toast.close()


def test_settings_dialog_creation(qapp):
    cfg = AppConfig()
    dialog = SettingsDialog(cfg)
    assert dialog.nav_list.count() == 5
    assert dialog.stack.count() == 5
    dialog.close()
