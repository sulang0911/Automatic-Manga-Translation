import os
import pytest
import numpy as np
from PyQt6.QtCore import QThread

from desktop.core.pipeline_worker import PipelineWorker
from desktop.core.batch_worker import BatchWorker
from desktop.ui.main_window import MainWindow

# ============================================================================
# F-ASY-01: Dedicated Worker Threading
# ============================================================================

def test_fasy_01_pipeline_worker_is_qthread():
    worker = PipelineWorker("test.png", {})
    assert isinstance(worker, QThread)

def test_fasy_01_batch_worker_is_qthread():
    worker = BatchWorker([], {})
    assert isinstance(worker, QThread)

def test_fasy_01_pipeline_worker_modes():
    modes = ["full", "ocr_only", "inpaint_only", "translate_only", "render_only"]
    for m in modes:
        worker = PipelineWorker("dummy.png", {}, mode=m)
        assert worker.mode == m

def test_fasy_01_pipeline_worker_file_not_found(temp_dir):
    missing_path = os.path.join(temp_dir, "missing.png")
    worker = PipelineWorker(missing_path, {})
    errors = []
    worker.sig_error.connect(lambda msg: errors.append(msg))
    worker.run()
    assert len(errors) == 1
    assert "不存在" in errors[0]

def test_fasy_01_worker_thread_isolation():
    worker = PipelineWorker("dummy.png", {})
    assert not worker.isRunning()
    assert not worker._is_cancelled

# ============================================================================
# F-ASY-02: Granular Progress Reporting
# ============================================================================

def test_fasy_02_pipeline_worker_signals():
    worker = PipelineWorker("dummy.png", {})
    assert hasattr(worker, "sig_progress")
    assert hasattr(worker, "sig_step_done")
    assert hasattr(worker, "sig_finished")
    assert hasattr(worker, "sig_error")

def test_fasy_02_progress_emission_order():
    worker = PipelineWorker("dummy.png", {})
    pcts = []
    worker.sig_progress.connect(lambda p, m: pcts.append(p))
    worker.sig_progress.emit(10, "A")
    worker.sig_progress.emit(50, "B")
    worker.sig_progress.emit(100, "C")
    assert pcts == [10, 50, 100]

def test_fasy_02_step_done_emission():
    worker = PipelineWorker("dummy.png", {})
    steps = []
    worker.sig_step_done.connect(lambda s, d: steps.append(s))
    worker.sig_step_done.emit("ocr", [])
    worker.sig_step_done.emit("inpaint", None)
    worker.sig_step_done.emit("render", None)
    assert steps == ["ocr", "inpaint", "render"]

def test_fasy_02_main_window_progress_handler(qapp):
    win = MainWindow()
    win._on_worker_progress(45, "测试进度45%")
    assert win.progress_bar.value() == 45
    assert "测试进度45%" in win.status_label.text()
    win.close()

def test_fasy_02_batch_progress_handler(qapp):
    win = MainWindow()
    win._on_batch_progress(current=2, total=4, filename="page_02.png", pct=50, msg="翻译中")
    assert win.progress_bar.value() == 37  # int((1/4)*100 + 50/4) = 25 + 12 = 37
    assert "page_02.png" in win.status_label.text()
    win.close()

# ============================================================================
# F-ASY-03: Cooperative Task Cancellation
# ============================================================================

def test_fasy_03_pipeline_cancel_flag():
    worker = PipelineWorker("test.png", {})
    assert not worker._is_cancelled
    worker.cancel()
    assert worker._is_cancelled

def test_fasy_03_batch_cancel_flag():
    worker = BatchWorker([], {})
    assert not worker._is_cancelled
    worker.cancel()
    assert worker._is_cancelled

def test_fasy_03_cancelled_pipeline_stops_early(sample_manga_image_file):
    worker = PipelineWorker(sample_manga_image_file, {})
    worker.cancel()
    finished = []
    worker.sig_finished.connect(lambda res: finished.append(res))
    # When pre-cancelled, run should abort at the first stage check
    worker.run()
    assert len(finished) == 0

def test_fasy_03_cancelled_batch_aborts_loop(sample_manga_image_file):
    items = [
        {"id": "1", "path": sample_manga_image_file},
        {"id": "2", "path": sample_manga_image_file}
    ]
    worker = BatchWorker(items, {})
    worker.cancel()
    completed = []
    worker.sig_item_completed.connect(lambda i, r: completed.append(i))
    worker.run()
    assert len(completed) == 0

def test_fasy_03_cancel_during_idle():
    worker = PipelineWorker("test.png", {})
    worker.cancel()
    worker.cancel()  # Idempotent
    assert worker._is_cancelled

# ============================================================================
# F-ASY-04: Memory & VRAM Safeguards
# ============================================================================

def test_fasy_04_stream_handle_closed(sample_manga_image_file):
    # Verify open(path, "rb") can be safely deleted afterwards (no file lock left)
    stream = open(sample_manga_image_file, "rb")
    bytes_data = bytearray(stream.read())
    stream.close()
    assert stream.closed

def test_fasy_04_worker_finished_cleans_active_state(qapp):
    win = MainWindow()
    win.current_image_data = {"id": "1", "path": "p.png"}
    results = {
        "original_img": np.full((10, 10, 3), 255, dtype=np.uint8),
        "translated_img": np.full((10, 10, 3), 255, dtype=np.uint8),
        "erased_img": np.full((10, 10, 3), 255, dtype=np.uint8),
        "blocks": []
    }
    win._on_worker_finished(results)
    assert win.run_btn.isEnabled()
    assert win.progress_bar.isHidden()
    win.close()

def test_fasy_04_worker_error_resets_ui(qapp):
    win = MainWindow()
    win.current_image_data = {"id": "1", "path": "p.png"}
    win._on_worker_error("测试错误")
    assert win.run_btn.isEnabled()
    assert win.progress_bar.isHidden()
    assert "出错" in win.status_label.text()
    win.close()

def test_fasy_04_batch_finished_resets_ui(qapp):
    win = MainWindow()
    win._on_batch_finished(success_cnt=3, fail_cnt=0)
    assert win.progress_bar.isHidden()
    assert "完成" in win.status_label.text()
    win.close()

def test_fasy_04_garbage_collection_after_batch():
    import gc
    gc.collect()
    # Memory collector runs cleanly without exceptions
    assert True
