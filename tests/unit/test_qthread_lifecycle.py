"""
tests/unit/test_qthread_lifecycle.py
Regression tests ensuring QThread workers and thumbnail thread pools are cleanly
cancelled, detached, or drained before owner widgets are destroyed — preventing
Qt's fatal `QThread: Destroyed while thread '' is still running` crash (-1073740791).
"""
import gc
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget

from app.core import hardware
from app.core.config import AppConfig
from app.ui.main_window import MainWindow
from app.ui.settings.settings_dialog import SettingsDialog
from app.ui.widgets.thumbnail_loader import AsyncThumbnailManager
from desktop.ui.main_window import MainWindow as LegacyMainWindow
from desktop.ui.settings_dialog import SettingsDialog as LegacySettingsDialog


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _is_cleaned_up(worker) -> bool:
    if worker in hardware._DETACHED_QTHREADS:
        return False
    try:
        return not worker.isRunning()
    except RuntimeError:
        return True


def _pump_events_until(qapp, predicate, timeout_s: float = 2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return predicate()


class _SlowWorker(QThread):
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)
    sig_progress = pyqtSignal(int, str)

    def __init__(self, delay_s: float = 0.25, ignore_cancel: bool = False, parent=None):
        super().__init__(parent)
        self.delay_s = delay_s
        self.ignore_cancel = ignore_cancel
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        deadline = time.monotonic() + self.delay_s
        while time.monotonic() < deadline:
            if self.cancelled and not self.ignore_cancel:
                return
            time.sleep(0.01)
        self.sig_finished.emit({"ok": True})


class _SlowTestEngine:
    def __init__(self, *args, **kwargs):
        pass

    def test_connection(self, text: str) -> str:
        time.sleep(0.25)
        return "pong"


def test_retire_qthread_detaches_slow_thread_and_releases_on_finish(qapp):
    parent = QWidget()
    worker = _SlowWorker(delay_s=0.25, ignore_cancel=True, parent=parent)
    received = []
    worker.sig_finished.connect(received.append)
    worker.start()
    time.sleep(0.02)

    hardware.retire_qthread(worker, wait_ms=20, signals=(worker.sig_finished,))

    assert worker.cancelled is True
    assert worker.parent() is None
    assert worker in hardware._DETACHED_QTHREADS

    # Destroying the former parent while worker is still running must not abort
    del parent
    gc.collect()
    assert received == []

    assert _pump_events_until(qapp, lambda: worker not in hardware._DETACHED_QTHREADS, timeout_s=2.0)
    assert received == []


@pytest.mark.parametrize("action", ["reject", "accept", "close"])
def test_app_settings_dialog_closes_safely_while_test_worker_running(qapp, action):
    with patch("app.ui.settings.settings_dialog.TranslationEngine", _SlowTestEngine):
        dlg = SettingsDialog(AppConfig())
        dlg._run_connection_test()
        worker = dlg._test_worker
        assert worker is not None

        time.sleep(0.02)
        getattr(dlg, action)()

        assert dlg._test_worker is None
        assert worker._is_cancelled is True
        assert worker.parent() is None

        del dlg
        gc.collect()

        assert _pump_events_until(qapp, lambda: _is_cleaned_up(worker))


def test_app_settings_dialog_retrigger_retires_previous_worker(qapp):
    with patch("app.ui.settings.settings_dialog.TranslationEngine", _SlowTestEngine):
        dlg = SettingsDialog(AppConfig())
        dlg._run_connection_test()
        first = dlg._test_worker
        time.sleep(0.02)

        dlg._run_connection_test()
        second = dlg._test_worker

        assert second is not None
        assert second is not first
        assert first._is_cancelled is True
        assert first.parent() is None

        dlg.reject()
        del dlg
        gc.collect()

        assert _pump_events_until(
            qapp,
            lambda: _is_cleaned_up(first) and _is_cleaned_up(second),
        )


def test_legacy_settings_dialog_closes_safely_while_test_worker_running(qapp):
    stub_cfg = SimpleNamespace(data={})
    with patch("desktop.ui.settings_dialog.TranslationEngine", _SlowTestEngine):
        dlg = LegacySettingsDialog(stub_cfg)
        dlg._run_connection_test()
        worker = dlg._test_worker
        assert worker is not None

        time.sleep(0.02)
        dlg.reject()

        assert dlg._test_worker is None
        assert worker._is_cancelled is True
        assert worker.parent() is None

        del dlg
        gc.collect()

        assert _pump_events_until(qapp, lambda: _is_cleaned_up(worker))


def test_main_window_close_and_replace_retires_all_workers(qapp, monkeypatch):
    win = MainWindow()
    monkeypatch.setattr(win, "_save_workspace_session", lambda: None)

    w_active = _SlowWorker(delay_s=0.25, ignore_cancel=True, parent=win)
    w_batch = _SlowWorker(delay_s=0.2, ignore_cancel=False, parent=win)
    w_block = _SlowWorker(delay_s=0.2, ignore_cancel=False, parent=win)
    w_archive = _SlowWorker(delay_s=0.2, ignore_cancel=False, parent=win)

    stale_events = []
    w_active.sig_finished.connect(lambda res: stale_events.append(res))

    win.active_worker = w_active
    win.active_batch_worker = w_batch
    win.block_worker = w_block
    win.active_archive_worker = w_archive

    for thr in (w_active, w_batch, w_block, w_archive):
        thr.start()

    time.sleep(0.02)

    # Simulate replacing a live slot before the old thread finishes
    win._retire_worker_slot("active_worker", wait_ms=20)
    assert win.active_worker is None
    assert w_active.parent() is None
    assert w_active.cancelled is True

    # Attach an uncooperative worker longer than wait_ms=1500 to verify detach path in closeEvent
    w_active2 = _SlowWorker(delay_s=1.7, ignore_cancel=True, parent=win)
    win.active_worker = w_active2
    w_active2.start()

    # Ensure thumbnail manager shutdown helper also succeeds cleanly
    mgr = AsyncThumbnailManager.instance()
    assert mgr is not None

    win.close()

    assert win.active_worker is None
    assert win.active_batch_worker is None
    assert win.block_worker is None
    assert win.active_archive_worker is None
    assert w_active2.parent() is None
    assert w_active2 in hardware._DETACHED_QTHREADS

    del win
    gc.collect()

    assert _pump_events_until(
        qapp,
        lambda: all(_is_cleaned_up(w) for w in (w_active, w_active2, w_batch, w_block, w_archive)),
        timeout_s=3.0,
    )
    assert stale_events == []


def test_legacy_main_window_close_retires_workers(qapp):
    win = LegacyMainWindow()
    w1 = _SlowWorker(delay_s=0.2, ignore_cancel=True, parent=win)
    w2 = _SlowWorker(delay_s=0.2, ignore_cancel=False, parent=win)
    win.active_worker = w1
    win.active_batch_worker = w2
    w1.start()
    w2.start()
    time.sleep(0.02)

    win.close()

    assert win.active_worker is None
    assert win.active_batch_worker is None

    del win
    gc.collect()

    assert _pump_events_until(
        qapp,
        lambda: _is_cleaned_up(w1) and _is_cleaned_up(w2),
        timeout_s=3.0,
    )
