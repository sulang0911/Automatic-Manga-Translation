"""
tests/unit/test_page_list_multiselect.py
Tests Ctrl/Shift multi-selection and right-click batch context menu operations
in PageListWidget and MainWindow without breaking existing single-page workflows.
"""
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractItemView, QMenu

from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imwrite
from app.ui.sidebar.page_list import PageListWidget


@pytest.fixture
def temp_pages_env():
    tmp_dir = tempfile.mkdtemp(prefix="amt_multiselect_")
    paths = []
    for i in range(5):
        p = os.path.join(tmp_dir, f"page_{i:02d}.png")
        img = np.full((32, 32, 3), 240, dtype=np.uint8)
        safe_cv2_imwrite(p, img, ext=".png")
        paths.append(p)
    yield tmp_dir, paths
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _label_widget(page_list: PageListWidget, row: int):
    lw_item = page_list.list_widget.item(row)
    return page_list.list_widget.itemWidget(lw_item).name_label


def _item_center(page_list: PageListWidget, row: int):
    return page_list.list_widget.visualItemRect(page_list.list_widget.item(row)).center()


def _capture_menu_actions(page_list: PageListWidget, pos, trigger_text_substring: str = ""):
    """Invokes _show_context_menu while capturing built menu actions without blocking."""
    captured = []
    chosen_action = [None]

    def _fake_exec(menu_self, *args, **kwargs):
        for act in menu_self.actions():
            txt = act.text()
            captured.append((txt, act.isEnabled(), act.isSeparator()))
            if trigger_text_substring and trigger_text_substring in txt and act.isEnabled():
                chosen_action[0] = act
        return chosen_action[0]

    with patch.object(QMenu, "exec", _fake_exec):
        page_list._show_context_menu(pos)
    return captured


def test_ctrl_shift_multiselect_and_right_click_preserves_selection(qapp, temp_pages_env):
    """Ctrl and Shift clicks multi-select visible pages; right-clicking selected keeps selection."""
    _, paths = temp_pages_env
    pl = PageListWidget()
    pl.add_paths(paths)
    pl.resize(260, 420)
    pl.show()

    assert pl.list_widget.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection

    # Single click row 0 works normally
    QTest.mouseClick(_label_widget(pl, 0), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    assert [it["path"] for it in pl.get_selected_items()] == [paths[0]]

    # Ctrl+click row 3 adds row 3 while keeping row 0 (even when clicked out of order, sorted by list order)
    pl.list_widget.clearSelection()
    QTest.mouseClick(_label_widget(pl, 3), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QTest.mouseClick(_label_widget(pl, 1), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    selected_paths = [it["path"] for it in pl.get_selected_items()]
    assert selected_paths == [paths[1], paths[3]]

    # Right-clicking an already-selected item (row 3) shows multi-select menu and preserves [1, 3]
    actions = _capture_menu_actions(pl, _item_center(pl, 3))
    action_texts = [txt for txt, _, sep in actions if not sep]
    assert action_texts[0] == "已选中 2 个页面"
    assert any("批量翻译选中页面 (2 页" in t for t in action_texts)
    assert any("按缓存重译选中页面 (2 页" in t for t in action_texts)
    assert any("完全重译选中页面 (2 页" in t for t in action_texts)
    assert any("批量导出选中页面 (2 页)" in t for t in action_texts)
    assert any("批量清除选中页面缓存 (2 页)" in t for t in action_texts)
    assert any("从列表中移除选中页面 (2 页)" in t for t in action_texts)
    assert [it["path"] for it in pl.get_selected_items()] == [paths[1], paths[3]]

    # Shift+click from row 1 to row 4 selects contiguous range [1, 2, 3, 4]
    QTest.mouseClick(_label_widget(pl, 1), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QTest.mouseClick(_label_widget(pl, 4), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    assert [it["path"] for it in pl.get_selected_items()] == paths[1:5]

    # Hidden filtered items are excluded even if previously selected
    pl.update_item_status(pl.items_data[2]["id"], "completed", "已完成")
    pl.set_status_filter("queued")
    assert [it["path"] for it in pl.get_selected_items()] == [paths[1], paths[3], paths[4]]
    pl.set_status_filter("all")

    # Right-clicking an UNSELECTED item (row 0) selects only row 0 and opens single-page menu
    pl.list_widget.item(0).setSelected(False)
    # Simulate Qt's unselected right-click followed by customContextMenuRequested
    QTest.mouseClick(_label_widget(pl, 0), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier)
    single_actions = _capture_menu_actions(pl, _item_center(pl, 0))
    single_texts = [txt for txt, _, sep in single_actions if not sep]
    assert "翻译此页面" in single_texts
    assert not any("已选中" in t for t in single_texts)
    assert [it["path"] for it in pl.get_selected_items()] == [paths[0]]


def test_multiselect_right_click_emits_batch_signals_and_preserves_sibling_caches(qapp, temp_pages_env):
    """All right-click multi-selection batch actions operate strictly on the selected pages."""
    _, paths = temp_pages_env
    cache_mgr = get_cache_manager()
    for p in paths:
        cache_mgr.save_page_cache(
            p,
            blocks=[{"id": "b1", "original_text": "こんにちは", "translated_text": "你好"}],
            erased_img=np.full((16, 16, 3), 255, dtype=np.uint8),
            rendered_img=np.full((16, 16, 3), 200, dtype=np.uint8),
        )

    pl = PageListWidget()
    pl.add_paths(paths)
    pl.resize(260, 420)
    pl.show()

    # Select rows 1 and 3 (including a currently 'processing' item)
    pl.update_item_status(pl.items_data[1]["id"], "processing", "处理中")
    pl.list_widget.item(0).setSelected(False)
    pl.list_widget.item(1).setSelected(True)
    pl.list_widget.item(3).setSelected(True)
    center_3 = _item_center(pl, 3)

    got = {}
    pl.sig_batch_translate_selected.connect(lambda items: got.setdefault("batch", items))
    pl.sig_retranslate_selected_from_cache.connect(lambda items: got.setdefault("cached", items))
    pl.sig_retranslate_selected.connect(lambda items: got.setdefault("retrans", items))
    pl.sig_export_selected.connect(lambda items: got.setdefault("export", items))
    pl.sig_batch_cache_cleared.connect(lambda items: got.setdefault("cleared", items))
    pl.sig_pages_removed.connect(lambda ids: got.setdefault("removed", ids))

    _capture_menu_actions(pl, center_3, "批量翻译选中页面")
    _capture_menu_actions(pl, center_3, "按缓存重译选中页面")
    _capture_menu_actions(pl, center_3, "完全重译选中页面")
    _capture_menu_actions(pl, center_3, "批量导出选中页面")

    for key in ("batch", "cached", "retrans", "export"):
        assert [it["path"] for it in got[key]] == [paths[1], paths[3]]

    # Batch clear cache: ONLY selected paths [1, 3] are cleared; siblings [0, 2, 4] stay intact!
    _capture_menu_actions(pl, center_3, "批量清除选中页面缓存")
    assert [it["path"] for it in got["cleared"]] == [paths[1], paths[3]]
    assert not cache_mgr.has_cache(paths[1])["blocks"]
    assert not cache_mgr.has_cache(paths[3])["blocks"]
    for intact_p in (paths[0], paths[2], paths[4]):
        assert cache_mgr.has_cache(intact_p)["blocks"]
        assert cache_mgr.has_cache(intact_p)["rendered"]

    # Batch remove from list
    id1, id3 = pl.items_data[1]["id"], pl.items_data[3]["id"]
    _capture_menu_actions(pl, center_3, "从列表中移除选中页面")
    assert got["removed"] == [id1, id3]
    assert [it["path"] for it in pl.items_data] == [paths[0], paths[2], paths[4]]


def test_mainwindow_selected_batch_scoped_cache_and_active_canvas_safety(qapp, temp_pages_env):
    """MainWindow._start_selected_batch clears only selected caches and protects sibling/canvas state."""
    from app.ui.main_window import MainWindow

    tmp_dir, paths = temp_pages_env
    cache_mgr = get_cache_manager()
    for p in paths:
        cache_mgr.save_page_cache(
            p,
            blocks=[{"id": "b1", "original_text": "テスト", "translated_text": "测试"}],
            erased_img=np.full((16, 16, 3), 255, dtype=np.uint8),
            rendered_img=np.full((16, 16, 3), 210, dtype=np.uint8),
        )

    win = MainWindow()
    try:
        win.page_list.add_paths(paths[:4])
        # Populate in-memory state for all 4 items
        for it in win.page_list.items_data:
            it["blocks"] = [{"id": "b1"}]
            it["erased_img"] = np.zeros((8, 8, 3), dtype=np.uint8)
            it["translated_img"] = np.ones((8, 8, 3), dtype=np.uint8)

        # User is viewing unselected page 0 while selecting pages [1, 3]
        win.current_image_data = win.page_list.items_data[0]
        selected = [win.page_list.items_data[1], win.page_list.items_data[3]]

        with patch("app.core.pipeline.batch_worker.BatchWorker") as mock_worker_cls:
            mock_worker = MagicMock()
            mock_worker.isRunning.return_value = False
            mock_worker_cls.return_value = mock_worker

            win._start_selected_batch(selected, force_retranslate=True)

            mock_worker_cls.assert_called_once()
            kwargs = mock_worker_cls.call_args.kwargs
            assert [it["path"] for it in kwargs["queue_items"]] == [paths[1], paths[3]]
            assert kwargs["force_retranslate"] is True
            assert kwargs["clear_cache_dirs"] is False

        # Selected items cleared; unselected page 0 & 2 keep both disk cache AND in-memory state
        assert win.page_list.items_data[1]["blocks"] == []
        assert win.page_list.items_data[3]["blocks"] == []
        assert win.page_list.items_data[0]["blocks"] == [{"id": "b1"}]
        assert win.page_list.items_data[2]["blocks"] == [{"id": "b1"}]
        assert not cache_mgr.has_cache(paths[1])["blocks"]
        assert not cache_mgr.has_cache(paths[3])["blocks"]
        assert cache_mgr.has_cache(paths[0])["blocks"]
        assert cache_mgr.has_cache(paths[2])["blocks"]

        # Removing selected items [1, 3] leaves [0, 2] cleanly and keeps count accurate
        win.page_list.remove_items([selected[0]["id"], selected[1]["id"]])
        assert [it["path"] for it in win.page_list.items_data] == [paths[0], paths[2]]

        # Active batch worker guard: does NOT cancel running worker when selected action triggered
        running_worker = MagicMock()
        running_worker.isRunning.return_value = True
        win.active_batch_worker = running_worker
        win._start_selected_batch([win.page_list.items_data[0]], force_retranslate=False)
        running_worker.cancel.assert_not_called()
        win.active_batch_worker = None

        # _export_selected_pages forwards ONLY selected items & export_dir with clear_cache_dirs=False
        with patch("app.ui.main_window.QFileDialog.getExistingDirectory", return_value=tmp_dir), \
             patch("app.core.pipeline.batch_worker.BatchWorker") as mock_exp_worker_cls:
            mock_exp_worker = MagicMock()
            mock_exp_worker.isRunning.return_value = False
            mock_exp_worker_cls.return_value = mock_exp_worker

            win._export_selected_pages([win.page_list.items_data[1]])
            mock_exp_worker_cls.assert_called_once()
            exp_kwargs = mock_exp_worker_cls.call_args.kwargs
            assert [it["path"] for it in exp_kwargs["queue_items"]] == [paths[2]]
            assert exp_kwargs["export_dir"] == os.path.abspath(tmp_dir)
            assert exp_kwargs["clear_cache_dirs"] is False

        # Delete key removes all multi-selected items at once
        win.page_list.list_widget.item(0).setSelected(True)
        win.page_list.list_widget.item(1).setSelected(True)
        QTest.keyClick(win.page_list.list_widget, Qt.Key.Key_Delete)
        assert win.page_list.items_data == []
    finally:
        win.close()
