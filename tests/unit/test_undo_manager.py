"""
tests/unit/test_undo_manager.py
Unit tests for UndoManager, PageSnapshot, and block equality checking.
"""
import pytest
from app.core.undo_manager import UndoManager, PageSnapshot, are_blocks_equal


def test_undo_manager_initial_state():
    mgr = UndoManager(max_depth=10)
    assert not mgr.can_undo()
    assert not mgr.can_redo()
    assert mgr.get_undo_description() == ""
    assert mgr.get_redo_description() == ""
    assert mgr.undo(PageSnapshot.create("page1.png", [])) is None
    assert mgr.redo(PageSnapshot.create("page1.png", [])) is None


def test_undo_manager_push_and_undo_redo():
    mgr = UndoManager(max_depth=10)

    # Initial state S0
    s0 = PageSnapshot.create("page1.png", [{"id": "b1", "translated_text": "初始"}], description="初始状态")

    # Action 1: Add bubble (Push S0 before Action 1)
    mgr.push(PageSnapshot.create(s0.page_path, s0.blocks, description="添加气泡"))

    assert mgr.can_undo()
    assert not mgr.can_redo()
    assert mgr.get_undo_description() == "添加气泡"

    # Current state S1 (after Action 1)
    s1 = PageSnapshot.create("page1.png", [
        {"id": "b1", "translated_text": "初始"},
        {"id": "b2", "translated_text": "新建"}
    ], description="当前状态")

    # Execute Undo
    restored_s0 = mgr.undo(s1)
    assert restored_s0 is not None
    assert len(restored_s0.blocks) == 1
    assert restored_s0.blocks[0]["id"] == "b1"
    assert restored_s0.description == "添加气泡"

    # Now can redo, cannot undo
    assert not mgr.can_undo()
    assert mgr.can_redo()
    assert mgr.get_redo_description() == "添加气泡"

    # Execute Redo
    restored_s1 = mgr.redo(s0)
    assert restored_s1 is not None
    assert len(restored_s1.blocks) == 2
    assert restored_s1.blocks[1]["id"] == "b2"

    assert mgr.can_undo()
    assert not mgr.can_redo()


def test_undo_manager_new_action_clears_redo():
    mgr = UndoManager(max_depth=10)
    s0 = PageSnapshot.create("page.png", [{"id": "b1"}])
    mgr.push(s0)

    s1 = PageSnapshot.create("page.png", [{"id": "b1"}, {"id": "b2"}])
    mgr.undo(s1)
    assert mgr.can_redo()

    # Now user does a brand new action
    s2 = PageSnapshot.create("page.png", [{"id": "b1"}, {"id": "b3"}])
    mgr.push(s2)

    assert not mgr.can_redo()
    assert mgr.can_undo()


def test_undo_manager_max_depth():
    mgr = UndoManager(max_depth=3)
    for i in range(5):
        mgr.push(PageSnapshot.create("page.png", [{"id": f"b{i}"}], description=f"操作{i}"))

    # Should retain at most 3 items
    assert len(mgr._undo_stack) == 3
    # The oldest items (0, 1) should have been dropped
    assert mgr._undo_stack[0].description == "操作2"
    assert mgr._undo_stack[1].description == "操作3"
    assert mgr._undo_stack[2].description == "操作4"


def test_are_blocks_equal():
    b1 = [{"id": "1", "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0, "original_text": "A", "translated_text": "甲", "type": "bubble"}]
    b2 = [{"id": "1", "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0, "original_text": "A", "translated_text": "甲", "type": "bubble"}]
    assert are_blocks_equal(b1, b2)

    # Coords change
    b3 = [{"id": "1", "xmin": 15.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0, "original_text": "A", "translated_text": "甲", "type": "bubble"}]
    assert not are_blocks_equal(b1, b3)

    # Text change
    b4 = [{"id": "1", "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0, "original_text": "A", "translated_text": "乙", "type": "bubble"}]
    assert not are_blocks_equal(b1, b4)

    # Length change
    assert not are_blocks_equal(b1, [])


def test_mainwindow_undo_redo():
    from PyQt6.QtWidgets import QApplication
    import numpy as np
    from app.ui.main_window import MainWindow

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    win = MainWindow()
    fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
    win.current_image_data = {
        "path": "test_page.png",
        "blocks": [{"id": "b1", "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0, "original_text": "A", "translated_text": "甲", "type": "bubble"}],
        "erased_img": fake_img,
        "translated_img": fake_img
    }
    win.canvas_view.original_cv = fake_img
    win.canvas_view.blocks = win.current_image_data["blocks"]

    assert not win.undo_btn.isEnabled()
    assert not win.redo_btn.isEnabled()

    # 1. Create a bubble
    new_b = {"id": "b2", "xmin": 30.0, "ymin": 30.0, "xmax": 40.0, "ymax": 40.0, "original_text": "B", "translated_text": "乙", "type": "bubble"}
    win._on_bubble_created(new_b)

    assert len(win.current_image_data["blocks"]) == 2
    assert win.undo_btn.isEnabled()
    assert not win.redo_btn.isEnabled()

    # 2. Undo creation
    win._undo()
    assert len(win.current_image_data["blocks"]) == 1
    assert win.current_image_data["blocks"][0]["id"] == "b1"
    assert not win.undo_btn.isEnabled()
    assert win.redo_btn.isEnabled()

    # 3. Redo creation
    win._redo()
    assert len(win.current_image_data["blocks"]) == 2
    assert win.undo_btn.isEnabled()
    assert not win.redo_btn.isEnabled()

    # 4. Delete bubble
    win._on_block_deleted("b2")
    assert len(win.current_image_data["blocks"]) == 1
    assert win.undo_btn.isEnabled()

    # 5. Undo delete
    win._undo()
    assert len(win.current_image_data["blocks"]) == 2
    assert win.current_image_data["blocks"][1]["id"] == "b2"

    win.close()

