"""
tests/unit/test_ui_interactive_polish.py
Unit tests verifying the interactive polish features:
1. DragDropOverlay show / hide / layout.
2. ShortcutsDialog shortcuts cataloging and display.
3. BatchProgressPill progress updating, finish state, and cancel signal.
4. InPlaceBubbleEditor canvas floating editor, commit, and advance.
5. ContextualTypographyBar font sizing, orientation toggle, and signal emission.
6. CanvasZoomHud pager controls and set_page_info.
7. MangaCanvasView keyboard (A/D/PageUp/PageDown) and continuous wheel page flip.
8. MainWindow integration with HUD pager, shortcuts dialog, and batch pill.
"""
import pytest
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QKeyEvent, QWheelEvent
from PyQt6.QtWidgets import QWidget

from app.core.config import AppConfig
from app.ui.widgets.drag_overlay import DragDropOverlay
from app.ui.widgets.shortcuts_dialog import ShortcutsDialog
from app.ui.widgets.batch_progress_pill import BatchProgressPill
from app.ui.widgets.in_place_editor import InPlaceBubbleEditor, ContextualTypographyBar
from app.ui.canvas.view import MangaCanvasView, CanvasZoomHud
from app.ui.canvas.items.bubble_item import BubbleItem
from app.ui.main_window import MainWindow


def test_drag_drop_overlay(qapp):
    """Verify DragDropOverlay visibility and resize synchronization."""
    parent = QWidget()
    parent.resize(800, 600)
    parent.show()
    overlay = DragDropOverlay(parent)

    assert not overlay.isVisible()
    overlay.show_overlay()
    assert overlay.isVisible()
    assert overlay.geometry() == parent.rect()

    overlay.hide_overlay()
    assert not overlay.isVisible()
    parent.close()


def test_shortcuts_dialog(qapp):
    """Verify ShortcutsDialog instantiates with structured shortcut categories."""
    dlg = ShortcutsDialog()
    assert "Shortcuts" in dlg.windowTitle() or "快捷键" in dlg.windowTitle()
    assert dlg.isModal() is True

    # Ensure shortcut row widgets exist
    children = dlg.findChildren(QWidget)
    assert len(children) > 10
    dlg.close()


def test_batch_progress_pill(qapp):
    """Verify BatchProgressPill updates progress and emits cancel signal."""
    parent = QWidget()
    pill = BatchProgressPill(parent)

    pill.update_progress(current=3, total=10, step_text="Inpainting", eta_seconds=45)
    assert "3/10" in pill.status_lbl.text()
    assert "Inpainting" in pill.status_lbl.text()
    assert "45秒" in pill.eta_lbl.text()
    assert pill.progress_bar.value() == 30

    cancel_called = []
    pill.sig_cancel_requested.connect(lambda: cancel_called.append(True))
    pill.btn_cancel.click()
    assert len(cancel_called) == 1

    pill.show_finished(success_count=10, failed_count=0)
    assert "全部完成" in pill.status_lbl.text() or "10 成功" in pill.status_lbl.text()
    assert pill.progress_bar.value() == 100

    parent.close()


def test_canvas_zoom_hud_pager(qapp):
    """Verify CanvasZoomHud pager buttons and page label formatting."""
    parent = QWidget()
    hud = CanvasZoomHud(parent)

    hud.set_page_info(0, 0)
    assert hud.lbl_page.text() == "-- / --"
    assert hud.btn_prev_page.isEnabled() is False
    assert hud.btn_next_page.isEnabled() is False

    hud.set_page_info(1, 5)
    assert hud.lbl_page.text() == "01 / 05"
    assert hud.btn_prev_page.isEnabled() is False
    assert hud.btn_next_page.isEnabled() is True

    hud.set_page_info(3, 5)
    assert hud.lbl_page.text() == "03 / 05"
    assert hud.btn_prev_page.isEnabled() is True
    assert hud.btn_next_page.isEnabled() is True

    hud.set_page_info(5, 5)
    assert hud.lbl_page.text() == "05 / 05"
    assert hud.btn_prev_page.isEnabled() is True
    assert hud.btn_next_page.isEnabled() is False

    parent.close()


def test_in_place_bubble_editor(qapp):
    """Verify InPlaceBubbleEditor commits on Ctrl+Enter and updates block translation."""
    canvas = MangaCanvasView()
    canvas.resize(800, 600)
    canvas.show()
    fake_img = np.zeros((500, 500, 3), dtype=np.uint8)
    blocks = [
        {"id": "b1", "text": "オッス！", "translation": "你好！", "box": [10, 10, 40, 40]},
        {"id": "b2", "text": "元気？", "translation": "还好吗？", "box": [50, 50, 80, 80]},
    ]
    canvas.set_data(fake_img, blocks=blocks)

    # Double click bubble 1
    canvas._on_bubble_double_clicked(canvas.bubble_items[0].block_data)
    editor = canvas.in_place_editor

    assert editor.isVisible()
    assert editor.text_edit.toPlainText() == "你好！"

    # Simulate editing translation
    editor.text_edit.setPlainText("新的翻译文本")

    committed = []
    canvas.sig_commit_bubble_text.connect(lambda b, t: committed.append((b["id"], t)))
    editor._on_commit_and_next()

    assert len(committed) == 1
    assert committed[0] == ("b1", "新的翻译文本")
    assert canvas.bubble_items[0].block_data.get("translation") == "新的翻译文本"

    canvas.close()


def test_contextual_typography_bar(qapp):
    """Verify ContextualTypographyBar changes font size and orientation."""
    canvas = MangaCanvasView()
    canvas.resize(800, 600)
    canvas.show()
    fake_img = np.zeros((500, 500, 3), dtype=np.uint8)
    blocks = [
        {"id": "b1", "text": "オッス！", "translation": "你好！", "box": [10, 10, 40, 40], "style": {"font_size": 14, "direction": "vertical"}},
    ]
    canvas.set_data(fake_img, blocks=blocks)

    # Click bubble to show typography bar
    canvas._on_bubble_clicked(canvas.bubble_items[0].block_data)
    bar = canvas.typography_bar
    assert bar.isVisible()

    # Step font size up
    bar._step_font_size(2)
    assert bar.lbl_size.text() == "16"
    assert bar.block_data["style"]["font_size"] == 16

    # Step font size down
    bar._step_font_size(-2)
    assert bar.lbl_size.text() == "14"
    assert bar.block_data["style"]["font_size"] == 14

    # Toggle orientation
    bar._on_toggle_orientation()
    assert bar.block_data["style"]["direction"] == "horizontal"

    canvas.close()


def test_canvas_keyboard_navigation(qapp):
    """Verify A/D and PageUp/PageDown keys emit page navigation signals."""
    canvas = MangaCanvasView()
    nav_signals = []
    canvas.sig_prev_page.connect(lambda: nav_signals.append("prev"))
    canvas.sig_next_page.connect(lambda: nav_signals.append("next"))

    # Key D -> next
    event_d = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.NoModifier, "d")
    canvas.keyPressEvent(event_d)
    assert nav_signals[-1] == "next"

    # Key A -> prev
    event_a = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "a")
    canvas.keyPressEvent(event_a)
    assert nav_signals[-1] == "prev"

    # Key PageDown -> next
    event_pgdn = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_PageDown, Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(event_pgdn)
    assert nav_signals[-1] == "next"

    # Key PageUp -> prev
    event_pgup = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_PageUp, Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(event_pgup)
    assert nav_signals[-1] == "prev"

    canvas.close()


def test_main_window_page_navigation(qapp, tmp_path):
    """Verify MainWindow page navigation updates current page and HUD."""
    win = MainWindow()
    win.show()

    # Create dummy images
    import cv2
    img1_path = str(tmp_path / "page_01.png")
    img2_path = str(tmp_path / "page_02.png")
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.imwrite(img1_path, dummy)
    cv2.imwrite(img2_path, dummy)

    win._on_paths_dropped([img1_path, img2_path])
    assert len(win.page_list.items_data) == 2
    assert win.canvas_view.hud.lbl_page.text() == "01 / 02"

    # Navigate to next page
    win._navigate_page(1)
    assert win.current_image_data["path"] == img2_path
    assert win.canvas_view.hud.lbl_page.text() == "02 / 02"

    # Navigate to previous page
    win._navigate_page(-1)
    assert win.current_image_data["path"] == img1_path
    assert win.canvas_view.hud.lbl_page.text() == "01 / 02"

    # Clear pages
    win.page_list.clear_all()
    assert win.canvas_view.hud.lbl_page.text() == "-- / --"

    win.close()
