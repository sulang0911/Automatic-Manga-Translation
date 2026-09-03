"""
tests/unit/test_ui_ux_improvements.py
Unit tests verifying Priority 1, 2, and 3 UI/UX improvements:
- Decoupled single-block typography styling in InspectorPanel
- QSplitter elastic layout and Ctrl+Enter keyboard workflow
- Interactive BubbleItem 8-directional resize handles and body dragging
- MangaCanvasView update_translated_image and selection preservation
- Manual bubble drag creation on Canvas and CanvasZoomHud
"""
import pytest
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent

from app.core.config import AppConfig
from app.core.models import TranslationBlock
from app.ui.inspector.inspector_panel import InspectorPanel
from app.ui.canvas.items.bubble_item import (
    BubbleItem,
    HANDLE_BOTTOM_RIGHT,
    HANDLE_TOP_LEFT,
    HANDLE_RIGHT_MID,
    HANDLE_BOTTOM_MID,
    HANDLE_NONE,
)
from app.ui.canvas.view import MangaCanvasView
from app.ui.main_window import MainWindow


def test_inspector_panel_block_decoupled_styling(qapp):
    """Verify modifying single-block style in Inspector does NOT mutate global AppConfig."""
    cfg = AppConfig()
    initial_global_font = cfg.style.font_family
    initial_global_bold = cfg.style.font_bold

    panel = InspectorPanel(config=cfg)
    blocks = [
        {"id": "b1", "original_text": "オッス！", "translated_text": "你好！", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40},
        {"id": "b2", "original_text": "元気？", "translated_text": "还好吗？", "xmin": 50, "ymin": 50, "xmax": 80, "ymax": 80},
    ]
    panel.set_blocks(blocks)

    # Initially not overridden
    assert panel.block_style_override_cb.isChecked() is False
    assert panel.selected_block["id"] == "b1"

    # Enable override and modify font & bold for b1
    panel.block_style_override_cb.setChecked(True)
    panel.block_font_combo.setCurrentIndex(1)  # 幼圆
    panel.block_bold_cb.setChecked(False)

    # Check that b1 has overrides
    assert panel.selected_block["font_family_override"] == "幼圆"
    assert panel.selected_block["font_bold_override"] is False

    # Check that global config was NOT mutated
    assert cfg.style.font_family == initial_global_font
    assert cfg.style.font_bold == initial_global_bold

    # Reset b1 to inherited
    panel._on_reset_block_style_clicked()
    assert panel.block_style_override_cb.isChecked() is False
    assert panel.selected_block["font_family_override"] is None
    assert panel.selected_block["font_bold_override"] is None

    panel.close()


def test_inspector_ctrl_enter_fast_translation_flow(qapp):
    """Verify Ctrl+Enter commits translation and automatically advances to next bubble."""
    cfg = AppConfig()
    panel = InspectorPanel(config=cfg)

    blocks = [
        {"id": "b1", "original_text": "ライン1", "translated_text": "旧翻译1"},
        {"id": "b2", "original_text": "ライン2", "translated_text": "旧翻译2"},
    ]
    panel.set_blocks(blocks)
    assert panel.selected_block["id"] == "b1"

    # Simulate typing in translation edit
    panel.trans_text_edit.setPlainText("新翻译1完成")
    panel._on_apply_and_next_clicked()

    # b1 updated
    assert blocks[0]["translated_text"] == "新翻译1完成"
    # Auto-advanced to b2
    assert panel.selected_block["id"] == "b2"
    assert panel.trans_text_edit.toPlainText() == "旧翻译2"

    panel.close()


def test_bubble_item_8_directional_handles_and_body_dragging(qapp):
    """Verify BubbleItem body moving, 8-directional resize handles, and release commit."""
    block = {
        "id": "test_b",
        "xmin": 20.0,
        "ymin": 20.0,
        "xmax": 60.0,
        "ymax": 60.0,
    }
    img_w, img_h = 1000, 1000
    item = BubbleItem(block, img_w, img_h)
    item.setSelected(True)

    rect = item.rect()

    # 1. Check handle detection
    assert item._get_handle_at(QPointF(rect.right(), rect.bottom())) == HANDLE_BOTTOM_RIGHT
    assert item._get_handle_at(QPointF(rect.left(), rect.top())) == HANDLE_TOP_LEFT
    assert item._get_handle_at(QPointF(rect.right(), rect.center().y())) == HANDLE_RIGHT_MID
    assert item._get_handle_at(QPointF(rect.center().x(), rect.bottom())) == HANDLE_BOTTOM_MID

    # 2. Test handle resize
    item._active_handle = HANDLE_BOTTOM_RIGHT
    item._drag_start_pos = QPointF(rect.right(), rect.bottom())
    item._drag_start_rect = rect
    item._drag_start_scene_pos = item.pos()

    drag_event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(rect.right() + 80, rect.bottom() + 80),
        QPointF(rect.right() + 80, rect.bottom() + 80),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    item.mouseMoveEvent(drag_event)

    assert item.rect().width() == rect.width() + 80
    assert item.rect().height() == rect.height() + 80
    assert block["xmax"] > 60.0
    assert block["ymax"] > 60.0

    # 3. Test body dragging
    item._active_handle = HANDLE_NONE
    item._is_moving_body = True
    start_pos = item.pos()
    item._drag_start_pos = QPointF(100, 100)
    item._drag_start_scene_pos = start_pos

    body_drag_event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(150, 160),
        QPointF(150, 160),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    item.mouseMoveEvent(body_drag_event)

    assert item.pos().x() == start_pos.x() + 50
    assert item.pos().y() == start_pos.y() + 60

    # 4. Test release commit
    commits = []
    item.signals.geometry_commit.connect(lambda b: commits.append(b))
    rel_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(150, 160),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    item.mouseReleaseEvent(rel_event)
    assert len(commits) == 1
    assert item._is_moving_body is False
    assert item._active_handle == HANDLE_NONE


def test_canvas_draw_tool_and_zoom_hud(qapp):
    """Verify CanvasZoomHud interactions and manual bubble creation."""
    view = MangaCanvasView()
    canvas = np.full((800, 600, 3), 255, dtype=np.uint8)
    view.set_data(original_cv=canvas)

    assert hasattr(view, "hud")
    assert view.hud is not None

    # Test draw tool toggle
    assert view.tool_mode == "select"
    view.toggle_draw_tool()
    assert view.tool_mode == "draw"
    assert view.hud.btn_draw.isChecked() is True

    # Test manual bubble creation signal
    created_bubbles = []
    view.sig_bubble_created.connect(lambda b: created_bubbles.append(b))

    # Simulate drawing a box on scene
    view._is_drawing_rect = True
    view._draw_start_scene_pt = QPointF(50, 50)
    from PyQt6.QtWidgets import QGraphicsRectItem
    from PyQt6.QtCore import QRectF
    view._rubber_band_item = QGraphicsRectItem(QRectF(50, 50, 200, 150))
    view._scene.addItem(view._rubber_band_item)

    # Release
    release_event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(250, 200),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    view.mouseReleaseEvent(release_event)

    assert len(created_bubbles) == 1
    new_b = created_bubbles[0]
    assert "xmin" in new_b and "ymin" in new_b
    assert new_b["xmin"] > 0
    assert new_b["xmax"] > new_b["xmin"]

    view.close()


def test_canvas_update_translated_image_preserves_selection(qapp):
    """Verify update_translated_image updates background without destroying bubble items or selection."""
    view = MangaCanvasView()
    canvas = np.full((400, 400, 3), 255, dtype=np.uint8)
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40}]
    view.set_data(original_cv=canvas, blocks=blocks)

    item = view.bubble_items[0]
    item.setSelected(True)
    assert item.isSelected() is True

    # Update translated image
    new_trans = np.full((400, 400, 3), 128, dtype=np.uint8)
    view.update_translated_image(new_trans)

    # Verify bubble item is preserved and still selected
    assert len(view.bubble_items) == 1
    assert view.bubble_items[0] is item
    assert item.isSelected() is True

    # Test select_bubble_by_id
    view.select_bubble_by_id("b1")
    assert item.isSelected() is True
    view.select_bubble_by_id("non_existent")
    assert item.isSelected() is False

    view.close()


def test_main_window_manual_bubble_creation_integration(qapp, tmp_path):
    """Verify MainWindow integrates canvas manual bubble creation into inspector and blocks."""
    import cv2
    win = MainWindow()

    img_path = str(tmp_path / "page_manual.png")
    cv2.imwrite(img_path, np.full((300, 300, 3), 255, dtype=np.uint8))
    win.page_list.add_paths([img_path])
    win._on_page_selected(win.page_list.items_data[0])

    # Simulate bubble created signal from canvas
    mock_block = {
        "id": "mb_1234",
        "xmin": 15.0,
        "ymin": 15.0,
        "xmax": 45.0,
        "ymax": 45.0,
        "original_text": "",
        "translated_text": "",
        "type": "bubble",
    }
    win._on_bubble_created(mock_block)

    # Check block added to page data
    assert len(win.current_image_data["blocks"]) == 1
    assert win.current_image_data["blocks"][0]["id"] == "mb_1234"

    # Check inspector selected it
    assert win.inspector_panel.selected_block["id"] == "mb_1234"

    win.close()


def test_bubble_drag_moves_rendered_text(qapp, tmp_path):
    """Verify dragging a bubble updates block coordinates and moves rendered text on canvas."""
    import cv2
    from PyQt6.QtTest import QTest

    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    img_path = str(tmp_path / "page_drag.png")
    clean_img = np.full((600, 600, 3), 255, dtype=np.uint8)
    cv2.imwrite(img_path, clean_img)
    win.page_list.add_paths([img_path])
    win._on_page_selected(win.page_list.items_data[0])

    b = {
        "id": "b_move",
        "xmin": 10.0,
        "ymin": 10.0,
        "xmax": 20.0,
        "ymax": 20.0,
        "translated_text": "移动测试",
    }
    win.current_image_data["blocks"] = [b]
    win.current_image_data["erased_img"] = clean_img.copy()
    win.canvas_view.set_data(original_cv=clean_img, translated_cv=clean_img, erased_cv=clean_img, blocks=[b])
    win.inspector_panel.set_blocks([b])

    # Initial render: text is at (60, 60)
    win._re_render_current_page()
    assert np.min(win.canvas_view.translated_cv[60:120, 60:120]) < 50

    # Drag bubble by 150px
    item = win.canvas_view.bubble_items[0]
    center = win.canvas_view.mapFromScene(item.pos() + QPointF(20, 20))
    dest = center + QPoint(150, 150)

    QTest.mousePress(win.canvas_view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)
    QTest.mouseMove(win.canvas_view.viewport(), dest)
    QTest.mouseRelease(win.canvas_view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, dest)
    QTest.qWait(200)

    # 1. Coordinates updated
    blk = win.current_image_data["blocks"][0]
    assert blk["xmin"] > 25.0
    assert blk["ymin"] > 25.0

    # 2. Old position cleaned up (pure 255)
    assert np.min(win.canvas_view.translated_cv[60:120, 60:120]) == 255

    # 3. New position has text (< 50)
    px_x1 = int(blk["xmin"] / 100.0 * 600)
    px_y1 = int(blk["ymin"] / 100.0 * 600)
    px_x2 = int(blk["xmax"] / 100.0 * 600)
    px_y2 = int(blk["ymax"] / 100.0 * 600)
    assert np.min(win.canvas_view.translated_cv[px_y1:px_y2, px_x1:px_x2]) < 50

    win.close()

