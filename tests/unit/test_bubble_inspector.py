"""
tests/unit/test_bubble_inspector.py
Unit tests for M3 Interactive Bubble Editor & Inspector Panel in app/ architecture.
Tests F-EDT-01, F-EDT-02, F-EDT-03, F-EDT-04, F-EDT-05, and F-EDT-06.
"""
import pytest
import numpy as np
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtWidgets import QGraphicsItem

from app.ui.canvas.items.bubble_item import BubbleItem
from app.ui.canvas.view import MangaCanvasView
from app.ui.inspector.inspector_panel import InspectorPanel
from app.ui.main_window import MainWindow
from app.core.config import AppConfig


# =========================================================================
# F-EDT-01 & F-EDT-02: Bubble Item Geometry & Interaction
# =========================================================================

def test_bubble_item_geometry_and_normalization(qapp):
    block = {"id": "b1", "xmin": 10.0, "ymin": 20.0, "xmax": 30.0, "ymax": 40.0, "type": "bubble"}
    item = BubbleItem(block, img_w=1000, img_h=1000)
    assert item.img_w == 1000
    assert item.img_h == 1000
    assert item.rect().width() == 200.0
    assert item.rect().height() == 200.0
    assert item.pos().x() == 100.0
    assert item.pos().y() == 200.0


def test_bubble_item_flags_and_hover(qapp):
    block = {"id": "b1", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
    item = BubbleItem(block, 100, 100)
    flags = item.flags()
    assert bool(flags & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
    assert bool(flags & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
    assert bool(flags & QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    assert not item.is_hovered
    item.is_hovered = True
    assert item.is_hovered
    item.is_hovered = False
    assert not item.is_hovered


def test_bubble_item_signals(qapp):
    block = {"id": "b1", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
    item = BubbleItem(block, 100, 100)
    clicked = []
    item.signals.clicked.connect(lambda b: clicked.append(b))
    item.signals.clicked.emit(block)
    assert len(clicked) == 1
    assert clicked[0]["id"] == "b1"


def test_bubble_item_drag_position_update(qapp):
    block = {"id": "b1", "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0}
    item = BubbleItem(block, img_w=1000, img_h=1000)
    item.setPos(200.0, 300.0)
    item.itemChange(QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged, item.pos())
    assert block["xmin"] == 20.0
    assert block["ymin"] == 30.0
    assert block["xmax"] == 30.0
    assert block["ymax"] == 40.0


# =========================================================================
# F-EDT-02 & F-EDT-03: Canvas View Bubble Management
# =========================================================================

def test_canvas_view_creates_and_clears_bubbles(qapp):
    canvas = MangaCanvasView()
    img = np.ones((500, 500, 3), dtype=np.uint8) * 200
    blocks = [
        {"id": "b1", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30, "original_text": "t1", "translated_text": "y1"},
        {"id": "b2", "xmin": 50, "ymin": 50, "xmax": 70, "ymax": 70, "original_text": "t2", "translated_text": "y2"},
    ]
    canvas.set_data(img, blocks=blocks)
    assert len(canvas.bubble_items) == 2

    canvas.set_show_bubbles(False)
    for b in canvas.bubble_items:
        assert not b.isVisible()

    canvas.set_show_bubbles(True)
    for b in canvas.bubble_items:
        assert b.isVisible()

    canvas._clear_bubbles()
    assert len(canvas.bubble_items) == 0
    canvas.close()


def test_canvas_view_side_by_side_mode_clears_bubbles(qapp):
    canvas = MangaCanvasView()
    img = np.ones((500, 500, 3), dtype=np.uint8) * 200
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30}]
    canvas.set_data(img, blocks=blocks)
    assert len(canvas.bubble_items) == 1

    canvas.set_view_mode("side_by_side")
    assert len(canvas.bubble_items) == 0
    canvas.close()


# =========================================================================
# F-EDT-04 & F-EDT-05: Inspector Panel
# =========================================================================

def test_inspector_panel_tab_switching_and_population(qapp):
    panel = InspectorPanel(AppConfig())
    assert panel.tab_widget.count() == 3

    blocks = [
        {"id": "b1", "original_text": "原1", "translated_text": "译1", "xmin": 10, "ymin": 10},
        {"id": "b2", "original_text": "原2", "translated_text": "译2", "xmin": 40, "ymin": 40},
    ]
    panel.set_blocks(blocks)
    assert panel.bubble_list.count() == 2

    panel.select_block_by_id("b2")
    assert panel.selected_block is not None
    assert panel.selected_block["id"] == "b2"
    assert panel.trans_text_edit.toPlainText() == "译2"
    panel.close()


def test_inspector_panel_text_edits_emit_signal(qapp):
    panel = InspectorPanel(AppConfig())
    blocks = [{"id": "b1", "original_text": "原1", "translated_text": "译1", "xmin": 10, "ymin": 10}]
    panel.set_blocks(blocks)

    updated = []
    panel.sig_block_updated.connect(lambda b: updated.append(b))
    panel.trans_text_edit.setText("新的翻译")
    assert len(updated) >= 1
    assert blocks[0]["translated_text"] == "新的翻译"
    panel.close()


def test_inspector_panel_delete_block(qapp):
    panel = InspectorPanel(AppConfig())
    blocks = [{"id": "b1", "original_text": "原1", "translated_text": "译1", "xmin": 10, "ymin": 10}]
    panel.set_blocks(blocks)

    deleted = []
    panel.sig_block_deleted.connect(lambda bid: deleted.append(bid))
    panel._on_delete_block()
    assert len(deleted) == 1
    assert deleted[0] == "b1"
    panel.close()


def test_inspector_panel_style_controls(qapp):
    cfg = AppConfig()
    panel = InspectorPanel(cfg)

    # Populate blocks and select block 0
    blocks = [
        {"id": "b1", "original_text": "Hello", "translated_text": "你好"},
        {"id": "b2", "original_text": "World", "translated_text": "世界"},
    ]
    panel.set_blocks(blocks)
    assert panel.selected_block["id"] == "b1"

    # Enable block-level override
    panel.block_style_override_cb.setChecked(True)
    panel.block_font_combo.setCurrentIndex(1)  # 幼圆
    assert panel.selected_block["font_family_override"] == "幼圆"
    assert cfg.style.font_family != "幼圆"  # Global config remains isolated and clean!

    panel.block_bold_cb.setChecked(False)
    assert panel.selected_block["font_bold_override"] is False

    # Test Ctrl+Enter quick translation workflow
    panel.trans_text_edit.setPlainText("你好呀！")
    panel._on_apply_and_next_clicked()
    # Verified b1 was updated and selection automatically advanced to b2
    assert blocks[0]["translated_text"] == "你好呀！"
    assert panel.selected_block["id"] == "b2"

    panel.close()


# =========================================================================
# F-EDT-06: Main Window Re-Rendering & Block Deletion
# =========================================================================

def test_main_window_block_deletion_and_re_render(qapp):
    win = MainWindow()
    img = np.ones((600, 400, 3), dtype=np.uint8) * 240
    blocks = [
        {"id": "b1", "original_text": "A", "translated_text": "甲", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30},
        {"id": "b2", "original_text": "B", "translated_text": "乙", "xmin": 40, "ymin": 40, "xmax": 60, "ymax": 60},
    ]
    win.current_image_data = {
        "id": "p1",
        "path": "dummy.png",
        "blocks": blocks,
        "erased_img": img.copy(),
        "translated_img": None
    }
    win.canvas_view.original_cv = img.copy()

    # Re-render
    win._re_render_current_page()
    assert win.current_image_data["translated_img"] is not None
    assert win.canvas_view.translated_cv is not None

    # Delete block b1
    win._on_block_deleted("b1")
    assert len(win.current_image_data["blocks"]) == 1
    assert win.current_image_data["blocks"][0]["id"] == "b2"
    win.close()
