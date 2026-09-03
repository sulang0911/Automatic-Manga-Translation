import pytest
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtWidgets import QGraphicsItem

from desktop.ui.bubble_item import BubbleItem
from desktop.ui.canvas_view import CanvasView
from desktop.ui.inspector_panel import InspectorPanel
from desktop.ui.main_window import MainWindow
from desktop.core.config_manager import ConfigManager

# ============================================================================
# F-EDT-01: Interactive Canvas Bubble Overlays
# ============================================================================

def test_fedt_01_bubble_item_instantiation(qapp):
    block = {"id": "b1", "xmin": 10.0, "ymin": 20.0, "xmax": 30.0, "ymax": 40.0, "type": "bubble"}
    item = BubbleItem(block, img_w=1000, img_h=1000)
    assert item.img_w == 1000
    assert item.img_h == 1000
    assert item.rect().width() == 200.0
    assert item.rect().height() == 200.0
    assert item.pos().x() == 100.0
    assert item.pos().y() == 200.0

def test_fedt_01_flags_set(qapp):
    block = {"id": "b1", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
    item = BubbleItem(block, 100, 100)
    flags = item.flags()
    assert bool(flags & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
    assert bool(flags & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
    assert bool(flags & QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

def test_fedt_01_hover_state_toggle(qapp):
    block = {"id": "b1", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
    item = BubbleItem(block, 100, 100)
    assert not item.is_hovered
    item.is_hovered = True
    assert item.is_hovered
    item.is_hovered = False
    assert not item.is_hovered

def test_fedt_01_clicked_signal(qapp):
    block = {"id": "b1", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
    item = BubbleItem(block, 100, 100)
    emitted = []
    item.signals.clicked.connect(lambda b: emitted.append(b))
    item.signals.clicked.emit(block)
    assert len(emitted) == 1
    assert emitted[0]["id"] == "b1"

def test_fedt_01_double_clicked_signal(qapp):
    block = {"id": "b2", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
    item = BubbleItem(block, 100, 100)
    emitted = []
    item.signals.double_clicked.connect(lambda b: emitted.append(b))
    item.signals.double_clicked.emit(block)
    assert len(emitted) == 1
    assert emitted[0]["id"] == "b2"

# ============================================================================
# F-EDT-02: Direct Canvas Bounding Box Drag & Resize
# ============================================================================

def test_fedt_02_item_position_change_updates_data(qapp):
    block = {"id": "b1", "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0}
    item = BubbleItem(block, img_w=1000, img_h=1000)
    # Move item to (200, 300)
    item.setPos(200.0, 300.0)
    item.itemChange(QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged, item.pos())
    assert block["xmin"] == 20.0
    assert block["ymin"] == 30.0
    assert block["xmax"] == 30.0
    assert block["ymax"] == 40.0

def test_fedt_02_position_change_emits_signal(qapp):
    block = {"id": "b1", "xmin": 10.0, "ymin": 10.0, "xmax": 20.0, "ymax": 20.0}
    item = BubbleItem(block, img_w=1000, img_h=1000)
    emitted = []
    item.signals.changed.connect(lambda b: emitted.append(b))
    item.itemChange(QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged, item.pos())
    assert len(emitted) == 1

def test_fedt_02_canvas_view_creates_bubble_items(qapp, sample_manga_image_np, sample_translation_blocks):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np, blocks=sample_translation_blocks)
    assert len(canvas.bubble_items) == len(sample_translation_blocks)
    canvas.close()

def test_fedt_02_bubble_visibility_toggle(qapp, sample_manga_image_np, sample_translation_blocks):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np, blocks=sample_translation_blocks)
    canvas.set_show_bubbles(False)
    for b in canvas.bubble_items:
        assert not b.isVisible()
    canvas.set_show_bubbles(True)
    for b in canvas.bubble_items:
        assert b.isVisible()
    canvas.close()

def test_fedt_02_side_by_side_hides_bubbles(qapp, sample_manga_image_np, sample_translation_blocks):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np, translated_cv=sample_manga_image_np.copy(), blocks=sample_translation_blocks)
    canvas.set_view_mode("side_by_side")
    # In side by side mode bubbles are not created to prevent coordinate misalignment
    assert len(canvas.bubble_items) == 0
    canvas.close()

# ============================================================================
# F-EDT-03: Manual Bubble Creation & Deletion
# ============================================================================

def test_fedt_03_delete_block_from_main_window(qapp, sample_translation_blocks):
    win = MainWindow()
    win.current_image_data = {
        "id": "img1",
        "path": "test.png",
        "blocks": sample_translation_blocks.copy()
    }
    win._on_block_deleted("b1001")
    assert len(win.current_image_data["blocks"]) == 1
    assert win.current_image_data["blocks"][0]["id"] == "b1002"
    win.close()

def test_fedt_03_delete_block_via_inspector(qapp, sample_translation_blocks):
    panel = InspectorPanel(ConfigManager())
    panel.set_blocks(sample_translation_blocks)
    emitted = []
    panel.sig_block_deleted.connect(lambda b_id: emitted.append(b_id))
    # Select first item
    panel.select_block_by_id("b1001")
    panel._on_delete_block()
    assert len(emitted) == 1
    assert emitted[0] == "b1001"
    panel.close()

def test_fedt_03_delete_non_existent_block_safe(qapp, sample_translation_blocks):
    win = MainWindow()
    win.current_image_data = {"id": "img1", "path": "test.png", "blocks": sample_translation_blocks.copy()}
    win._on_block_deleted("non_existent_id")
    assert len(win.current_image_data["blocks"]) == len(sample_translation_blocks)
    win.close()

def test_fedt_03_clear_bubbles_in_canvas(qapp, sample_manga_image_np, sample_translation_blocks):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np, blocks=sample_translation_blocks)
    assert len(canvas.bubble_items) == 2
    canvas._clear_bubbles()
    assert len(canvas.bubble_items) == 0
    canvas.close()

def test_fedt_03_empty_blocks_list_delete_safe(qapp):
    win = MainWindow()
    win.current_image_data = {"id": "img1", "path": "test.png", "blocks": []}
    win._on_block_deleted("any_id")
    assert win.current_image_data["blocks"] == []
    win.close()

# ============================================================================
# F-EDT-04: Inspector Text & Type Editor
# ============================================================================

def test_fedt_04_inspector_populate_blocks(qapp, sample_translation_blocks):
    panel = InspectorPanel(ConfigManager())
    panel.set_blocks(sample_translation_blocks)
    assert panel.bubble_list.count() == 2
    panel.close()

def test_fedt_04_inspector_item_clicked_populates_edits(qapp, sample_translation_blocks):
    panel = InspectorPanel(ConfigManager())
    panel.set_blocks(sample_translation_blocks)
    panel._on_bubble_list_clicked(panel.bubble_list.item(0))
    assert panel.orig_text_edit.toPlainText() == sample_translation_blocks[0]["original_text"]
    assert panel.trans_text_edit.toPlainText() == sample_translation_blocks[0]["translated_text"]
    panel.close()

def test_fedt_04_inspector_text_change_updates_block(qapp, sample_translation_blocks):
    panel = InspectorPanel(ConfigManager())
    panel.set_blocks(sample_translation_blocks)
    panel._on_bubble_list_clicked(panel.bubble_list.item(0))
    panel.trans_text_edit.setText("修改后的翻译文本")
    assert sample_translation_blocks[0]["translated_text"] == "修改后的翻译文本"
    panel.close()

def test_fedt_04_select_block_by_id(qapp, sample_translation_blocks):
    panel = InspectorPanel(ConfigManager())
    panel.set_blocks(sample_translation_blocks)
    panel.select_block_by_id("b1002")
    assert panel.selected_block is not None
    assert panel.selected_block["id"] == "b1002"
    panel.close()

def test_fedt_04_tab_switching(qapp):
    panel = InspectorPanel(ConfigManager())
    assert panel.tab_widget.count() == 3
    panel.tab_widget.setCurrentIndex(1)
    assert panel.tab_widget.currentIndex() == 1
    panel.close()

# ============================================================================
# F-EDT-05: Per-Block Typographic Overrides
# ============================================================================

def test_fedt_05_font_family_combo_updates_cfg(qapp):
    cfg_mgr = ConfigManager()
    panel = InspectorPanel(cfg_mgr)
    panel.font_combo.setCurrentText("SimHei")
    assert cfg_mgr.get("font_family") == "SimHei"
    panel.close()

def test_fedt_05_font_size_slider_updates_cfg(qapp):
    cfg_mgr = ConfigManager()
    panel = InspectorPanel(cfg_mgr)
    panel.size_slider.setValue(15)  # 1.5x
    assert cfg_mgr.get("font_size_scale") == 1.5
    assert panel.size_val_label.text() == "1.5x"
    panel.close()

def test_fedt_05_auto_fit_checkbox_toggle(qapp):
    cfg_mgr = ConfigManager()
    panel = InspectorPanel(cfg_mgr)
    panel.auto_fit_cb.setChecked(False)
    assert cfg_mgr.get("auto_fit_font_size") is False
    panel.auto_fit_cb.setChecked(True)
    assert cfg_mgr.get("auto_fit_font_size") is True
    panel.close()

def test_fedt_05_bold_checkbox_toggle(qapp):
    cfg_mgr = ConfigManager()
    panel = InspectorPanel(cfg_mgr)
    panel.bold_cb.setChecked(True)
    assert cfg_mgr.get("font_bold") is True
    panel.bold_cb.setChecked(False)
    assert cfg_mgr.get("font_bold") is False
    panel.close()

def test_fedt_05_block_title_displays_selected_id(qapp, sample_translation_blocks):
    panel = InspectorPanel(ConfigManager())
    panel.set_blocks(sample_translation_blocks)
    panel.select_block_by_id("b1001")
    assert "b1001" in panel.block_title.text()
    panel.close()

# ============================================================================
# F-EDT-06: Debounced Canvas Re-rendering
# ============================================================================

def test_fedt_06_text_edit_emits_block_updated_signal(qapp, sample_translation_blocks):
    panel = InspectorPanel(ConfigManager())
    panel.set_blocks(sample_translation_blocks)
    panel.select_block_by_id("b1001")
    emitted = []
    panel.sig_block_updated.connect(lambda b: emitted.append(b))
    panel.trans_text_edit.setText("实时重绘测试")
    assert len(emitted) >= 1
    panel.close()

def test_fedt_06_main_window_re_render(qapp, sample_manga_image_np, sample_translation_blocks):
    win = MainWindow()
    win.current_image_data = {
        "id": "test1",
        "path": "test.png",
        "blocks": sample_translation_blocks,
        "erased_img": sample_manga_image_np.copy(),
        "translated_img": None
    }
    win._re_render_current_page()
    assert win.current_image_data["translated_img"] is not None
    assert win.canvas_view.translated_cv is not None
    win.close()

def test_fedt_06_re_render_without_image_data_safe(qapp):
    win = MainWindow()
    win.current_image_data = None
    # Should not crash
    win._re_render_current_page()
    win.close()

def test_fedt_06_re_render_with_empty_blocks_safe(qapp, sample_manga_image_np):
    win = MainWindow()
    win.current_image_data = {
        "id": "test1",
        "path": "test.png",
        "blocks": [],
        "erased_img": sample_manga_image_np.copy()
    }
    win._re_render_current_page()
    # Nothing re-rendered, no crash
    win.close()

def test_fedt_06_apply_button_emits_re_render(qapp):
    panel = InspectorPanel(ConfigManager())
    emitted = []
    panel.sig_re_render_requested.connect(lambda: emitted.append(True))
    panel.apply_block_btn.click()
    assert len(emitted) == 1
    panel.close()
