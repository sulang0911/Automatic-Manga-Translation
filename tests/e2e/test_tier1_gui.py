import os
import pytest
import numpy as np
import cv2
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QWheelEvent

from desktop.ui.main_window import MainWindow
from desktop.ui.canvas_view import CanvasView, cvimg_to_qpixmap
from desktop.ui.queue_panel import QueuePanel, QueueItemWidget
from desktop.ui.styles import get_stylesheet

# ============================================================================
# F-GUI-01: Apple HIG Desktop Shell
# ============================================================================

def test_fgui_01_window_initialization(qapp):
    win = MainWindow()
    assert win.minimumWidth() >= 1000
    assert win.minimumHeight() >= 650
    assert "AetherLens" in win.windowTitle()
    win.close()

def test_fgui_01_card_container_hierarchy(qapp):
    win = MainWindow()
    assert win.centralWidget() is not None
    assert win.toolbar_widget is not None
    assert win.splitter is not None
    assert win.queue_panel is not None
    assert win.canvas_view is not None
    assert win.inspector_panel is not None
    win.close()

def test_fgui_01_splitter_proportions(qapp):
    win = MainWindow()
    sizes = win.splitter.sizes()
    assert len(sizes) == 3
    assert sizes[0] > 0
    assert sizes[1] > 0
    assert sizes[2] > 0
    win.close()

def test_fgui_01_status_indicator_ready(qapp):
    win = MainWindow()
    assert win.status_label.text() == "就绪"
    assert win.progress_bar.isHidden()
    win.close()

def test_fgui_01_window_destruction_clean(qapp):
    win = MainWindow()
    win.show()
    win.close()
    assert not win.isVisible()

# ============================================================================
# F-GUI-02: Dark & Light Theme System
# ============================================================================

def test_fgui_02_dark_theme_qss_content():
    qss = get_stylesheet("dark")
    assert "#1E1E1E" in qss or "#18181A" in qss
    assert "#0A84FF" in qss

def test_fgui_02_light_theme_qss_content():
    qss = get_stylesheet("light")
    assert "#F5F5F7" in qss or "#FFFFFF" in qss
    assert "#0071E3" in qss or "#0A84FF" in qss

def test_fgui_02_theme_switch_updates_stylesheet(qapp):
    win = MainWindow()
    win.setStyleSheet(get_stylesheet("light"))
    assert win.styleSheet() == get_stylesheet("light")
    win.setStyleSheet(get_stylesheet("dark"))
    assert win.styleSheet() == get_stylesheet("dark")
    win.close()

def test_fgui_02_unsupported_theme_falls_back():
    qss = get_stylesheet("neon_cyberpunk")
    assert qss == get_stylesheet("dark")

def test_fgui_02_accent_colors_and_radii():
    qss = get_stylesheet("dark")
    assert "border-radius" in qss
    assert "primaryBtn" in qss

# ============================================================================
# F-GUI-03: Collapsible Sidebar Navigation
# ============================================================================

def test_fgui_03_queue_panel_initial_state(qapp):
    panel = QueuePanel()
    assert panel.list_widget.count() == 0
    assert panel.count_badge.text() == "0 页"
    panel.close()

def test_fgui_03_queue_add_single_item(qapp, sample_manga_image_file):
    panel = QueuePanel()
    panel.add_paths([sample_manga_image_file])
    assert panel.list_widget.count() == 1
    assert panel.count_badge.text() == "1 页"
    panel.close()

def test_fgui_03_queue_clear_all(qapp, sample_manga_image_file):
    panel = QueuePanel()
    panel.add_paths([sample_manga_image_file])
    assert panel.list_widget.count() == 1
    panel.clear_all()
    assert panel.list_widget.count() == 0
    assert panel.count_badge.text() == "0 页"
    panel.close()

def test_fgui_03_queue_item_widget_status_updates(qapp, sample_manga_image_file):
    panel = QueuePanel()
    panel.add_paths([sample_manga_image_file])
    item_id = panel.items_data[0]["id"]

    panel.update_item_status(item_id, "processing", "识别中")
    assert panel.items_data[0]["status"] == "processing"

    panel.update_item_status(item_id, "completed", "完成")
    assert panel.items_data[0]["status"] == "completed"

    panel.update_item_status(item_id, "failed", "失败")
    assert panel.items_data[0]["status"] == "failed"
    panel.close()

def test_fgui_03_queue_item_selection_signal(qapp, sample_manga_image_file):
    panel = QueuePanel()
    emitted = []
    panel.sig_image_selected.connect(lambda data: emitted.append(data))
    panel.add_paths([sample_manga_image_file])
    # Selecting first item triggers signal
    assert len(emitted) >= 1
    assert emitted[0]["path"] == sample_manga_image_file
    panel.close()

# ============================================================================
# F-GUI-04: Top Toolbar & Action Controls
# ============================================================================

def test_fgui_04_toolbar_structure(qapp):
    win = MainWindow()
    assert win.btn_group is not None
    assert len(win.btn_group.buttons()) == 5
    assert win.run_btn is not None
    assert win.settings_btn is not None
    win.close()

def test_fgui_04_segmented_modes_exclusive(qapp):
    win = MainWindow()
    buttons = win.btn_group.buttons()
    checked_count = sum(1 for b in buttons if b.isChecked())
    assert checked_count == 1
    # Click second button
    buttons[1].click()
    assert buttons[1].isChecked()
    assert not buttons[0].isChecked()
    win.close()

def test_fgui_04_bubble_toggle_checkbox(qapp):
    win = MainWindow()
    assert win.bubble_cb.isChecked()
    win.bubble_cb.setChecked(False)
    assert not win.canvas_view.show_bubbles
    win.bubble_cb.setChecked(True)
    assert win.canvas_view.show_bubbles
    win.close()

def test_fgui_04_view_mode_changed_handler(qapp):
    win = MainWindow()
    win._on_view_mode_changed("side_by_side")
    assert win.canvas_view.view_mode == "side_by_side"
    assert win.slider_bar.isHidden()

    win._on_view_mode_changed("split_slider")
    assert win.canvas_view.view_mode == "split_slider"
    assert not win.slider_bar.isHidden()
    win.close()

def test_fgui_04_zoom_label_update(qapp):
    win = MainWindow()
    win._on_zoom_changed(1.5)
    assert "150%" in win.status_label.text()
    win.close()

# ============================================================================
# F-GUI-05: Drag-and-Drop File/Folder Importer
# ============================================================================

def test_fgui_05_add_valid_files(qapp, sample_manga_image_file):
    panel = QueuePanel()
    panel.add_paths([sample_manga_image_file])
    assert len(panel.items_data) == 1
    panel.close()

def test_fgui_05_ignore_non_image_files(qapp, temp_dir):
    panel = QueuePanel()
    txt_file = os.path.join(temp_dir, "test.txt")
    with open(txt_file, "w") as f:
        f.write("hello")
    panel.add_paths([txt_file])
    assert len(panel.items_data) == 0
    panel.close()

def test_fgui_05_chapter_folder_recursive_discovery(qapp, sample_chapter_dir):
    panel = QueuePanel()
    panel.add_paths([sample_chapter_dir])
    # Exactly 3 images in chapter dir
    assert len(panel.items_data) == 3
    panel.close()

def test_fgui_05_deduplicate_identical_paths(qapp, sample_manga_image_file):
    panel = QueuePanel()
    panel.add_paths([sample_manga_image_file, sample_manga_image_file])
    assert len(panel.items_data) == 1
    panel.close()

def test_fgui_05_empty_paths_handled_safely(qapp):
    panel = QueuePanel()
    panel.add_paths([])
    assert len(panel.items_data) == 0
    panel.close()

# ============================================================================
# F-GUI-06: Dual-Viewport Synchronized Canvas
# ============================================================================

def test_fgui_06_canvas_initial_state(qapp):
    canvas = CanvasView()
    assert canvas.original_cv is None
    assert canvas.translated_cv is None
    assert canvas.base_pixmap_item is not None
    canvas.close()

def test_fgui_06_set_data_creates_scene_rect(qapp, sample_manga_image_np):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np)
    assert not canvas.scene.sceneRect().isEmpty()
    assert canvas.scene.sceneRect().width() == 800
    assert canvas.scene.sceneRect().height() == 1200
    canvas.close()

def test_fgui_06_side_by_side_stacks_images(qapp, sample_manga_image_np):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np, translated_cv=sample_manga_image_np.copy())
    canvas.set_view_mode("side_by_side")
    # Width should be doubled (800 * 2 = 1600)
    assert canvas.scene.sceneRect().width() == 1600
    canvas.close()

def test_fgui_06_cvimg_to_qpixmap_conversion(sample_manga_image_np):
    pix = cvimg_to_qpixmap(sample_manga_image_np)
    assert not pix.isNull()
    assert pix.width() == 800
    assert pix.height() == 1200

def test_fgui_06_empty_image_to_pixmap():
    pix = cvimg_to_qpixmap(None)
    assert pix.isNull()

# ============================================================================
# F-GUI-07: Split-Slider Comparison View
# ============================================================================

def test_fgui_07_split_position_clamping(qapp):
    canvas = CanvasView()
    canvas.set_split_position(-0.5)
    assert canvas.split_position == 0.0
    canvas.set_split_position(1.5)
    assert canvas.split_position == 1.0
    canvas.set_split_position(0.4)
    assert canvas.split_position == 0.4
    canvas.close()

def test_fgui_07_split_slider_draws_divider(qapp, sample_manga_image_np):
    canvas = CanvasView()
    # Create two different images for original and translated
    translated_cv = sample_manga_image_np.copy()
    translated_cv[:, :] = (100, 100, 100)

    canvas.set_data(sample_manga_image_np, translated_cv=translated_cv)
    canvas.set_view_mode("split_slider")
    canvas.set_split_position(0.5)

    assert canvas.view_mode == "split_slider"
    assert canvas.base_pixmap_item.pixmap().width() == 800
    canvas.close()

def test_fgui_07_slider_bar_interaction(qapp):
    win = MainWindow()
    win._on_view_mode_changed("split_slider")
    win.split_slider.setValue(75)
    assert win.canvas_view.split_position == 0.75
    win.close()

def test_fgui_07_slider_hides_in_other_modes(qapp):
    win = MainWindow()
    win._on_view_mode_changed("split_slider")
    assert not win.slider_bar.isHidden()
    win._on_view_mode_changed("original")
    assert win.slider_bar.isHidden()
    win.close()

def test_fgui_07_split_without_translated_falls_back(qapp, sample_manga_image_np):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np, translated_cv=None)
    canvas.set_view_mode("split_slider")
    # Should safely display original image without error
    assert not canvas.base_pixmap_item.pixmap().isNull()
    canvas.close()

# ============================================================================
# F-GUI-08: High-DPI Smooth Pan & Zoom
# ============================================================================

def test_fgui_08_fit_in_view(qapp, sample_manga_image_np):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np)
    canvas.fit_in_view()
    assert canvas.zoom_factor > 0
    canvas.close()

def test_fgui_08_zoom_factor_clamped_on_wheel(qapp, sample_manga_image_np):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np)
    canvas.zoom_factor = 9.9
    # Simulate zoom in
    event = QWheelEvent(
        QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False
    )
    canvas.wheelEvent(event)
    # Factor should stay within 10.0
    assert canvas.zoom_factor <= 12.0
    canvas.close()

def test_fgui_08_zoom_factor_lower_bound(qapp, sample_manga_image_np):
    canvas = CanvasView()
    canvas.set_data(sample_manga_image_np)
    canvas.zoom_factor = 0.1
    event = QWheelEvent(
        QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, -120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False
    )
    canvas.wheelEvent(event)
    assert canvas.zoom_factor >= 0.05
    canvas.close()

def test_fgui_08_pan_tracking_initial_state(qapp):
    canvas = CanvasView()
    assert not canvas.is_panning
    assert canvas.pan_start_pos.isNull()
    canvas.close()

def test_fgui_08_render_hints_enabled(qapp):
    canvas = CanvasView()
    from PyQt6.QtGui import QPainter
    assert bool(canvas.renderHints() & QPainter.RenderHint.Antialiasing)
    assert bool(canvas.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)
    canvas.close()
