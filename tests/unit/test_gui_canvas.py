"""
tests/unit/test_gui_canvas.py
Comprehensive unit test suite for Milestone M2:
Apple HIG Theme Tokens, Vector Icons, Card, Segmented Control, Progress Pill,
High-DPI Canvas Engine (BackgroundItem, SplitSliderItem, Scene, View),
Sidebar (NavRail, PageList, DropZone), and MainWindow Shell.
"""
import os
import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QPainter

from app.ui.theme.tokens import get_tokens, build_stylesheet, DARK_TOKENS, LIGHT_TOKENS
from app.ui.theme.icons import get_icon, render_svg_pixmap
from app.ui.widgets.card import CardWidget
from app.ui.widgets.segmented_control import SegmentedControl
from app.ui.widgets.progress_pill import ProgressPill, StatusDot
from app.ui.canvas.items.background_item import BackgroundItem
from app.ui.canvas.items.split_slider_item import SplitSliderItem
from app.ui.canvas.scene import MangaCanvasScene
from app.ui.canvas.view import MangaCanvasView, cvimg_to_qpixmap
from app.ui.sidebar.nav_rail import NavRail
from app.ui.sidebar.page_list import PageListWidget, natural_sort_key
from app.ui.sidebar.drop_zone import DropZoneWidget
from app.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_image_cv():
    img = np.zeros((1200, 800, 3), dtype=np.uint8)
    img[:, :] = (240, 240, 240)
    return img


# =============================================================================
# 1. Theme, Tokens & Icon Engine Tests
# =============================================================================

def test_theme_tokens_dark_and_light():
    dark = get_tokens("dark")
    assert dark.name == "dark"
    assert dark.bg_base == "#18181B"
    assert dark.bg_surface == "#202024"
    assert dark.accent_primary == "#3B82F6"
    assert dark.radius_lg == 12

    light = get_tokens("light")
    assert light.name == "light"
    assert light.bg_base == "#F8F9FA"
    assert light.bg_surface == "#FFFFFF"
    assert light.accent_primary == "#2563EB"

    # Default fallback
    unknown = get_tokens("custom_neon")
    assert unknown.name == "dark"


def test_build_stylesheet_generation():
    qss_dark = build_stylesheet(DARK_TOKENS)
    assert "#18181B" in qss_dark
    assert "card" in qss_dark
    assert "primaryBtn" in qss_dark
    assert "border-radius" in qss_dark

    qss_light = build_stylesheet(LIGHT_TOKENS)
    assert "#F8F9FA" in qss_light
    assert "#FFFFFF" in qss_light


def test_icon_rendering(qapp):
    pix = render_svg_pixmap("split", color="#3B82F6", size=24)
    assert not pix.isNull()
    assert pix.width() == 24
    assert pix.height() == 24

    icon = get_icon("sparkles", color="#A1A1AA", active_color="#3B82F6", size=18)
    assert not icon.isNull()
    assert not icon.pixmap(18, 18).isNull()


# =============================================================================
# 2. Reusable Apple HIG Widget Tests
# =============================================================================

def test_card_widget_creation(qapp):
    card = CardWidget(title="测试卡片", subtitle="副标题说明")
    assert card.property("class") == "card"
    assert card._title_label is not None
    assert card._title_label.text() == "测试卡片"
    assert card._subtitle_label.text() == "副标题说明"
    card.close()


def test_segmented_control_toggling(qapp):
    seg = SegmentedControl()
    b1 = seg.add_segment("opt1", "选项 1")
    b2 = seg.add_segment("opt2", "选项 2")
    b3 = seg.add_segment("opt3", "选项 3")

    assert len(seg.buttons()) == 3
    assert seg.current_segment() == "opt1"
    assert b1.isChecked()

    signals_emitted = []
    seg.sig_segment_changed.connect(lambda k: signals_emitted.append(k))

    b2.click()
    assert seg.current_segment() == "opt2"
    assert b2.isChecked()
    assert not b1.isChecked()
    assert signals_emitted == ["opt2"]

    seg.set_selected("opt3")
    assert seg.current_segment() == "opt3"
    assert b3.isChecked()
    seg.close()


def test_progress_pill_states(qapp):
    pill = ProgressPill()
    assert pill.text() == "就绪"
    assert not pill.is_progress_visible()

    pill.set_status("processing", "正在翻译第 3 页...", 45)
    assert pill.text() == "正在翻译第 3 页..."
    assert pill.is_progress_visible()
    assert pill._progress_bar.value() == 45

    pill.set_status("completed", "章节翻译完成")
    assert not pill.is_progress_visible()
    pill.close()


# =============================================================================
# 3. Canvas Graphics Items Tests
# =============================================================================

def test_cvimg_to_qpixmap_conversion(sample_image_cv):
    pix = cvimg_to_qpixmap(sample_image_cv)
    assert not pix.isNull()
    assert pix.width() == 800
    assert pix.height() == 1200

    empty_pix = cvimg_to_qpixmap(None)
    assert empty_pix.isNull()

    gray = np.zeros((100, 100), dtype=np.uint8)
    gray_pix = cvimg_to_qpixmap(gray)
    assert not gray_pix.isNull()
    assert gray_pix.width() == 100


def test_background_item_modes(sample_image_cv):
    item = BackgroundItem()
    orig_pix = cvimg_to_qpixmap(sample_image_cv)
    trans_pix = cvimg_to_qpixmap(sample_image_cv.copy())

    item.set_pixmaps(orig_pix, trans_pix)

    # Single mode: rect width is 800
    item.set_mode("translated")
    assert item.boundingRect().width() == 800
    assert item.boundingRect().height() == 1200

    # Side-by-side mode: rect width doubles to 1600
    item.set_mode("side_by_side")
    assert item.boundingRect().width() == 1600
    assert item.boundingRect().height() == 1200


def test_split_slider_item_geometry_and_clamping(sample_image_cv):
    item = SplitSliderItem()
    orig_pix = cvimg_to_qpixmap(sample_image_cv)
    trans_pix = cvimg_to_qpixmap(sample_image_cv.copy())

    item.set_pixmaps(orig_pix, trans_pix)
    assert item.boundingRect().width() == 800
    assert item.boundingRect().height() == 1200

    # Test clamping
    item.set_split_ratio(-0.2)
    assert item.split_ratio() == 0.0

    item.set_split_ratio(1.5)
    assert item.split_ratio() == 1.0

    item.set_split_ratio(0.65)
    assert abs(item.split_ratio() - 0.65) < 1e-4


# =============================================================================
# 4. Canvas Scene & Viewport Tests
# =============================================================================

def test_canvas_scene_mode_switching(sample_image_cv):
    scene = MangaCanvasScene()
    orig_pix = cvimg_to_qpixmap(sample_image_cv)
    trans_pix = cvimg_to_qpixmap(sample_image_cv.copy())

    scene.set_images(orig_pix, trans_pix)
    assert scene.sceneRect().width() == 800

    scene.set_view_mode("split_slider")
    assert scene.split_slider_item.isVisible()
    assert not scene.background_item.isVisible()

    scene.set_view_mode("side_by_side")
    assert not scene.split_slider_item.isVisible()
    assert scene.background_item.isVisible()
    assert scene.sceneRect().width() == 1600


def test_canvas_view_initialization_and_render_hints(qapp):
    view = MangaCanvasView()
    assert view.scene is not None
    assert bool(view.renderHints() & QPainter.RenderHint.Antialiasing)
    assert bool(view.renderHints() & QPainter.RenderHint.SmoothPixmapTransform)
    assert not view.is_panning
    assert view.zoom_factor == 1.0
    view.close()


def test_canvas_view_set_data_and_view_modes(qapp, sample_image_cv):
    view = MangaCanvasView()
    view.set_data(sample_image_cv, translated_cv=sample_image_cv.copy())
    assert view.scene.sceneRect().width() == 800
    assert view.scene.sceneRect().height() == 1200

    view.set_view_mode("side_by_side")
    assert view.scene.sceneRect().width() == 1600
    assert view.view_mode == "side_by_side"

    view.set_view_mode("split_slider")
    assert view.view_mode == "split_slider"
    view.set_split_position(0.7)
    assert view.split_position == 0.7
    view.close()


def test_canvas_view_wheel_zoom_clamping(qapp, sample_image_cv):
    view = MangaCanvasView()
    view.set_data(sample_image_cv)

    # Zoom in repeatedly
    for _ in range(30):
        evt = QWheelEvent(
            QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False
        )
        view.wheelEvent(evt)
    assert view.zoom_factor <= 10.0

    # Zoom out repeatedly
    for _ in range(60):
        evt = QWheelEvent(
            QPointF(100, 100), QPointF(100, 100), QPoint(0, 0), QPoint(0, -120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False
        )
        view.wheelEvent(evt)
    assert view.zoom_factor >= 0.05

    # Reset
    view.reset_zoom()
    assert view.zoom_factor == 1.0
    view.close()


def test_canvas_view_fit_in_view(qapp, sample_image_cv):
    view = MangaCanvasView()
    view.resize(800, 600)
    view.set_data(sample_image_cv)
    view.fit_in_view()
    assert view.zoom_factor > 0
    view.close()


# =============================================================================
# 5. Sidebar Components Tests
# =============================================================================

def test_natural_sort_key():
    files = ["p10.png", "p2.png", "p1.png", "p20.png", "p3.png"]
    sorted_files = sorted(files, key=natural_sort_key)
    assert sorted_files == ["p1.png", "p2.png", "p3.png", "p10.png", "p20.png"]


def test_page_list_add_and_natural_sort(qapp, tmp_path):
    panel = PageListWidget()

    # Create dummy files
    p1 = tmp_path / "page1.png"
    p2 = tmp_path / "page2.png"
    p10 = tmp_path / "page10.png"
    for p in [p1, p2, p10]:
        p.write_bytes(b"\x89PNG\r\n\x1a\n")

    # Pass in unsorted order [page10, page1, page2]
    panel.add_paths([str(p10), str(p1), str(p2)])

    assert len(panel.items_data) == 3
    assert panel.count_badge.text() == "3 页"
    # Should be sorted naturally: page1, page2, page10
    names = [os.path.basename(item["path"]) for item in panel.items_data]
    assert names == ["page1.png", "page2.png", "page10.png"]

    # Deduplication
    panel.add_paths([str(p1)])
    assert len(panel.items_data) == 3

    # Clear all
    panel.clear_all()
    assert len(panel.items_data) == 0
    assert panel.count_badge.text() == "0 页"
    panel.close()


def test_nav_rail_signals_and_collapse(qapp):
    rail = NavRail()
    signals = []
    rail.sig_nav_changed.connect(lambda k: signals.append(k))

    rail.set_active_section("settings")
    rail.set_collapsed_icon(True)
    rail.set_collapsed_icon(False)
    rail.close()


def test_drop_zone_creation(qapp):
    drop_zone = DropZoneWidget()
    assert drop_zone.acceptDrops()
    drop_zone.close()


# =============================================================================
# 6. Main Window Desktop Shell Tests
# =============================================================================

def test_main_window_assembly(qapp):
    win = MainWindow()
    assert win.minimumWidth() >= 1000
    assert win.minimumHeight() >= 650
    assert "AetherLens" in win.windowTitle()

    # Structural components
    assert win.toolbar_widget is not None
    assert win.splitter is not None
    assert len(win.splitter.sizes()) == 3
    assert win.queue_panel is not None
    assert win.canvas_view is not None
    assert win.inspector_panel is not None
    assert win.status_label.text() == "就绪"
    assert win.progress_bar.isHidden()

    # Toolbar controls
    assert len(win.btn_group.buttons()) == 5
    assert win.run_btn is not None
    assert win.settings_btn is not None
    assert win.bubble_cb.isChecked()

    win.close()


def test_main_window_view_mode_and_slider_bar(qapp):
    win = MainWindow()

    win._on_view_mode_changed("side_by_side")
    assert win.canvas_view.view_mode == "side_by_side"
    assert win.slider_bar.isHidden()

    win._on_view_mode_changed("split_slider")
    assert win.canvas_view.view_mode == "split_slider"
    assert not win.slider_bar.isHidden()

    win.split_slider.setValue(80)
    assert win.canvas_view.split_position == 0.8

    win._on_view_mode_changed("original")
    assert win.slider_bar.isHidden()
    win.close()


def test_main_window_zoom_label_update(qapp):
    win = MainWindow()
    win._on_zoom_changed(2.0)
    assert "200%" in win.status_label.text()
    win.close()


def test_main_window_theme_switching(qapp):
    win = MainWindow()
    win.set_theme("light")
    assert win._current_theme == "light"

    win.set_theme("dark")
    assert win._current_theme == "dark"
    win.close()


def test_folder_and_file_import_flow(qapp, tmp_path):
    # Create mock chapter folder with images and subfolders
    ch_dir = tmp_path / "chapter_01"
    ch_dir.mkdir()
    (ch_dir / "p01.png").write_text("fake_png")
    (ch_dir / "p02.jpg").write_text("fake_jpg")
    sub_dir = ch_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "p03.webp").write_text("fake_webp")
    (sub_dir / "ignore.txt").write_text("not_an_image")

    win = MainWindow()
    assert win.acceptDrops()

    # Test dropping directory path onto MainWindow
    win._on_paths_dropped([str(ch_dir)])
    assert len(win.page_list.items_data) == 3
    paths = [item["path"] for item in win.page_list.items_data]
    assert any("p01.png" in p for p in paths)
    assert any("p02.jpg" in p for p in paths)
    assert any("p03.webp" in p for p in paths)
    assert not any("ignore.txt" in p for p in paths)

    # DropZone widget has buttons
    assert win.drop_zone.btn_import_files is not None
    assert win.drop_zone.btn_import_folder is not None
    assert win.page_list.add_btn is not None
    win.close()

