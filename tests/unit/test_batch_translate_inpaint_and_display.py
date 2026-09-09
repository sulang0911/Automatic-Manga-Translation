import os
import shutil
import tempfile
import numpy as np
import cv2
import pytest
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QMouseEvent

from app.core.config import AppConfig, StyleConfig
from app.core.models import TranslationBlock, OnomatopoeiaMode
from app.core.cache.cache_manager import get_cache_manager
from desktop.core.inpaint_engine import InpaintEngine
from app.core.inpaint.lama_engine import LaMaInpainter
from app.core.inpaint.opencv_engine import OpenCVInpainter
from app.ui.main_window import MainWindow
from app.ui.canvas.items.bubble_item import BubbleItem


@pytest.fixture
def temp_manga_env():
    temp_dir = tempfile.mkdtemp(prefix="manga_inpaint_test_")
    cache_mgr = get_cache_manager()

    images = []
    for i in range(3):
        p = os.path.join(temp_dir, f"page_{i:02d}.png")
        # White background with black text stroke
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        # Draw a vertical line of text-like pixels (simulating tall vertical Japanese characters)
        cv2.line(img, (50, 20), (50, 180), (0, 0, 0), 4)
        cv2.imwrite(p, img)
        images.append(p)

    yield temp_dir, images

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_inpaint_erases_extreme_aspect_vertical_block_by_default(temp_manga_env):
    """
    Verifies that a narrow vertical block (aspect ratio < 0.15) labeled as onomatopoeia
    is NOT skipped during inpainting when onomatopoeia_mode is 'normal' (default) or 'transparent'.
    """
    temp_dir, images = temp_manga_env
    p = images[0]
    img = cv2.imread(p)

    # Vertical text box: width 16px (8%), height 160px (80%) -> aspect = 0.10
    block = TranslationBlock(
        id="vert_1",
        original_text="あいつを許さない",
        translated_text="",
        xmin=20.0,
        ymin=10.0,
        xmax=28.0,
        ymax=90.0,
        type="onomatopoeia"
    )

    inpaint_eng = InpaintEngine(mode="opencv_telea")

    # Inpaint with default mode ('normal')
    erased = inpaint_eng.inpaint(img, [block], onomatopoeia_mode="normal")
    assert erased is not None

    # Verify that the black text line at (50, y) was erased (turned into background white)
    orig_center_lum = np.mean(img[30:170, 48:52])
    erased_center_lum = np.mean(erased[30:170, 48:52])
    assert orig_center_lum < 100, "Original image should contain black text strokes"
    assert erased_center_lum > 220, "Inpainted image must erase the vertical block text strokes!"


def test_inpaint_erases_when_block_has_translation_even_in_ignore_mode(temp_manga_env):
    """
    Verifies that if an onomatopoeia block already has translated_text,
    it is erased to prevent text overlapping with the original text.
    """
    temp_dir, images = temp_manga_env
    p = images[0]
    img = cv2.imread(p)

    block = TranslationBlock(
        id="vert_2",
        original_text="あいつを許さない",
        translated_text="绝不原谅他",
        xmin=20.0,
        ymin=10.0,
        xmax=28.0,
        ymax=90.0,
        type="onomatopoeia"
    )

    inpaint_eng = InpaintEngine(mode="opencv_telea")
    erased = inpaint_eng.inpaint(img, [block], onomatopoeia_mode="ignore")

    erased_center_lum = np.mean(erased[30:170, 48:52])
    assert erased_center_lum > 220, "Block with translated_text must be erased to prevent overlap"


def test_app_core_inpainters_respect_style_config(temp_manga_env):
    """
    Verifies that OpenCVInpainter and LaMaInpainter properly inspect style_config parameter.
    """
    temp_dir, images = temp_manga_env
    p = images[0]
    img = cv2.imread(p)

    block = TranslationBlock(
        id="vert_3",
        original_text="ドンッ",
        translated_text="咚！",
        xmin=20.0,
        ymin=10.0,
        xmax=28.0,
        ymax=90.0,
        type="onomatopoeia"
    )

    inpainter = OpenCVInpainter(method="telea")
    cfg = StyleConfig(onomatopoeia_mode="normal")
    erased = inpainter.inpaint(img, [block], style_config=cfg)
    assert erased is not None
    erased_center_lum = np.mean(erased[30:170, 48:52])
    assert erased_center_lum > 220


def test_on_page_selected_automatically_switches_to_translated_mode(qapp, temp_manga_env):
    """
    Verifies that when user selects a translated page in the list:
    1. If rendered image does not exist yet, it is rendered on demand.
    2. View mode is automatically switched from 'original' to 'translated'.
    3. The 'translated' mode button is checked.
    """
    temp_dir, images = temp_manga_env
    cache_mgr = get_cache_manager()

    p = images[0]
    block = TranslationBlock(
        id="b_1",
        original_text="こんにちは",
        translated_text="你好世界",
        xmin=10.0,
        ymin=10.0,
        xmax=80.0,
        ymax=80.0
    )
    erased = np.full((200, 200, 3), 255, dtype=np.uint8)
    cache_mgr.save_page_cache(p, erased_img=erased, blocks=[block])

    win = MainWindow()

    # Simulate canvas being in 'original' mode (e.g. as left by batch process)
    win.canvas_view.set_view_mode("original")
    if "original" in win._mode_buttons:
        win._mode_buttons["original"].setChecked(True)

    item_data = {
        "id": "page_0",
        "path": p,
        "filename": os.path.basename(p),
        "blocks": [block],
        "erased_img": erased,
        "translated_img": None
    }

    win._on_page_selected(item_data)

    # Verification:
    # 1. translated_img is generated on demand
    assert win.current_image_data["translated_img"] is not None
    assert win.canvas_view.translated_cv is not None
    # 2. View mode automatically switched to 'translated'
    assert win.canvas_view.view_mode == "translated"
    # 3. Mode button state updated
    if "translated" in win._mode_buttons:
        assert win._mode_buttons["translated"].isChecked()

    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_on_batch_finished_automatically_refreshes_canvas(qapp, temp_manga_env):
    """
    Verifies that MainWindow._on_batch_finished refreshes the active canvas page
    so the translation is displayed immediately without manual user intervention.
    """
    temp_dir, images = temp_manga_env
    cache_mgr = get_cache_manager()

    p = images[0]
    block = TranslationBlock(
        id="b_1",
        original_text="テスト",
        translated_text="测试成功",
        xmin=10.0,
        ymin=10.0,
        xmax=80.0,
        ymax=80.0
    )
    erased = np.full((200, 200, 3), 255, dtype=np.uint8)
    cache_mgr.save_page_cache(p, erased_img=erased, blocks=[block])

    win = MainWindow()
    win.canvas_view.set_view_mode("original")

    item_data = {
        "id": "page_0",
        "path": p,
        "filename": os.path.basename(p),
        "blocks": [block],
        "erased_img": erased,
        "translated_img": None
    }
    win.current_image_data = item_data

    # Call _on_batch_finished
    win._on_batch_finished(success_count=1, fail_count=0)

    # Canvas must now be in translated mode with translated image populated
    assert win.canvas_view.view_mode == "translated"
    assert win.canvas_view.translated_cv is not None

    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_bubble_click_does_not_emit_geometry_commit(qapp):
    """
    Verifies that simply clicking a BubbleItem without moving or resizing
    does NOT emit geometry_commit, avoiding redundant re-renders,
    while dragging DOES emit geometry_commit.
    """
    class MockMouseEvent:
        def __init__(self, pos: QPointF, button=Qt.MouseButton.LeftButton):
            self._pos = pos
            self._button = button
            self.accepted = False

        def button(self):
            return self._button

        def pos(self):
            return self._pos

        def scenePos(self):
            return self._pos

        def accept(self):
            self.accepted = True

    block_dict = {
        "id": "test_b",
        "xmin": 10.0,
        "ymin": 10.0,
        "xmax": 50.0,
        "ymax": 50.0,
        "translated_text": "气泡文字"
    }
    item = BubbleItem(block_dict, img_w=1000, img_h=1000)

    commit_emitted = []
    item.signals.geometry_commit.connect(lambda b: commit_emitted.append(b))

    # 1. Click without moving: should emit clicked, but NOT geometry_commit
    press_event = MockMouseEvent(QPointF(20, 20))
    item.mousePressEvent(press_event)
    assert press_event.accepted is True

    release_event = MockMouseEvent(QPointF(20, 20))
    item.mouseReleaseEvent(release_event)
    assert release_event.accepted is True

    assert len(commit_emitted) == 0, "Simply clicking a bubble should NOT emit geometry_commit"

    # 2. Dragging: should emit geometry_commit upon release
    item.mousePressEvent(MockMouseEvent(QPointF(20, 20)))
    item.mouseMoveEvent(MockMouseEvent(QPointF(60, 60)))
    item.mouseReleaseEvent(MockMouseEvent(QPointF(60, 60)))

    assert len(commit_emitted) == 1, "Dragging a bubble MUST emit geometry_commit"


def test_on_page_selected_inpaints_on_demand_when_erased_missing(qapp, temp_manga_env):
    """
    Verifies that if a page only has translations and blocks in cache, but erased_img
    is missing (e.g. from an interrupted batch run), MainWindow._on_page_selected
    automatically runs on-demand inpainting so the background is properly cleared
    before rendering typography, preventing text overlay on top of original Japanese text.
    """
    temp_dir, images = temp_manga_env
    cache_mgr = get_cache_manager()

    p = images[0]
    block = TranslationBlock(
        id="b_missing_erased",
        original_text="あいつを許さない",
        translated_text="绝不原谅他",
        xmin=20.0,
        ymin=10.0,
        xmax=28.0,
        ymax=90.0,
        type="bubble"
    )
    # Save cache with ONLY blocks (no erased_img, no rendered_img)
    cache_mgr.save_page_cache(p, blocks=[block])

    win = MainWindow()
    win.canvas_view.set_view_mode("original")

    item_data = {
        "id": "page_missing_erased",
        "path": p,
        "filename": os.path.basename(p),
        "blocks": [block],
        "erased_img": None,
        "translated_img": None
    }

    win._on_page_selected(item_data)

    # Verification:
    # 1. erased_img was automatically synthesized and saved to cache
    assert win.current_image_data["erased_img"] is not None
    assert win.canvas_view.erased_cv is not None
    cached = cache_mgr.load_page_cache(p, load_images=True)
    assert cached["erased_img"] is not None

    # 2. translated_img was rendered on top of the erased background
    assert win.current_image_data["translated_img"] is not None
    assert win.canvas_view.translated_cv is not None

    # 3. Canvas view mode automatically switched to 'translated'
    assert win.canvas_view.view_mode == "translated"

    win.close()
    win.deleteLater()
    qapp.processEvents()



