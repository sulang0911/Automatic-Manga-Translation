"""
tests/unit/test_deferred_rendering.py
Unit tests verifying the Deferred / Lazy Typography & Decoupled Export Architecture:
1. Zero UI freeze on global typography settings update.
2. Stale pre-rendered cache invalidation.
3. On-demand lazy rendering upon sidebar page selection.
4. BatchWorker skipping heavy raster baking when export_dir is None.
5. BatchWorker on-demand raster rendering during chapter export.
6. MainWindow export workflows.
"""
import os
import time
import shutil
import tempfile
import numpy as np
import cv2
import pytest
from PyQt6.QtWidgets import QApplication

from app.core.config import AppConfig, StyleConfig
from app.core.models import TranslationBlock
from app.core.cache.cache_manager import get_cache_manager
from app.core.pipeline.batch_worker import BatchWorker
from app.ui.main_window import MainWindow


@pytest.fixture
def sample_test_env():
    temp_dir = tempfile.mkdtemp(prefix="manga_deferred_test_")
    cache_mgr = get_cache_manager()

    # Create dummy images
    images = []
    for i in range(5):
        img_path = os.path.join(temp_dir, f"page_{i:02d}.png")
        blank = np.full((100, 100, 3), 240, dtype=np.uint8)
        cv2.imwrite(img_path, blank)
        images.append(img_path)

    yield temp_dir, images

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_re_render_all_pages_zero_freeze_and_invalidation(qapp, sample_test_env):
    """
    Verifies that _re_render_all_pages():
    1. Runs near-instantaneously (< 200ms) for multiple pages without freezing.
    2. Invalidates in-memory and disk rendered bitmaps while leaving erased & blocks intact.
    """
    temp_dir, images = sample_test_env
    cache_mgr = get_cache_manager()

    win = MainWindow()

    # Populate 5 pages with erased images, blocks, and old rendered images
    items = []
    for i, p in enumerate(images):
        block = TranslationBlock(id=f"b_{i}", original_text="こんにちは", translated_text="你好", xmin=10.0, ymin=10.0, xmax=80.0, ymax=80.0)
        erased = np.full((100, 100, 3), 255, dtype=np.uint8)
        old_rendered = np.full((100, 100, 3), 128, dtype=np.uint8)
        cache_mgr.save_page_cache(p, erased_img=erased, blocks=[block], rendered_img=old_rendered)

        # Check rendered cache exists
        cache_paths = cache_mgr.get_cache_paths(p)
        assert os.path.exists(cache_paths["rendered_webp"])

        item = {
            "id": str(i),
            "path": p,
            "filename": os.path.basename(p),
            "blocks": [block],
            "erased_img": erased,
            "translated_img": old_rendered,
            "style": None
        }
        items.append(item)

    win.page_list.items_data = items

    # Set page 0 as active
    win.current_image_data = items[0]
    win.canvas_view.original_cv = np.full((100, 100, 3), 240, dtype=np.uint8)
    win.canvas_view.translated_cv = items[0]["translated_img"]
    win.canvas_view.erased_cv = items[0]["erased_img"]

    # Change global font color to red
    win.config.style.font_color = "#FF0000"

    t0 = time.perf_counter()
    win._re_render_all_pages()
    duration = time.perf_counter() - t0

    # Must complete almost instantly (benchmark < 200ms)
    assert duration < 0.2, f"Expected < 200ms, took {duration*1000:.2f}ms"

    # Current page must have been re-rendered
    assert win.canvas_view.translated_cv is not None

    # Background pages (1 to 4) must have translated_img invalidated
    for item in items[1:]:
        assert item["translated_img"] is None
        paths = cache_mgr.get_cache_paths(item["path"])
        assert not os.path.exists(paths["rendered_webp"]), "Stale rendered_webp should have been deleted"
        # Erased & blocks must still be preserved!
        assert os.path.exists(paths["erased_webp"])
        assert os.path.exists(paths["blocks_json"])

    win.close()


def test_on_demand_rendering_on_page_select(qapp, sample_test_env):
    """
    Verifies that selecting an invalidated/unrendered page in the sidebar
    triggers fast on-demand typography rendering using current global style.
    """
    temp_dir, images = sample_test_env
    cache_mgr = get_cache_manager()

    p = images[0]
    block = TranslationBlock(id="b_0", original_text="テスト", translated_text="测试文本", xmin=10.0, ymin=10.0, xmax=80.0, ymax=80.0)
    erased = np.full((100, 100, 3), 255, dtype=np.uint8)
    cache_mgr.save_page_cache(p, erased_img=erased, blocks=[block])

    win = MainWindow()
    item_data = {
        "id": "item_0",
        "path": p,
        "filename": os.path.basename(p),
        "blocks": [block],
        "erased_img": erased,
        "translated_img": None
    }

    # Select page
    win._on_page_selected(item_data)

    # translated_img should have been generated on demand
    assert win.current_image_data["translated_img"] is not None
    assert win.canvas_view.translated_cv is not None

    # Cache should now have the newly rendered image
    paths = cache_mgr.get_cache_paths(p)
    assert os.path.exists(paths["rendered_webp"])

    win.close()


def test_batch_worker_deferred_typography_when_export_dir_none(sample_test_env):
    """
    Verifies that BatchWorker with export_dir=None does NOT render final image
    or write rendered cache, saving disk space & processing time.
    """
    temp_dir, images = sample_test_env
    cache_mgr = get_cache_manager()
    p = images[0]

    # Pre-save blocks with translation
    block = TranslationBlock(id="b_0", original_text="テスト", translated_text="测试", xmin=10.0, ymin=10.0, xmax=80.0, ymax=80.0)
    cache_mgr.save_page_cache(p, blocks=[block])

    items = [{"id": "0", "path": p}]
    worker = BatchWorker(items, config={"inpaint_engine": "opencv"}, export_dir=None)
    worker.run()

    # Cached erased_img and blocks must exist
    cache_paths = cache_mgr.get_cache_paths(p)
    assert os.path.exists(cache_paths["erased_webp"])
    assert os.path.exists(cache_paths["blocks_json"])

    # Rendered webp should NOT be baked when export_dir is None
    assert not os.path.exists(cache_paths["rendered_webp"])


def test_batch_worker_on_the_fly_export_from_cache(sample_test_env):
    """
    Verifies that BatchWorker with export_dir provided successfully resumes from
    cache (erased + blocks) and renders the exported file on the fly.
    """
    temp_dir, images = sample_test_env
    cache_mgr = get_cache_manager()
    p = images[0]

    block = TranslationBlock(id="b_0", original_text="テスト", translated_text="已翻译文字", xmin=10.0, ymin=10.0, xmax=80.0, ymax=80.0)
    erased = np.full((100, 100, 3), 255, dtype=np.uint8)
    cache_mgr.save_page_cache(p, erased_img=erased, blocks=[block])

    export_dir = os.path.join(temp_dir, "output")
    items = [{"id": "0", "path": p}]
    worker = BatchWorker(items, config={}, export_dir=export_dir)
    worker.run()

    expected_export = os.path.join(export_dir, "page_00.png")
    assert os.path.exists(expected_export), f"Exported file should exist at {expected_export}"

    exported_img = cv2.imread(expected_export)
    assert exported_img is not None
    assert exported_img.shape == (100, 100, 3)


def test_export_current_page_lazy_renders_if_missing(qapp, sample_test_env, monkeypatch):
    """
    Verifies that calling _export_current_page() triggers on-demand rendering
    if translated_cv is currently None but blocks exist.
    """
    temp_dir, images = sample_test_env
    cache_mgr = get_cache_manager()
    p = images[0]

    block = TranslationBlock(id="b_0", original_text="テスト", translated_text="已翻译文字", xmin=10.0, ymin=10.0, xmax=80.0, ymax=80.0)
    erased = np.full((100, 100, 3), 255, dtype=np.uint8)
    cache_mgr.save_page_cache(p, erased_img=erased, blocks=[block])

    win = MainWindow()
    win.current_image_data = {
        "id": "0",
        "path": p,
        "blocks": [block],
        "erased_img": erased,
        "translated_img": None
    }
    win.canvas_view.translated_cv = None
    win.canvas_view.original_cv = erased

    export_target = os.path.join(temp_dir, "test_out.png")
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (export_target, "PNG Image (*.png)")
    )

    win._export_current_page()

    # Must have lazily rendered
    assert win.canvas_view.translated_cv is not None
    assert os.path.exists(export_target)

    win.close()
