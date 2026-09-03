"""
tests/unit/test_cache_manager.py
Unit tests for MangaCacheManager file-based disk caching, WebP compression,
blocks.json persistence, and lightweight metadata loading.
"""
import os
import shutil
import tempfile
import numpy as np
import pytest

from app.core.cache.cache_manager import MangaCacheManager, safe_cv2_imread, safe_cv2_imwrite
from app.core.models import TranslationBlock


@pytest.fixture
def temp_workspace():
    tmp_dir = tempfile.mkdtemp(prefix="amt_cache_test_")
    yield tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_safe_cv2_read_write(temp_workspace):
    img_path = os.path.join(temp_workspace, "测试页面_01.webp")
    test_img = np.zeros((150, 100, 3), dtype=np.uint8)
    test_img[10:30, 10:30] = [255, 128, 64]

    ok = safe_cv2_imwrite(img_path, test_img, ext=".webp", quality=90)
    assert ok is True
    assert os.path.exists(img_path)

    loaded = safe_cv2_imread(img_path)
    assert loaded is not None
    assert loaded.shape == (150, 100, 3)
    # Check pixel value roughly preserved under lossy/near-lossless
    assert abs(int(loaded[15, 15, 0]) - 255) < 10


def test_manga_cache_lifecycle(temp_workspace):
    mgr = MangaCacheManager()
    img_path = os.path.join(temp_workspace, "page_001.png")
    # Touch original file
    with open(img_path, "wb") as f:
        f.write(b"dummy image data")

    # Initial state: no cache
    has = mgr.has_cache(img_path)
    assert has["erased"] is False
    assert has["blocks"] is False
    assert has["rendered"] is False
    assert mgr.is_fully_translated(img_path) is False

    # 1. Save intermediate erased background & blocks
    erased = np.ones((200, 150, 3), dtype=np.uint8) * 255
    blocks = [
        TranslationBlock(id="b1", original_text="こんにちは", translated_text="你好", xmin=10.0, ymin=20.0),
        TranslationBlock(id="b2", original_text="さようなら", translated_text="再见", xmin=50.0, ymin=60.0),
    ]
    rendered = np.ones((200, 150, 3), dtype=np.uint8) * 128

    saved = mgr.save_page_cache(img_path, erased_img=erased, blocks=blocks, rendered_img=rendered)
    assert "erased" in saved
    assert "blocks" in saved
    assert "rendered" in saved

    # Check cache status
    has_after = mgr.has_cache(img_path)
    assert has_after["erased"] is True
    assert has_after["blocks"] is True
    assert has_after["rendered"] is True
    assert mgr.is_fully_translated(img_path) is True

    # 2. Test lightweight metadata loading (load_images=False)
    # This ensures page list views do NOT blow up RAM!
    meta_cache = mgr.load_page_cache(img_path, load_images=False)
    assert meta_cache["has_cache"] is True
    assert len(meta_cache["blocks"]) == 2
    assert meta_cache["blocks"][0].original_text == "こんにちは"
    assert meta_cache["blocks"][0].translated_text == "你好"
    assert meta_cache["erased_img"] is None    # Image not in RAM!
    assert meta_cache["rendered_img"] is None  # Image not in RAM!
    assert meta_cache["erased_path"] is not None
    assert meta_cache["rendered_path"] is not None

    # 3. Test on-demand full loading (load_images=True)
    full_cache = mgr.load_page_cache(img_path, load_images=True)
    assert full_cache["erased_img"] is not None
    assert full_cache["rendered_img"] is not None
    assert full_cache["erased_img"].shape == (200, 150, 3)

    # 4. Clear cache for page
    mgr.clear_cache(img_path)
    has_cleared = mgr.has_cache(img_path)
    assert has_cleared["erased"] is False
    assert has_cleared["blocks"] is False
    assert has_cleared["rendered"] is False