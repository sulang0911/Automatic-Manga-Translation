"""
tests/unit/test_block_restoration.py
Comprehensive unit tests for block pixel restoration upon deletion,
including active block collision protection and undo/redo snapshot safety.
"""
import pytest
import numpy as np
import cv2

from app.core.models import TranslationBlock
from app.core.inpaint.restore_helper import get_block_pixel_mask, restore_block_pixels
from app.core.undo_manager import PageSnapshot, UndoManager


def test_get_block_pixel_mask_box():
    """Verifies that bounding box coordinates produce expected binary mask."""
    block = {
        "id": "b1",
        "xmin": 10.0,
        "ymin": 20.0,
        "xmax": 50.0,
        "ymax": 60.0
    }
    mask = get_block_pixel_mask(block, img_w=100, img_h=100, padding=0)
    assert mask.shape == (100, 100)
    # Inside box (e.g. x=30, y=40) must be 255
    assert mask[40, 30] == 255
    # Outside box (e.g. x=5, y=5) must be 0
    assert mask[5, 5] == 0
    assert mask[80, 80] == 0


def test_get_block_pixel_mask_polygon():
    """Verifies that polygon vertices produce accurate mask."""
    poly = [[10, 10], [50, 10], [50, 50], [10, 50]]
    block = {
        "id": "b_poly",
        "polygon": poly
    }
    mask = get_block_pixel_mask(block, img_w=100, img_h=100, padding=0)
    assert mask[30, 30] == 255
    assert mask[5, 5] == 0
    assert mask[70, 70] == 0


def test_get_block_pixel_mask_padding():
    """Verifies that padding expands the mask via morphological dilation."""
    block = {
        "id": "b1",
        "xmin": 20.0,
        "ymin": 20.0,
        "xmax": 40.0,
        "ymax": 40.0
    }
    mask_no_pad = get_block_pixel_mask(block, img_w=100, img_h=100, padding=0)
    mask_padded = get_block_pixel_mask(block, img_w=100, img_h=100, padding=4)
    assert np.sum(mask_padded) > np.sum(mask_no_pad)
    # Just outside the box boundary (x=18, y=30) should be included in padded mask
    assert mask_padded[30, 18] == 255
    assert mask_no_pad[30, 18] == 0


def test_restore_single_block_pixels():
    """Tests that deleting a block restores original image pixels inside the block."""
    H, W = 100, 100
    # Original image: distinct color pattern (e.g. blue=[255, 0, 0])
    original_img = np.full((H, W, 3), [255, 0, 0], dtype=np.uint8)
    # Erased image: white inpainted background (e.g. [255, 255, 255])
    erased_img = np.full((H, W, 3), [255, 255, 255], dtype=np.uint8)

    deleted_block = {
        "id": "del_1",
        "xmin": 10.0,
        "ymin": 10.0,
        "xmax": 40.0,
        "ymax": 40.0
    }

    restored = restore_block_pixels(
        original_img=original_img,
        erased_img=erased_img,
        deleted_block=deleted_block,
        remaining_blocks=[],
        padding=0
    )

    # Inside deleted block: must be restored to original blue
    assert np.array_equal(restored[25, 25], [255, 0, 0])
    # Outside deleted block: must remain white (inpainted)
    assert np.array_equal(restored[80, 80], [255, 255, 255])


def test_restore_overlapping_blocks_collision_protection():
    """
    Critical test: If Block A (deleted) overlaps with Block B (active),
    Block B's region must NOT be overwritten by original image pixels!
    """
    H, W = 100, 100
    # Original image: red [0, 0, 255]
    original_img = np.full((H, W, 3), [0, 0, 255], dtype=np.uint8)
    # Erased image: green [0, 255, 0]
    erased_img = np.full((H, W, 3), [0, 255, 0], dtype=np.uint8)

    # Block A: x from 10 to 50, y from 10 to 50
    block_a = {"id": "A", "xmin": 10.0, "ymin": 10.0, "xmax": 50.0, "ymax": 50.0}
    # Block B (active): overlaps with A, x from 30 to 70, y from 10 to 50
    block_b = {"id": "B", "xmin": 30.0, "ymin": 10.0, "xmax": 70.0, "ymax": 50.0}

    # Delete block A while keeping block B active
    restored = restore_block_pixels(
        original_img=original_img,
        erased_img=erased_img,
        deleted_block=block_a,
        remaining_blocks=[block_b],
        padding=0
    )

    # Region unique to A (x=20, y=30): MUST be restored to original red [0, 0, 255]
    assert np.array_equal(restored[30, 20], [0, 0, 255])

    # Overlapping region (x=40, y=30): MUST STAY erased green [0, 255, 0] because B is active!
    assert np.array_equal(restored[30, 40], [0, 255, 0])

    # Region unique to B (x=60, y=30): MUST STAY erased green [0, 255, 0]
    assert np.array_equal(restored[30, 60], [0, 255, 0])

    # Region outside both (x=85, y=85): stays green [0, 255, 0]
    assert np.array_equal(restored[85, 85], [0, 255, 0])


def test_restore_all_blocks_empty():
    """Tests that deleting all blocks cleanly restores the full original image."""
    H, W = 50, 50
    orig = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    erased = np.zeros((H, W, 3), dtype=np.uint8)

    restored = restore_block_pixels(
        original_img=orig,
        erased_img=erased,
        deleted_block=None,
        remaining_blocks=[]
    )

    assert np.array_equal(restored, orig)


def test_undo_manager_snapshot_erased_img_immutability():
    """
    Verifies that PageSnapshot.create performs a defensive copy of erased_img,
    ensuring that modifying erased_img afterwards does not corrupt the undo history.
    """
    erased = np.full((50, 50, 3), 100, dtype=np.uint8)
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30}]

    snap = PageSnapshot.create(
        page_path="test.jpg",
        blocks=blocks,
        erased_img=erased,
        description="测试快照"
    )

    # Mutate the original erased array
    erased[0, 0] = [255, 0, 0]

    # Snapshot's erased_img must remain unmodified (100)
    assert np.array_equal(snap.erased_img[0, 0], [100, 100, 100])


def test_main_window_block_deletion_and_undo_restoration(qapp):
    """
    Tests full UI flow:
    1. MainWindow loads an image with erased_img and a block.
    2. Deleting block restores original_cv pixels in erased_img.
    3. Undo restores the erased_img back to erased state.
    4. Redo restores original_cv pixels again.
    """
    from app.ui.main_window import MainWindow

    H, W = 100, 100
    orig_img = np.full((H, W, 3), [10, 20, 30], dtype=np.uint8)
    eras_img = np.full((H, W, 3), [250, 250, 250], dtype=np.uint8)

    win = MainWindow()
    win.current_image_data = {
        "id": "img_restore_test",
        "path": "dummy_test.png",
        "blocks": [
            {
                "id": "b_del",
                "xmin": 20.0,
                "ymin": 20.0,
                "xmax": 60.0,
                "ymax": 60.0,
                "original_text": "テスト",
                "translated_text": "测试",
                "type": "bubble"
            }
        ],
        "erased_img": eras_img.copy(),
        "translated_img": eras_img.copy()
    }
    win.canvas_view.original_cv = orig_img.copy()
    win.canvas_view.erased_cv = eras_img.copy()
    win.canvas_view.blocks = win.current_image_data["blocks"]

    # Point inside the block
    pt = (40, 40)

    # Initially, erased_cv has [250, 250, 250]
    assert np.array_equal(win.current_image_data["erased_img"][pt], [250, 250, 250])

    # 1. Delete the block
    win._on_block_deleted("b_del")

    # Assert block is deleted from data list
    assert len(win.current_image_data["blocks"]) == 0

    # Assert pixels in erased_img are restored to original_cv [10, 20, 30]
    assert np.array_equal(win.current_image_data["erased_img"][pt], [10, 20, 30])
    assert np.array_equal(win.canvas_view.erased_cv[pt], [10, 20, 30])

    # 2. Undo the deletion
    win._undo()
    assert len(win.current_image_data["blocks"]) == 1
    assert win.current_image_data["blocks"][0]["id"] == "b_del"
    # Erased image must be restored back to erased state [250, 250, 250]
    assert np.array_equal(win.current_image_data["erased_img"][pt], [250, 250, 250])

    # 3. Redo the deletion
    win._redo()
    assert len(win.current_image_data["blocks"]) == 0
    # Erased image must be restored back to original pixels [10, 20, 30]
    assert np.array_equal(win.current_image_data["erased_img"][pt], [10, 20, 30])

    win.close()


def test_disk_file_cache_persistence_on_block_deleted(qapp, tmp_path):
    """
    Tests that deleting a block actually writes and updates the LOCAL DISK FILES:
    - .amt_cache/*.erased.webp
    - .amt_cache/*.blocks.json
    - .amt_cache/*.rendered.webp
    And survives complete in-memory destruction (re-loading from disk cache).
    """
    import os
    import json
    from app.ui.main_window import MainWindow
    from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imwrite, safe_cv2_imread

    # Create real dummy image on disk
    img_path = str(tmp_path / "page_disk_test.png")
    H, W = 100, 100
    orig_img = np.full((H, W, 3), [15, 30, 45], dtype=np.uint8)
    safe_cv2_imwrite(img_path, orig_img)

    eras_img = np.full((H, W, 3), [220, 220, 220], dtype=np.uint8)
    initial_blocks = [
        {
            "id": "b_disk",
            "xmin": 20.0,
            "ymin": 20.0,
            "xmax": 60.0,
            "ymax": 60.0,
            "original_text": "原文",
            "translated_text": "译文",
            "type": "bubble"
        }
    ]

    # Save initial cache to local disk
    cache_mgr = get_cache_manager()
    paths = cache_mgr.save_page_cache(
        img_path,
        erased_img=eras_img,
        blocks=initial_blocks,
        rendered_img=eras_img
    )

    # Verify initial disk files exist
    assert os.path.exists(paths["erased"])
    assert os.path.exists(paths["blocks"])
    assert os.path.exists(paths["rendered"])

    # Verify initial disk file has erased color [220, 220, 220]
    initial_disk_erased = safe_cv2_imread(paths["erased"])
    assert np.array_equal(initial_disk_erased[40, 40], [220, 220, 220])

    # Open MainWindow and load this item
    win = MainWindow()
    win.current_image_data = {
        "id": "item_1",
        "path": img_path,
        "blocks": initial_blocks.copy(),
        "erased_img": eras_img.copy(),
        "translated_img": eras_img.copy()
    }
    win.canvas_view.original_cv = orig_img.copy()
    win.canvas_view.erased_cv = eras_img.copy()
    win.canvas_view.blocks = win.current_image_data["blocks"]

    # Delete the block
    win._on_block_deleted("b_disk")
    win.close()

    # Now read the DISK FILES directly from the filesystem!
    updated_disk_erased = safe_cv2_imread(paths["erased"])
    assert updated_disk_erased is not None

    # The pixel inside the deleted block on the DISK FILE must be restored to original [15, 30, 45]!
    # Note: WebP quality=95 has small DCT quantization tolerance (atol=2)
    assert np.all(np.abs(updated_disk_erased[40, 40].astype(int) - np.array([15, 30, 45])) <= 2)

    # Check the JSON file on disk
    with open(paths["blocks"], "r", encoding="utf-8") as f:
        disk_blocks = json.load(f)
    assert len(disk_blocks) == 0

    # Test complete reload from disk cache (no in-memory references)
    loaded_cache = cache_mgr.load_page_cache(img_path, load_images=True)
    assert loaded_cache["blocks"] == []
    assert np.all(np.abs(loaded_cache["erased_img"][40, 40].astype(int) - np.array([15, 30, 45])) <= 2)


