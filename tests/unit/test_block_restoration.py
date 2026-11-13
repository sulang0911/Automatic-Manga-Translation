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


def test_restore_expanded_unboxed_and_feathered_inpaint_halo():
    """Verifies that when padding > 0, the entire expanded unboxed & feathered halo is restored."""
    H, W = 300, 300
    orig = np.full((H, W, 3), [40, 80, 120], dtype=np.uint8)
    # Draw simulated text + extended unboxed halo up to 25px outside the tight OCR box (100..150, 100..180)
    erased = orig.copy()
    erased[72:178, 68:212] = [200, 200, 200]

    deleted_block = {
        "id": "b_unboxed",
        "xmin": (100.0 / W) * 100.0,
        "ymin": (100.0 / H) * 100.0,
        "xmax": (180.0 / W) * 100.0,
        "ymax": (150.0 / H) * 100.0,
        "type": "onomatopoeia"
    }

    restored = restore_block_pixels(
        original_img=orig,
        erased_img=erased,
        deleted_block=deleted_block,
        remaining_blocks=[],
        padding=4
    )
    assert np.array_equal(restored, orig)


def test_restore_moved_and_merged_blocks():
    """Verifies that deleting a moved (_orig_box) or merged (_merged_boxes) bubble restores all original erased areas."""
    H, W = 200, 200
    orig = np.full((H, W, 3), [10, 50, 90], dtype=np.uint8)
    erased = np.full((H, W, 3), [240, 240, 240], dtype=np.uint8)

    # Simulate a block originally at (10..30%, 10..30%) that was dragged to (60..80%, 60..80%)
    moved_block = {
        "id": "b_moved",
        "xmin": 60.0,
        "ymin": 60.0,
        "xmax": 80.0,
        "ymax": 80.0,
        "_orig_box": [10.0, 10.0, 30.0, 30.0]
    }
    restored = restore_block_pixels(orig, erased, moved_block, remaining_blocks=[], padding=0)
    assert np.array_equal(restored[20, 20], [10, 50, 90])
    assert np.array_equal(restored[140, 140], [10, 50, 90])


def test_translation_block_to_dict_omits_none_internal_metadata():
    """Ensures TranslationBlock.to_dict omits internal tracking keys when None and keeps them when populated."""
    tb = TranslationBlock.from_pixel_box(10, 20, 60, 80, 200, 400, id="m1")
    d = tb.to_dict()
    assert "_orig_box" not in d
    assert "_orig_polygon" not in d
    assert "_merged_boxes" not in d

    tb._orig_box = [5.0, 5.0, 30.0, 20.0]
    d2 = tb.to_dict()
    assert d2["_orig_box"] == [5.0, 5.0, 30.0, 20.0]


def test_restore_adjacent_rotated_polygon_no_aabb_overblock():
    """
    Verifies a remaining rotated/diagonal block only protects its actual rotated polygon
    and never blocks restoration across empty corners of its axis-aligned bounding box.
    """
    w, h = 800, 800
    orig = np.full((h, w, 3), 200, dtype=np.uint8)
    for y in range(0, h, 6):
        orig[y:y + 2, :] = (50, 80, 150)

    b_del = TranslationBlock.from_pixel_box(250, 200, 420, 260, w, h, id="del")
    poly_rot = [[260, 380], [460, 180], [500, 220], [300, 420]]
    b_rem = TranslationBlock.from_pixel_box(
        260, 180, 500, 420, w, h, id="rem", polygon=poly_rot, angle=-45.0
    )

    # Empty corner inside b_rem's AABB [260..500, 180..420] must not be masked by rem_core
    rem_mask = get_block_pixel_mask(b_rem, w, h, padding=0)
    assert rem_mask[220, 280] == 0
    assert rem_mask[250, 400] == 255

    erased = orig.copy()
    erased[195:265, 245:425] = 255
    cv2.fillPoly(erased, [np.array(poly_rot, dtype=np.int32)], (255, 255, 255))

    restored = restore_block_pixels(orig, erased, b_del, remaining_blocks=[b_rem], padding=4)
    # Inside b_del but outside poly_rot (and inside b_rem's AABB) must restore to original art
    assert np.array_equal(restored[220, 280], orig[220, 280])
    # Inside poly_rot itself must remain protected/erased
    assert np.array_equal(restored[250, 400], [255, 255, 255])


def test_restore_adjacent_inactive_border_no_voronoi_suppression():
    """
    Verifies a large nearby remaining block whose adjacent edge was untouched by inpainting
    does not suppress restoration of the deleted block's own erased halo in the gap.
    """
    w, h = 800, 600
    orig = np.full((h, w, 3), 180, dtype=np.uint8)
    for y in range(0, h, 8):
        orig[y:y + 3, :] = (60, 90, 140)

    b_del = TranslationBlock.from_pixel_box(100, 200, 300, 400, w, h, id="del")
    b_rem = TranslationBlock.from_pixel_box(312, 200, 700, 500, w, h, id="rem")

    erased = orig.copy()
    # b_del erased with +10px halo reaching x=310 (closer to b_rem x=312 than b_del x=300)
    erased[195:405, 95:310] = 245
    # b_rem only erased its interior (x=400..550), leaving its left border x=312..325 untouched
    erased[300:400, 400:550] = 250

    restored = restore_block_pixels(orig, erased, b_del, remaining_blocks=[b_rem], padding=4)
    assert np.array_equal(restored[280, 307], orig[280, 307])
    assert np.array_equal(restored[350, 450], [250, 250, 250])


def test_dragged_translation_block_restores_original_location_on_delete(qapp, tmp_path):
    """
    Verifies moving/dragging a TranslationBlock on canvas records _orig_box/_orig_polygon
    and restores the pre-drag original erased position when the block is deleted.
    """
    from app.core.models import StyleConfig
    from app.core.cache.cache_manager import safe_cv2_imwrite
    from app.ui.main_window import MainWindow

    img_path = str(tmp_path / "drag_test.png")
    w, h = 600, 800
    orig = np.full((h, w, 3), 220, dtype=np.uint8)
    for y in range(0, h, 8):
        orig[y:y + 3, :] = (65, 105, 185)
    safe_cv2_imwrite(img_path, orig)

    b1 = TranslationBlock.from_pixel_box(
        50, 100, 200, 160, w, h, id="b1",
        original_text="AAA", translated_text="甲",
        polygon=[[50, 100], [200, 100], [200, 160], [50, 160]],
    )
    b2 = TranslationBlock.from_pixel_box(50, 300, 200, 360, w, h, id="b2", original_text="BBB", translated_text="乙")
    erased = orig.copy()
    erased[95:165, 45:205] = 255
    erased[295:365, 45:205] = 255

    win = MainWindow()
    win.page_list.add_paths([img_path])
    win._on_page_selected(win.page_list.items_data[0])
    win._on_pipeline_finished({
        "mode": "full", "path": img_path, "original_img": orig.copy(),
        "erased_img": erased.copy(), "translated_img": erased.copy(),
        "blocks": [b1, b2], "page_style": StyleConfig(),
    })

    item_b1 = next(it for it in win.canvas_view.bubble_items if it.block_data.get("id") == "b1")
    assert item_b1.block_data.get("_orig_box") is not None
    assert item_b1.block_data.get("_orig_polygon") is not None

    # Simulate dragging b1 to a distant box
    item_b1.signals.geometry_start.emit(item_b1.block_data)
    item_b1.block_data["xmin"] = 50.0
    item_b1.block_data["ymin"] = 62.5
    item_b1.block_data["xmax"] = 75.0
    item_b1.block_data["ymax"] = 70.0
    item_b1.block_data["polygon"] = None
    item_b1.signals.changed.emit(item_b1.block_data)
    item_b1.signals.geometry_commit.emit(item_b1.block_data)

    win._on_block_deleted("b1")
    assert np.array_equal(win.current_image_data["erased_img"][100:160, 50:200], orig[100:160, 50:200])
    win.close()


def test_merge_blocks_with_none_metadata_and_delete(qapp, tmp_path):
    """
    Verifies merging blocks when _merged_boxes/_orig_box keys are None does not raise TypeError
    and deleting the merged block restores all original constituent erased regions.
    """
    from app.core.models import StyleConfig
    from app.core.cache.cache_manager import safe_cv2_imwrite
    from app.ui.main_window import MainWindow

    img_path = str(tmp_path / "merge_test.png")
    w, h = 600, 800
    orig = np.full((h, w, 3), 220, dtype=np.uint8)
    for y in range(0, h, 8):
        orig[y:y + 3, :] = (65, 105, 185)
    safe_cv2_imwrite(img_path, orig)

    b1 = TranslationBlock.from_pixel_box(50, 100, 200, 160, w, h, id="b1", original_text="A", translated_text="甲")
    b2 = TranslationBlock.from_pixel_box(50, 300, 200, 360, w, h, id="b2", original_text="B", translated_text="乙")
    # Also simulate legacy dict with explicit None tracking keys
    b1_dict = b1.to_dict()
    b1_dict["_merged_boxes"] = None
    b1_dict["_orig_box"] = None
    b2_dict = b2.to_dict()
    b2_dict["_merged_boxes"] = None
    b2_dict["_orig_box"] = None

    erased = orig.copy()
    erased[95:165, 45:205] = 255
    erased[295:365, 45:205] = 255

    win = MainWindow()
    win.page_list.add_paths([img_path])
    win._on_page_selected(win.page_list.items_data[0])
    win._on_pipeline_finished({
        "mode": "full", "path": img_path, "original_img": orig.copy(),
        "erased_img": erased.copy(), "translated_img": erased.copy(),
        "blocks": [b1_dict, b2_dict], "page_style": StyleConfig(),
    })

    win._on_canvas_bubble_merge_prev("b2")
    blocks_after_merge = win.current_image_data["blocks"]
    assert len(blocks_after_merge) == 1
    merged_id = blocks_after_merge[0]["id"] if isinstance(blocks_after_merge[0], dict) else blocks_after_merge[0].id

    win._on_block_deleted(merged_id)
    assert np.array_equal(win.current_image_data["erased_img"][100:160, 50:200], orig[100:160, 50:200])
    assert np.array_equal(win.current_image_data["erased_img"][300:360, 50:200], orig[300:360, 50:200])
    win.close()


def test_restore_overlapping_polygons_restores_non_overlapping_core():
    """Neighbor dilation/halo must not block pixels inside deleted_block's own non-overlapping core."""
    import cv2

    w, h = 600, 600
    orig = np.full((h, w, 3), 200, dtype=np.uint8)
    for y in range(0, h, 6):
        orig[y:y + 2, :] = 60

    p_del = [[100, 140], [260, 100], [300, 260], [140, 300]]
    p_rem = [[240, 100], [440, 100], [440, 300], [240, 300]]
    b_del = TranslationBlock.from_pixel_box(100, 100, 300, 300, w, h, id="b_del", polygon=p_del)
    b_rem = TranslationBlock.from_pixel_box(240, 100, 440, 300, w, h, id="b_rem", polygon=p_rem)

    m_del = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m_del, [np.array(p_del, dtype=np.int32)], 255)
    m_rem = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m_rem, [np.array(p_rem, dtype=np.int32)], 255)
    m_erased = cv2.dilate(cv2.bitwise_or(m_del, m_rem), np.ones((17, 17), np.uint8))

    erased = orig.copy()
    erased[m_erased > 0] = 255

    restored = restore_block_pixels(orig, erased, b_del, remaining_blocks=[b_rem], padding=4)
    pure_del = (m_del > 0) & (m_rem == 0)
    assert np.array_equal(restored[pure_del], orig[pure_del])
    assert np.array_equal(restored[m_rem > 0], erased[m_rem > 0])


def test_restore_block_pixels_channel_mismatch_safe():
    """Channel differences between original_img and erased_img must not crash restoration."""
    import cv2

    w, h = 200, 200
    orig_bgr = np.full((h, w, 3), 180, dtype=np.uint8)
    orig_bgr[60:120, 40:140] = (30, 60, 90)
    erased_bgra = cv2.cvtColor(np.full((h, w, 3), 255, dtype=np.uint8), cv2.COLOR_BGR2BGRA)
    b = TranslationBlock.from_pixel_box(40, 60, 140, 120, w, h, id="b1")

    restored = restore_block_pixels(orig_bgr, erased_bgra, b, remaining_blocks=[], padding=4)
    assert restored is not None
    assert restored.shape == (h, w, 4)
    assert np.array_equal(restored[60:120, 40:140, :3], orig_bgr[60:120, 40:140])


def test_canvas_delete_key_not_hijacked_by_inspector_shortcut(tmp_path):
    """Pressing Delete on CanvasView must delete the canvas-selected bubble rather than Inspector's stale selection."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from app.core.models import StyleConfig
    from app.core.cache.cache_manager import safe_cv2_imwrite
    from app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    img_path = str(tmp_path / "shortcut_del.png")
    w, h = 600, 800
    orig = np.full((h, w, 3), 210, dtype=np.uint8)
    orig[100:220, 80:240] = (40, 80, 140)
    orig[320:440, 80:240] = (80, 140, 40)
    safe_cv2_imwrite(img_path, orig)

    b0 = TranslationBlock.from_pixel_box(80, 100, 240, 220, w, h, id="b0", original_text="A", translated_text="甲")
    b1 = TranslationBlock.from_pixel_box(80, 320, 240, 440, w, h, id="b1", original_text="B", translated_text="乙")
    erased = orig.copy()
    erased[96:224, 76:244] = 255
    erased[316:444, 76:244] = 255

    win = MainWindow()
    win.show()
    app.processEvents()
    win.page_list.add_paths([img_path])
    win._on_page_selected(win.page_list.items_data[0])
    win._on_pipeline_finished({
        "mode": "full", "path": img_path, "original_img": orig.copy(),
        "erased_img": erased.copy(), "translated_img": erased.copy(),
        "blocks": [b0, b1], "page_style": StyleConfig(),
    })
    app.processEvents()

    # Desynchronize inspector state vs canvas selection, then press Delete on CanvasView
    win.inspector_panel.selected_block = None
    win.inspector_panel.bubble_list.clearSelection()
    win.canvas_view.select_bubble_by_id("b1")
    win.canvas_view.setFocus()
    app.processEvents()
    QTest.keyClick(win.canvas_view.viewport(), Qt.Key.Key_Delete)
    app.processEvents()

    rem_ids = [b["id"] if isinstance(b, dict) else b.id for b in win.current_image_data["blocks"]]
    assert rem_ids == ["b0"]
    assert np.array_equal(win.current_image_data["erased_img"][320:440, 80:240], orig[320:440, 80:240])
    win.close()
    app.processEvents()


def test_undo_manager_shared_edits_preserve_pre_delete_bitmap():
    """Pushing >max_bitmap_snapshots shared text edits must not evict an earlier pre-delete erased_img."""
    mgr = UndoManager(max_depth=10, max_bitmap_snapshots=4)
    erased_before = np.full((64, 64, 3), 255, dtype=np.uint8)
    erased_after = np.full((64, 64, 3), 120, dtype=np.uint8)

    mgr.push(PageSnapshot.create("p.png", [{"id": "b0"}], erased_img=erased_before, copy_erased=True, description="删除气泡"))
    for i in range(6):
        mgr.push(PageSnapshot.create("p.png", [{"id": "b1", "translated_text": f"t{i}"}], erased_img=erased_after, copy_erased=False, description=f"edit_{i}"))

    assert mgr._undo_stack[0].erased_img is not None
    assert np.array_equal(mgr._undo_stack[0].erased_img, erased_before)
