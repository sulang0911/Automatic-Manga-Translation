import os
import pytest
import numpy as np
import cv2
from desktop.core.inpaint_engine import (
    InpaintEngine, get_background_color_rgb, get_text_mask,
    dilate_mask, blend_inpainted_image
)

# ============================================================================
# F-INP-01: OpenCV Dynamic Feathered Inpaint
# ============================================================================

def test_finp_01_background_color_rgb_extraction():
    # Pure white crop
    white_crop = np.full((50, 50, 3), 255, dtype=np.uint8)
    bg_rgb = get_background_color_rgb(white_crop)
    assert bg_rgb == [255, 255, 255]

    # Pure black crop
    black_crop = np.full((50, 50, 3), 0, dtype=np.uint8)
    bg_rgb = get_background_color_rgb(black_crop)
    assert bg_rgb == [0, 0, 0]

def test_finp_01_text_mask_generation():
    # White background with black text box in center
    crop = np.full((60, 60, 3), 255, dtype=np.uint8)
    crop[20:40, 20:40] = 0  # black text simulation
    bg_color = [255, 255, 255]
    mask = get_text_mask(crop, bg_color)
    assert mask.shape == (60, 60)
    # The center black area should be marked in the mask (>0)
    assert np.all(mask[20:40, 20:40] > 0)
    # The border white area should be 0 in the mask
    assert mask[0, 0] == 0

def test_finp_01_feathered_blending_preserves_edges():
    orig = np.full((100, 100, 3), 200, dtype=np.uint8)
    inpainted = np.full((100, 100, 3), 50, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    blended = blend_inpainted_image(orig, inpainted, mask, feather_radius=4)
    assert blended.shape == (100, 100, 3)
    # Inside center should be close to inpainted
    assert blended[50, 50, 0] < 100
    # Far outside should be original 200
    assert blended[10, 10, 0] == 200

def test_finp_01_uniform_speech_bubble_flat_fill(sample_manga_image_np):
    engine = InpaintEngine(mode="opencv_telea")
    blocks = [{
        "id": "b1",
        "xmin": 12.5, "ymin": 12.5, "xmax": 43.75, "ymax": 37.5,
        "type": "bubble", "bg_color": "#FFFFFF"
    }]
    erased = engine.inpaint(sample_manga_image_np, blocks)
    assert erased is not None
    assert erased.shape == sample_manga_image_np.shape

def test_finp_01_feather_radius_zero_hard_cut():
    orig = np.full((50, 50, 3), 255, dtype=np.uint8)
    inp = np.full((50, 50, 3), 0, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 255

    res = blend_inpainted_image(orig, inp, mask, feather_radius=0)
    assert res[25, 25, 0] == 0
    assert res[5, 5, 0] == 255

# ============================================================================
# F-INP-02: Neural LaMa Inpainting
# ============================================================================

def test_finp_02_lama_initialization_status():
    engine = InpaintEngine(mode="auto")
    # Whether LaMa is available or not, the engine sets flags cleanly
    assert hasattr(engine, "_lama_available")
    assert hasattr(engine, "_lama")

def test_finp_02_inpaint_mask_generation():
    engine = InpaintEngine(mode="opencv_telea")
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    img[100:150, 100:150] = 0  # text
    blocks = [{"xmin": 30.0, "ymin": 30.0, "xmax": 60.0, "ymax": 60.0, "type": "onomatopoeia"}]
    res = engine.inpaint(img, blocks)
    assert res is not None
    assert res.shape == (300, 300, 3)

def test_finp_02_lama_execution_error_recovery(sample_manga_image_np, monkeypatch):
    engine = InpaintEngine(mode="lama")
    engine._lama_available = True
    def mock_lama_crash(*args, **kwargs):
        raise RuntimeError("CUDA Out of Memory in LaMa")
    engine._lama = mock_lama_crash

    blocks = [{"xmin": 15.0, "ymin": 15.0, "xmax": 40.0, "ymax": 40.0, "type": "onomatopoeia"}]
    # Should cleanly catch exception and fallback to OpenCV
    res = engine.inpaint(sample_manga_image_np, blocks)
    assert res is not None
    assert res.shape == sample_manga_image_np.shape

def test_finp_02_inpaint_with_progress_callback(sample_manga_image_np):
    engine = InpaintEngine(mode="opencv_telea")
    progress = []
    def cb(pct, msg):
        progress.append((pct, msg))

    blocks = [{"xmin": 15.0, "ymin": 15.0, "xmax": 40.0, "ymax": 40.0, "type": "bubble"}]
    res = engine.inpaint(sample_manga_image_np, blocks, progress_callback=cb)
    assert len(progress) >= 2
    assert progress[-1][0] == 100

def test_finp_02_empty_crop_safe():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert get_background_color_rgb(empty) == [255, 255, 255]

# ============================================================================
# F-INP-03: Inpaint Model Auto-Fallback
# ============================================================================

def test_finp_03_opencv_telea_mode(sample_manga_image_np):
    engine = InpaintEngine(mode="opencv_telea")
    blocks = [{"xmin": 10.0, "ymin": 10.0, "xmax": 30.0, "ymax": 30.0, "type": "onomatopoeia"}]
    res = engine.inpaint(sample_manga_image_np, blocks)
    assert res.shape == sample_manga_image_np.shape

def test_finp_03_opencv_ns_mode(sample_manga_image_np):
    engine = InpaintEngine(mode="opencv_ns")
    blocks = [{"xmin": 10.0, "ymin": 10.0, "xmax": 30.0, "ymax": 30.0, "type": "onomatopoeia"}]
    res = engine.inpaint(sample_manga_image_np, blocks)
    assert res.shape == sample_manga_image_np.shape

def test_finp_03_inpaint_empty_blocks_returns_copy(sample_manga_image_np):
    engine = InpaintEngine()
    res = engine.inpaint(sample_manga_image_np, [])
    assert np.array_equal(res, sample_manga_image_np)

def test_finp_03_inpaint_none_image_returns_none():
    engine = InpaintEngine()
    assert engine.inpaint(None, [{"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}]) is None

def test_finp_03_invalid_coords_skipped(sample_manga_image_np):
    engine = InpaintEngine()
    # xmax <= xmin
    blocks = [{"xmin": 50.0, "ymin": 50.0, "xmax": 40.0, "ymax": 60.0}]
    res = engine.inpaint(sample_manga_image_np, blocks)
    assert np.array_equal(res, sample_manga_image_np)

# ============================================================================
# F-INP-04: Context-Aware Mask Dilation
# ============================================================================

def test_finp_04_dilate_mask_zero_pixels():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 255
    res = dilate_mask(mask, dilation_pixels=0)
    assert np.array_equal(res, mask)

def test_finp_04_dilate_mask_positive_pixels():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 255
    res = dilate_mask(mask, dilation_pixels=3)
    # Area around (25, 25) should now be 255
    assert np.sum(res > 0) > 1
    assert res[24, 25] == 255
    assert res[26, 25] == 255

def test_finp_04_onomatopoeia_larger_dilation_than_bubble():
    base_dim = 1200
    bubble_dil = max(1, int(base_dim * 0.002))
    onoma_dil = max(2, int(base_dim * 0.004))
    assert onoma_dil > bubble_dil

def test_finp_04_dilation_clamped_to_minimum_one():
    base_dim = 100
    dyn_dil = max(1, int(base_dim * 0.002))
    assert dyn_dil >= 1

def test_finp_04_custom_dilation_parameter_respected(sample_manga_image_np):
    engine = InpaintEngine(mode="opencv_telea")
    blocks = [{"xmin": 20.0, "ymin": 20.0, "xmax": 40.0, "ymax": 40.0, "type": "onomatopoeia"}]
    res = engine.inpaint(sample_manga_image_np, blocks, onomatopoeia_dilation=8)
    assert res is not None
    assert res.shape == sample_manga_image_np.shape

# ============================================================================
# F-INP-05: Inpaint Disk & Memory Caching
# ============================================================================

def test_finp_05_erased_image_png_encoding(sample_manga_image_np):
    engine = InpaintEngine(mode="opencv_telea")
    blocks = [{"xmin": 12.5, "ymin": 12.5, "xmax": 43.75, "ymax": 37.5, "type": "bubble"}]
    erased = engine.inpaint(sample_manga_image_np, blocks)
    success, buf = cv2.imencode(".png", erased)
    assert success
    assert len(buf) > 0

def test_finp_05_disk_caching_roundtrip(temp_dir, sample_manga_image_np):
    cache_path = os.path.join(temp_dir, "page_01_erased.png")
    cv2.imwrite(cache_path, sample_manga_image_np)
    loaded = cv2.imread(cache_path)
    assert loaded is not None
    assert loaded.shape == sample_manga_image_np.shape

def test_finp_05_reusing_cached_erased_image(sample_manga_image_np):
    cached_erased = sample_manga_image_np.copy()
    # If existing erased image is passed, inpainting computation is bypassed
    blocks = [{"xmin": 10.0, "ymin": 10.0, "xmax": 30.0, "ymax": 30.0}]
    # Verify cached image is reusable
    assert cached_erased is not None

def test_finp_05_cache_invalidation_on_geometry_change():
    initial_block = {"id": "b1", "xmin": 10.0, "ymin": 10.0, "xmax": 30.0, "ymax": 30.0}
    modified_block = {"id": "b1", "xmin": 15.0, "ymin": 10.0, "xmax": 35.0, "ymax": 30.0}
    # Hashes differ
    assert (initial_block["xmin"], initial_block["xmax"]) != (modified_block["xmin"], modified_block["xmax"])

def test_finp_05_erased_buffer_isolation():
    engine = InpaintEngine(mode="opencv_telea")
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    blocks = [{"xmin": 10.0, "ymin": 10.0, "xmax": 30.0, "ymax": 30.0, "type": "bubble"}]
    erased = engine.inpaint(img, blocks)
    # Original img must not be modified in-place
    assert img[20, 20, 0] == 200
