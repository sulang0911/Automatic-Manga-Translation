"""
tests/unit/test_ocr_inpaint.py
Unit tests verifying OCR preprocessing, color analysis, inpainting pipelines, and hardware queries.
"""
import pytest
import numpy as np
import cv2

from app.core.models import TranslationBlock, BlockType, StyleConfig
from app.core.ocr.base import is_solid_color_page, merge_adjacent_boxes
from app.core.inpaint.color_analyzer import (
    get_background_color_rgb,
    get_background_color_hex,
    is_background_uniform,
    get_text_mask,
    analyze_text_color,
    dilate_mask
)
from app.core.inpaint.base import blend_inpainted_image
from app.core.inpaint.opencv_engine import OpenCVInpainter
from app.core.inpaint.lama_engine import LaMaInpainter
from app.core.hardware import (
    get_gpu_info,
    is_legacy_pascal_or_maxwell_gpu,
    is_vram_constrained,
    cleanup_gpu_memory
)


def test_solid_color_page_bypass():
    # Solid white blank page
    white_page = np.ones((800, 600, 3), dtype=np.uint8) * 255
    assert is_solid_color_page(white_page) is True

    # Solid black spacer page
    black_page = np.zeros((800, 600, 3), dtype=np.uint8)
    assert is_solid_color_page(black_page) is True

    # Page with content
    content_page = np.ones((800, 600, 3), dtype=np.uint8) * 255
    cv2.circle(content_page, (300, 400), 100, (0, 0, 0), -1)
    assert is_solid_color_page(content_page) is False


def test_merge_adjacent_boxes():
    # Two stacked lines in the same bubble
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 200, "ymax": 125, "text": "Line 1", "conf": 0.95},
        {"xmin": 105, "ymin": 130, "xmax": 195, "ymax": 155, "text": "Line 2", "conf": 0.90},
        # Distant bubble
        {"xmin": 400, "ymin": 500, "xmax": 500, "ymax": 540, "text": "Far line", "conf": 0.88},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 2
    # The first two should be merged
    assert "Line 1\nLine 2" in merged[0]["text"] or "Line 1\nLine 2" in merged[1]["text"]
    assert merged[0]["xmin"] <= 105
    assert merged[0]["ymax"] >= 155 or merged[1]["ymax"] >= 155


def test_color_analyzer_perimeter_and_uniformity():
    # 100x60 white crop with black text
    crop = np.ones((60, 100, 3), dtype=np.uint8) * 255
    cv2.putText(crop, "MANGA", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    bg_rgb = get_background_color_rgb(crop)
    assert bg_rgb == (255, 255, 255)
    assert get_background_color_hex(crop) == "#FFFFFF"
    assert is_background_uniform(crop) is True

    # Text mask: text strokes should be 255, borders should be 0
    mask = get_text_mask(crop, (255, 255, 255))
    assert mask[0, 0] == 0
    assert np.sum(mask == 255) > 40

    # Text stroke color
    text_color = analyze_text_color(crop, (255, 255, 255))
    assert text_color == "#000000"

    # Dilate mask expands text region
    dilated = dilate_mask(mask, dilation_pixels=2)
    assert np.sum(dilated == 255) > np.sum(mask == 255)


def test_color_analyzer_dark_inverted_bubble():
    # Dark bubble with white text
    crop = np.ones((60, 100, 3), dtype=np.uint8) * 15
    cv2.putText(crop, "DARK", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

    bg_rgb = get_background_color_rgb(crop)
    assert bg_rgb == (15, 15, 15)
    assert is_background_uniform(crop) is True

    text_color = analyze_text_color(crop, (15, 15, 15))
    assert text_color == "#FFFFFF" or text_color.upper() == "#F0F0F0"


def test_blend_inpainted_image_feathering():
    orig = np.zeros((100, 100, 3), dtype=np.uint8)
    inpainted = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255

    blended = blend_inpainted_image(orig, inpainted, mask, feather_radius=3)
    assert blended.shape == orig.shape
    # Center should be mostly 255, edges feathered
    assert blended[50, 50, 0] > 240
    assert blended[0, 0, 0] == 0


def test_opencv_inpainter_execution():
    img = np.ones((400, 300, 3), dtype=np.uint8) * 255
    cv2.circle(img, (150, 150), 60, (255, 255, 255), -1)
    cv2.putText(img, "TEST", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    block = TranslationBlock.from_pixel_box(
        xmin=100, ymin=100, xmax=200, ymax=200,
        img_width=300, img_height=400,
        original_text="TEST"
    )

    inpainter = OpenCVInpainter(method="telea")
    erased = inpainter.inpaint(img, [block])

    assert erased is not None
    assert erased.shape == img.shape
    # Center where text was should now be restored to white
    center_patch = erased[140:160, 130:170]
    assert np.mean(center_patch) > 240


def test_lama_inpainter_telea_fallback(monkeypatch):
    inpainter = LaMaInpainter(fallback_to_opencv=True)

    # Simulate CUDA OOM during neural inpainting
    def mock_lama_error(*args, **kwargs):
        raise RuntimeError("CUDA out of memory in test simulation")

    if inpainter._lama:
        monkeypatch.setattr(inpainter, "_lama", mock_lama_error)
    else:
        # Force lama available flag to test execution exception handling
        inpainter._lama_available = True
        inpainter._lama = mock_lama_error

    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, "OOM", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    b = TranslationBlock.from_pixel_box(40, 60, 160, 140, 200, 200, original_text="OOM")

    erased = inpainter.inpaint(img, [b])
    assert erased is not None
    assert erased.shape == img.shape


def test_hardware_gpu_detection():
    info = get_gpu_info()
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "device_name" in info
    assert "compute_capability" in info
    assert "is_legacy_gpu" in info

    # Cleanup call should execute without raising
    cleanup_gpu_memory()
