"""
tests/unit/test_unboxed_text_eraser.py
Unit tests for the unboxed text erasure algorithm.
Verifies clean erasure of chapter titles, margin texts, onomatopoeia without explicit speech bubbles,
panel frame boundary protection, and dual-mode flat-fill vs inpaint mask generation.
"""
import pytest
import numpy as np
import cv2

from app.core.inpaint.unboxed_text_eraser import (
    is_unboxed_text_block,
    erase_unboxed_text_block,
)
from app.core.models import TranslationBlock


def test_is_unboxed_text_block():
    assert is_unboxed_text_block({"type": "onomatopoeia"}) is True
    assert is_unboxed_text_block({"type": "other"}) is True
    assert is_unboxed_text_block({"type": "text"}) is True
    assert is_unboxed_text_block({"type": "bubble"}) is False
    assert is_unboxed_text_block({"type": "bubble", "is_unboxed": True}) is True

    # TranslationBlock model objects
    tb_onoma = TranslationBlock(id="t1", original_text="Title", type="onomatopoeia")
    assert is_unboxed_text_block(tb_onoma) is True

    tb_bubble = TranslationBlock(id="t2", original_text="Hi", type="bubble")
    assert is_unboxed_text_block(tb_bubble) is False


def test_erase_unboxed_chapter_title_uniform_margin():
    """
    Simulates a chapter title on a white margin with a nearby dark panel frame border.
    Verifies that the text is completely flat-filled with clean white,
    leaving no black smudges, and that the panel border is untouched.
    """
    canvas = np.ones((200, 500, 3), dtype=np.uint8) * 255
    # Dark panel border at y=140 to 143 across entire width
    canvas[140:144, 20:480] = 0

    # Draw dark chapter title text at y=80..115, x=50..350
    cv2.putText(canvas, "Chapter 1: The Beginning", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

    block = {
        "id": "b_title",
        "type": "onomatopoeia",
        "original_text": "Chapter 1: The Beginning",
        "polygon": [[50, 85], [350, 85], [350, 115], [50, 115]],
    }

    modified_img, mask_contrib = erase_unboxed_text_block(canvas, block)

    # For uniform white margin, inpaint_mask should be None (flat-fill executed)
    assert mask_contrib is None

    # Text area should now be restored to pure white (no dark pixels)
    text_area = modified_img[80:120, 45:360]
    gray_text_area = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.count_nonzero(gray_text_area < 200)
    assert dark_pixels == 0, f"Expected 0 dark pixels after erasure, found {dark_pixels}"

    # Panel frame border at y=140:144 must be completely preserved
    border_area = modified_img[140:144, 50:350]
    assert np.all(border_area == 0), "Panel frame border must not be damaged or eroded!"


def test_erase_unboxed_text_textured_background_yields_mask():
    """
    Verifies that unboxed text on a complex/textured background produces an expanded,
    properly dilated inpaint mask for neural/diffusion inpainting instead of flat-filling.
    """
    # Create random texture background
    np.random.seed(42)
    textured = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    # Add text
    cv2.putText(textured, "SFX", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    block = {
        "id": "b_sfx",
        "type": "onomatopoeia",
        "original_text": "SFX",
        "polygon": [[50, 70], [130, 70], [130, 110], [50, 110]],
    }

    modified_img, mask_contrib = erase_unboxed_text_block(textured, block)

    # Background is noisy/textured -> mask_contrib must be provided
    assert mask_contrib is not None
    assert mask_contrib.shape == (200, 200)
    assert np.sum(mask_contrib) > 0

    # The mask must cover the text area
    assert np.sum(mask_contrib[70:110, 50:130]) > 0


def test_erase_unboxed_text_dark_inverted_background():
    """
    Verifies that white unboxed text on a solid dark banner/background is flat-filled cleanly.
    """
    canvas = np.zeros((150, 300, 3), dtype=np.uint8)
    cv2.putText(canvas, "DARK TITLE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    block = {
        "id": "b_dark",
        "type": "onomatopoeia",
        "bg_color": "#000000",
        "polygon": [[30, 50], [250, 50], [250, 90], [30, 90]],
    }

    modified_img, mask_contrib = erase_unboxed_text_block(canvas, block)
    assert mask_contrib is None

    # Text region should be restored to dark
    text_area = modified_img[50:90, 30:250]
    gray_area = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.count_nonzero(gray_area > 50)
    assert bright_pixels == 0, f"Expected 0 bright pixels after dark flat-fill, found {bright_pixels}"


def test_erase_unboxed_text_respects_qr_mask():
    """
    Verifies that QR code regions are protected and never altered by unboxed text erasure.
    """
    canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "NOTE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # QR protection mask covering part of the region
    qr_mask = np.zeros((200, 200), dtype=np.uint8)
    qr_mask[60:90, 60:90] = 255
    canvas[60:90, 60:90] = 128  # Simulated QR pattern

    block = {
        "id": "b_note",
        "type": "onomatopoeia",
        "polygon": [[15, 60], [100, 60], [100, 90], [15, 90]],
    }

    modified_img, _ = erase_unboxed_text_block(canvas, block, qr_mask=qr_mask)
    # The QR region must be unchanged
    assert np.all(modified_img[60:90, 60:90] == 128)


def test_erase_unboxed_text_empty_and_invalid_inputs():
    """
    Verifies boundary safety with empty or degenerate inputs.
    """
    res_img, mask = erase_unboxed_text_block(None, {"type": "onomatopoeia"})
    assert res_img is None
    assert mask is None

    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    res_img, mask = erase_unboxed_text_block(empty_img, {"type": "onomatopoeia"})
    assert res_img.size == 0
    assert mask is None

    valid_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Degenerate zero-size polygon
    res_img, mask = erase_unboxed_text_block(valid_img, {"polygon": [[10, 10], [10, 10], [10, 10]]})
    assert mask is None


def test_erase_unboxed_text_dark_inverted_tight_box():
    """
    Verifies that white unboxed text on a dark background with a TIGHT OCR bounding box
    is completely and cleanly flat-filled (0 residual non-dark pixels).
    """
    canvas = np.full((120, 250, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, "TITLE", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)

    # Tight OCR bounding box
    block = {
        "id": "b_dark_tight",
        "type": "onomatopoeia",
        "bg_color": "#141414",
        "polygon": [[30, 42], [145, 42], [145, 68], [30, 68]],
    }

    modified_img, mask_contrib = erase_unboxed_text_block(canvas, block)
    assert mask_contrib is None

    # Text area should be restored to dark
    text_area = modified_img[35:75, 25:155]
    gray_area = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.count_nonzero(gray_area > 50)
    assert bright_pixels == 0, f"Expected 0 bright pixels after dark flat-fill, found {bright_pixels}"


def test_erase_unboxed_vertical_japanese_title():
    """
    Verifies that a vertical unboxed title in the page margin expands horizontally
    according to character scale and erases all glyph strokes cleanly.
    """
    canvas = np.ones((300, 150, 3), dtype=np.uint8) * 255
    # Draw vertical line simulating vertical text in margin
    cv2.line(canvas, (110, 30), (110, 200), (30, 30, 30), 4)

    # Tight OCR bounding box for vertical text
    block = {
        "id": "b_vert",
        "type": "onomatopoeia",
        "polygon": [[106, 30], [114, 30], [114, 200], [106, 200]],
    }

    modified_img, mask_contrib = erase_unboxed_text_block(canvas, block)
    assert mask_contrib is None

    text_area = modified_img[25:205, 100:120]
    gray_area = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.count_nonzero(gray_area < 200)
    assert dark_pixels == 0, f"Expected 0 dark pixels, found {dark_pixels}"


def test_is_unboxed_text_block_with_image_margin_detection():
    """
    Verifies that text with type='bubble' is recognized as unboxed when positioned
    in the outer page margin with uniform background.
    """
    # 1000x700 image with top margin
    canvas = np.ones((1000, 700, 3), dtype=np.uint8) * 255
    block_in_margin = {
        "id": "b_margin_title",
        "type": "bubble",  # Defaulted to bubble by OCR
        "polygon": [[100, 50], [500, 50], [500, 80], [100, 80]],
    }
    # Without image context: returns False because type is 'bubble'
    assert is_unboxed_text_block(block_in_margin) is False

    # With image context: recognizes it is in the top margin with uniform background
    assert is_unboxed_text_block(block_in_margin, canvas) is True


def test_erase_target_user_uploaded_image():
    """
    Verifies unboxed text erasure directly on the user-provided target image:
    1. Chapter title (Gachidom world...) dark pixels in top margin are completely erased.
    2. Panel frame border below the title (y=133..135) is 100% intact.
    """
    import os, json
    img_path = r"C:/Users/25362/.gemini/antigravity/brain/a0089734-515a-4735-8452-0e3422010ffc/.user_uploaded/media_1788757522370.jpg"
    if not os.path.exists(img_path):
        pytest.skip("Target user image not present in environment")

    orig_img = cv2.imread(img_path)
    with open("user_img_blocks.json", encoding="utf-8") as f:
        blocks = json.load(f)

    b0 = blocks[0]  # Chapter title
    modified_img, mask_contrib = erase_unboxed_text_block(orig_img, b0)
    assert mask_contrib is None

    # Verify title zone has 0 residual dark pixels
    title_zone = modified_img[95:132, 70:485]
    gray_title = cv2.cvtColor(title_zone, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.count_nonzero(gray_title < 180)
    assert dark_pixels == 0, f"Expected 0 residual dark pixels in title zone, found {dark_pixels}"

    # Verify panel frame border at y=133..136 is 100% intact
    border_orig = orig_img[133:136, 70:485]
    border_modified = modified_img[133:136, 70:485]
    diff = cv2.absdiff(border_orig, border_modified)
    assert diff.max() == 0, f"Panel frame border must not be modified, max diff={diff.max()}"

