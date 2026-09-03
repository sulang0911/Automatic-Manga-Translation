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


def test_merge_user_sample_vertically_adjacent_independent_bubbles():
    """
    Verifies that two vertically adjacent independent rectangular dialogue boxes
    (from the user problem sample) are NOT erroneously merged into one large box.
    """
    from desktop.core.ocr_engine import OCREngine
    desktop_engine = OCREngine(use_gpu=False)

    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 450, "ymax": 135, "text": "It's always good to pick one that loves dogs.", "conf": 0.95},
        {"xmin": 120, "ymin": 165, "xmax": 340, "ymax": 195, "text": "Their mind last longer.", "conf": 0.92},
    ]

    # Test app core
    merged_app = merge_adjacent_boxes(boxes, img_w=1000, img_h=1200)
    assert len(merged_app) == 2
    assert merged_app[0]["text"] == "It's always good to pick one that loves dogs."
    assert merged_app[1]["text"] == "Their mind last longer."
    assert merged_app[0]["line_count"] == 1
    assert merged_app[1]["line_count"] == 1

    # Test desktop engine consistency
    merged_desk = desktop_engine._merge_adjacent_boxes(boxes, w_img=1000, h_img=1200)
    assert len(merged_desk) == 2
    assert merged_desk[0]["text"] == merged_app[0]["text"]
    assert merged_desk[1]["text"] == merged_app[1]["text"]


def test_merge_user_sample_horizontally_adjacent_independent_bubbles():
    """
    Verifies that two horizontally adjacent independent rectangular dialogue boxes
    are NOT erroneously merged horizontally.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 420, "ymax": 135, "text": "It's always good to pick one that loves dogs.", "conf": 0.95},
        {"xmin": 450, "ymin": 105, "xmax": 660, "ymax": 138, "text": "Their mind last longer.", "conf": 0.92},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 2
    assert merged[0]["text"] == "It's always good to pick one that loves dogs."
    assert merged[1]["text"] == "Their mind last longer."


def test_merge_user_sample_staggered_independent_bubbles():
    """
    Verifies that two staggered / misaligned dialogue boxes are isolated.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 130, "text": "It's always good to pick one that loves dogs.", "conf": 0.95},
        {"xmin": 280, "ymin": 140, "xmax": 480, "ymax": 170, "text": "Their mind last longer.", "conf": 0.92},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 2
    assert merged[0]["text"] == "It's always good to pick one that loves dogs."
    assert merged[1]["text"] == "Their mind last longer."


def test_merge_single_bubble_multiline_horizontal():
    """
    Verifies that multi-line text (>= 2 lines) within a single bubble is correctly aggregated
    with newline character separator and updated line_count.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 350, "ymax": 125, "text": "It's always good to pick", "conf": 0.96},
        {"xmin": 105, "ymin": 132, "xmax": 345, "ymax": 157, "text": "one that loves dogs.", "conf": 0.94},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "It's always good to pick\none that loves dogs."
    assert merged[0]["line_count"] == 2
    assert merged[0]["xmin"] == 100
    assert merged[0]["ymin"] == 100
    assert merged[0]["xmax"] == 350
    assert merged[0]["ymax"] == 157
    assert 0.94 <= merged[0]["conf"] <= 0.96


def test_merge_multiline_bubble_with_nearby_independent_bubble():
    """
    Verifies that a multi-line bubble correctly aggregates its own lines
    while keeping an adjacent independent dialogue bubble separate.
    """
    boxes = [
        # Bubble 1: 2 lines
        {"xmin": 100, "ymin": 100, "xmax": 350, "ymax": 125, "text": "It's always good to pick", "conf": 0.96},
        {"xmin": 105, "ymin": 132, "xmax": 345, "ymax": 157, "text": "one that loves dogs.", "conf": 0.94},
        # Bubble 2: 1 line, nearby below
        {"xmin": 110, "ymin": 185, "xmax": 330, "ymax": 210, "text": "Their mind last longer.", "conf": 0.91},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 2
    assert merged[0]["text"] == "It's always good to pick\none that loves dogs."
    assert merged[0]["line_count"] == 2
    assert merged[1]["text"] == "Their mind last longer."
    assert merged[1]["line_count"] == 1


def test_merge_vertical_manga_text_multiline_and_separation():
    """
    Verifies that vertical Japanese text lines in the same bubble are merged Right-to-Left,
    while a nearby independent vertical dialogue bubble remains separate.
    """
    boxes = [
        # Bubble 1: 2 vertical columns (Japanese text read right-to-left)
        {"xmin": 500, "ymin": 100, "xmax": 525, "ymax": 260, "text": "犬が好きな人を", "conf": 0.95},
        {"xmin": 470, "ymin": 105, "xmax": 495, "ymax": 255, "text": "選ぶのは常に良い", "conf": 0.92},
        # Bubble 2: Independent vertical bubble to the left
        {"xmin": 390, "ymin": 100, "xmax": 415, "ymax": 250, "text": "心が長持ちする", "conf": 0.90},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 2
    # First bubble should have joined the 2 columns in RTL order
    assert merged[0]["text"] == "犬が好きな人を\n選ぶのは常に良い"
    assert merged[0]["line_count"] == 2
    assert merged[0]["xmin"] == 470
    assert merged[0]["xmax"] == 525
    # Second bubble remains isolated
    assert merged[1]["text"] == "心が長持ちする"
    assert merged[1]["line_count"] == 1


def test_merge_same_line_horizontal_word_fragments():
    """
    Verifies that words or fragments on the exact same horizontal line
    are joined with a space (not newline) and line_count is preserved as 1.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 160, "ymax": 130, "text": "Hello", "conf": 0.95},
        {"xmin": 170, "ymin": 100, "xmax": 250, "ymax": 130, "text": "world!", "conf": 0.92},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "Hello world!"
    assert merged[0]["line_count"] == 1
    assert merged[0]["xmin"] == 100
    assert merged[0]["xmax"] == 250


def test_merge_same_line_ascenders_descenders_xheight():
    """
    Verifies that lowercase words without ascenders/descenders (e.g. 'see')
    merge with adjacent words having ascenders/descenders ('happy') on the same line.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 180, "ymax": 130, "text": "happy", "conf": 0.95},
        {"xmin": 190, "ymin": 108, "xmax": 240, "ymax": 125, "text": "see", "conf": 0.90},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "happy see"
    assert merged[0]["line_count"] == 1


def test_merge_multiline_with_multiple_fragments_per_line():
    """
    Verifies that a 2-line bubble where each line has multiple detected word fragments
    joins same-line fragments with spaces, different lines with '\\n', and reports line_count = 2.
    """
    boxes = [
        # Line 1: 2 fragments
        {"xmin": 100, "ymin": 100, "xmax": 190, "ymax": 125, "text": "It's always", "conf": 0.95},
        {"xmin": 200, "ymin": 100, "xmax": 350, "ymax": 125, "text": "good to pick", "conf": 0.94},
        # Line 2: 2 fragments
        {"xmin": 105, "ymin": 132, "xmax": 210, "ymax": 157, "text": "one that", "conf": 0.93},
        {"xmin": 220, "ymin": 132, "xmax": 345, "ymax": 157, "text": "loves dogs.", "conf": 0.92},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "It's always good to pick\none that loves dogs."
    assert merged[0]["line_count"] == 2


def test_merge_vertical_japanese_collinear_fragments():
    """
    Verifies that two broken fragments in the same vertical Japanese column
    merge into 1 column without inserting '\\n', preserving line_count = 1.
    """
    boxes = [
        {"xmin": 500, "ymin": 100, "xmax": 525, "ymax": 180, "text": "我が名は", "conf": 0.95},
        {"xmin": 500, "ymin": 185, "xmax": 525, "ymax": 260, "text": "ルシファー", "conf": 0.92},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "我が名はルシファー"
    assert merged[0]["line_count"] == 1


def test_merge_small_cropped_image_multiline():
    """
    Verifies that a multi-line bubble on a small cropped canvas (e.g. 400x400)
    properly merges based on font size and is not choked by image-height ratio.
    """
    boxes = [
        {"xmin": 50, "ymin": 50, "xmax": 250, "ymax": 75, "text": "Line One", "conf": 0.95},
        {"xmin": 52, "ymin": 87, "xmax": 248, "ymax": 112, "text": "Line Two", "conf": 0.90},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=400, img_h=400)
    assert len(merged) == 1
    assert merged[0]["text"] == "Line One\nLine Two"
    assert merged[0]["line_count"] == 2


def test_merge_cjk_same_line_fragments_no_space():
    """
    Verifies that CJK characters on the same line are merged without western spaces.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 180, "ymax": 125, "text": "こんにちは", "conf": 0.95},
        {"xmin": 185, "ymin": 100, "xmax": 230, "ymax": 125, "text": "世界", "conf": 0.93},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "こんにちは世界"
    assert merged[0]["line_count"] == 1


def test_merge_narrow_single_character_multiline():
    """
    Verifies that a multi-line bubble with a single-character or narrow line (e.g. 'I')
    is correctly merged into a single bubble with newline and line_count = 2.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 108, "ymax": 125, "text": "I", "conf": 0.95},
        {"xmin": 100, "ymin": 133, "xmax": 250, "ymax": 158, "text": "know you", "conf": 0.92},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "I\nknow you"
    assert merged[0]["line_count"] == 2
    assert merged[0]["xmin"] == 100
    assert merged[0]["xmax"] == 250


def test_merge_short_western_words_multiline_preserves_newlines():
    """
    Verifies that short Western words on adjacent vertical lines (e.g. 'OH!\\nNO!')
    are NOT misclassified as Japanese vertical text and preserve newlines and line_count = 2.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 117, "ymax": 125, "text": "OH!", "conf": 0.95},
        {"xmin": 100, "ymin": 133, "xmax": 117, "ymax": 158, "text": "NO!", "conf": 0.93},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "OH!\nNO!"
    assert merged[0]["line_count"] == 2


def test_merge_short_words_two_lines_go_on():
    """
    Verifies that short 2-letter words ('Go', 'on') are correctly merged with newline.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 120, "ymax": 125, "text": "Go", "conf": 0.94},
        {"xmin": 100, "ymin": 133, "xmax": 120, "ymax": 158, "text": "on", "conf": 0.91},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "Go\non"
    assert merged[0]["line_count"] == 2


def test_merge_custom_v_thresh_ratio_loose_spacing():
    """
    Verifies that custom v_thresh_ratio parameter is respected to merge dialogue
    lines with unusually loose inter-line spacing.
    """
    # Line height 30, gap 30px: 30px > 0.70 * 30 = 21px (separated under default 0.025)
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 130, "text": "Loose line 1", "conf": 0.95},
        {"xmin": 100, "ymin": 160, "xmax": 300, "ymax": 190, "text": "Loose line 2", "conf": 0.92},
    ]

    # Under default threshold: separated
    default_merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000, v_thresh_ratio=0.025)
    assert len(default_merged) == 2

    # Under adjusted threshold: merged
    loose_merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000, v_thresh_ratio=0.045)
    assert len(loose_merged) == 1
    assert loose_merged[0]["text"] == "Loose line 1\nLoose line 2"
    assert loose_merged[0]["line_count"] == 2


def test_merge_output_schema_and_types():
    """
    Verifies that every output dictionary contains the complete schema required by downstream
    translation, typography, and inpaint pipelines with exact expected types.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125, "text": "Line one", "conf": 0.95},
        {"xmin": 102, "ymin": 132, "xmax": 298, "ymax": 157, "text": "Line two", "conf": 0.85},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    b = merged[0]
    expected_fields = {
        "xmin": int,
        "ymin": int,
        "xmax": int,
        "ymax": int,
        "text": str,
        "conf": float,
        "line_count": int,
    }
    for field_name, expected_type in expected_fields.items():
        assert field_name in b, f"Missing field {field_name}"
        assert isinstance(b[field_name], expected_type), f"Field {field_name} type {type(b[field_name])} != {expected_type}"


def test_merge_narrow_single_char_same_line_horizontal():
    """
    Verifies that narrow single letters or words (e.g. 'I') on the exact same horizontal line
    are correctly merged with adjacent words into a single box with line_count = 1.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 110, "ymax": 125, "text": "I", "conf": 0.95},
        {"xmin": 118, "ymin": 100, "xmax": 160, "ymax": 125, "text": "love", "conf": 0.94},
        {"xmin": 168, "ymin": 100, "xmax": 230, "ymax": 125, "text": "manga", "conf": 0.92},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "I love manga"
    assert merged[0]["line_count"] == 1
    assert merged[0]["xmin"] == 100
    assert merged[0]["xmax"] == 230


def test_merge_horizontal_cjk_with_western_digit_and_no_column_flip():
    """
    Verifies that a horizontal Japanese line containing a narrow Western digit ('第1話')
    is correctly merged into a single horizontal box, without being misclassified as a
    vertical cluster or having its reading order reversed.
    """
    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 125, "ymax": 125, "text": "第", "conf": 0.96},
        {"xmin": 128, "ymin": 100, "xmax": 136, "ymax": 125, "text": "1", "conf": 0.95},
        {"xmin": 140, "ymin": 100, "xmax": 165, "ymax": 125, "text": "話", "conf": 0.94},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "第1話"
    assert merged[0]["line_count"] == 1
    assert merged[0]["xmin"] == 100
    assert merged[0]["xmax"] == 165


def test_merge_cjk_punctuation_no_spurious_space():
    """
    Verifies that CJK punctuation symbols (like ideographic period '。', comma '、', and brackets '「', '」')
    do not cause spurious ASCII spaces when joined with adjacent CJK text.
    """
    from app.core.ocr.base import _join_line_texts
    assert _join_line_texts(["そうだね。", "」"]) == "そうだね。」"
    assert _join_line_texts(["はい、", "「えっ？」"]) == "はい、「えっ？」"

    boxes = [
        {"xmin": 100, "ymin": 100, "xmax": 180, "ymax": 125, "text": "そうだね。", "conf": 0.95},
        {"xmin": 182, "ymin": 100, "xmax": 200, "ymax": 125, "text": "」", "conf": 0.90},
    ]
    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "そうだね。」"
    assert merged[0]["line_count"] == 1


def test_merge_vertical_japanese_unequal_column_lengths():
    """
    Verifies that a multi-column vertical Japanese speech bubble with unequal column lengths
    (e.g. 7-character column and 1-character column '？') correctly merges into 1 bubble
    with newline separation and line_count = 2.
    """
    boxes = [
        {"xmin": 500, "ymin": 100, "xmax": 525, "ymax": 250, "text": "だれかいるのか", "conf": 0.95},
        {"xmin": 470, "ymin": 100, "xmax": 495, "ymax": 125, "text": "？", "conf": 0.90},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "だれかいるのか\n？"
    assert merged[0]["line_count"] == 2
    assert merged[0]["xmin"] == 470
    assert merged[0]["xmax"] == 525


def test_merge_vertical_japanese_short_punctuation_fragment():
    """
    Verifies that a vertical Japanese column broken into text and a trailing punctuation fragment
    (e.g. ellipsis '…' or dash '―') correctly merges into 1 column without inserting newline.
    """
    boxes = [
        {"xmin": 500, "ymin": 100, "xmax": 525, "ymax": 180, "text": "お前は", "conf": 0.95},
        {"xmin": 500, "ymin": 185, "xmax": 525, "ymax": 200, "text": "…", "conf": 0.90},
    ]

    merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
    assert len(merged) == 1
    assert merged[0]["text"] == "お前は…"
    assert merged[0]["line_count"] == 1
    assert merged[0]["ymin"] == 100
    assert merged[0]["ymax"] == 200


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
