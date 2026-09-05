"""
End-to-End Real Manga Test Suite: Pixiv 10-Page Benchmark
=========================================================
Target Dataset: D:\\baidu\\download\\baidu\\pixiv\\test (10 real images)

Dimensions Covered:
  Dimension 1: Bubble Segmentation & Physical Isolation (F1, F2, F3, F4)
  Dimension 2: OCR Accuracy & Language Routing (F5, F6, F7, F8)
  Dimension 3: Inpainting & Typography on Dark/Inverted Boxes (F9, F10, F11)
  Dimension 4: Regression & All-Image Dataset Integrity (F12, F13)

Runner Compatibility:
  pytest tests/e2e/test_pixiv_10pages.py -v
  python tests/e2e/test_pixiv_10pages.py
"""

import os
import sys
import re
import time
import math
import json
import pytest
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from app.core.models import TranslationBlock, StyleConfig
from app.core.ocr.base import merge_adjacent_boxes, can_merge_pair, compute_bubble_labels
from app.core.inpaint.opencv_engine import OpenCVInpainter
from app.core.typography.engine import TypographyEngine
from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine

TEST_DATASET_DIR = r"D:\baidu\download\baidu\pixiv\test"
CACHE_DIR = os.path.join(TEST_DATASET_DIR, ".amt_cache")

# Ground Truth Benchmark Specifications for the 10 real test images
DATASET_SPECS = {
    "126464149_p000.jpg": {
        "res": (1600, 1600),
        "min_bubbles": 6,
        "has_dark_box": False,
        "dark_boxes": [],
        "key_phrases": ["shared", "toy", "finishing touch"]
    },
    "126464149_p001.jpg": {
        "res": (1600, 1600),
        "min_bubbles": 6,
        "has_dark_box": False,
        "dark_boxes": [],
        "key_phrases": ["grey", "ghost", "nibble", "mated"]
    },
    "126464149_p002.jpg": {
        "res": (1600, 663),
        "min_bubbles": 5,
        "has_dark_box": False,
        "dark_boxes": [],
        "has_qr_code": True,
        "key_phrases": ["nimbletail", "support", "tired"]
    },
    "88061806_p000.jpg": {
        "res": (1536, 1536),
        "min_bubbles": 7,
        "has_dark_box": True,
        "dark_boxes": [
            {"label": "Friend B (20)", "box": (1111, 529, 1233, 603)},
            {"label": "Friend A (19)", "box": (265, 711, 386, 783)},
            {"label": "Chris (19)", "box": (511, 751, 591, 825)}
        ],
        "key_phrases": ["friend b", "friend a", "chris", "no sweat", "no-fap", "special challenge"]
    },
    "88061806_p001.jpg": {
        "res": (1536, 1536),
        "min_bubbles": 9,
        "has_dark_box": True,
        "dark_boxes": [
            {"label": "1 Week and 2 masturbations later", "box": (15, 34, 301, 103)},
            {"label": "Friend A (Runner-up)", "box": (1306, 136, 1475, 215)},
            {"label": "Friend B (Winner)", "box": (85, 195, 217, 269)},
            {"label": "Chris (Loser)", "box": (994, 954, 1103, 1027)}
        ],
        "key_phrases": ["friend a", "friend b", "chris", "runner-up", "winner", "loser", "maid"]
    },
    "88061806_p002.jpg": {
        "res": (1536, 1536),
        "min_bubbles": 10,
        "has_dark_box": True,
        "dark_boxes": [
            {"label": "2 days and 1 masturbation later", "box": (34, 13, 306, 84)}
        ],
        "isolation_pairs": [
            {
                "desc": "Right adjacent oval speech bubbles",
                "box_a": (1227, 137, 1477, 240),  # Bubble A: Hehe! Bold of you...
                "box_b": (1207, 311, 1501, 556)   # Bubble B: Now you have to wear...
            }
        ],
        "slanted_notes": [
            {"desc": "What a naughty maid!", "approx_box": (1335, 80, 1467, 149)},
            {"desc": "And you ruined his sister's dress...", "approx_box": (300, 395, 360, 440)}
        ],
        "key_phrases": ["bold of you", "maid costume", "slutty maid", "naughty maid", "sister's dress"]
    },
    "88061806_p003.jpg": {
        "res": (1536, 1536),
        "min_bubbles": 9,
        "has_dark_box": True,
        "dark_boxes": [
            {"label": "2 days and ? anal masturbations later", "box": (12, 13, 298, 82)}
        ],
        "key_phrases": ["what the fuck", "caught him", "bottle", "horny juice"]
    },
    "88061806_p004.jpg": {
        "res": (1536, 1536),
        "min_bubbles": 6,
        "has_dark_box": True,
        "dark_boxes": [
            {"label": "Minutes later", "box": (0, 19, 257, 61)}
        ],
        "key_phrases": ["minutes later", "check it out", "what the hell", "boner"]
    },
    "88061806_p005.jpg": {
        "res": (1536, 1536),
        "min_bubbles": 6,
        "has_dark_box": True,
        "dark_boxes": [
            {"label": "Several minutes later", "box": (17, 19, 325, 55)}
        ],
        "key_phrases": ["several minutes later", "chris", "slut", "cocks"]
    },
    "88061806_p006.jpg": {
        "res": (1536, 1536),
        "min_bubbles": 6,
        "has_dark_box": True,
        "dark_boxes": [
            {"label": "1 week later", "box": (33, 225, 271, 267)}
        ],
        "key_phrases": ["1 week later", "crystal", "girlfriend", "cocks", "party"]
    }
}


# --- Helpers & Cached Pipeline Execution ---

_GLOBAL_OCR_ENGINE = None

def get_ocr_engine() -> OCREngine:
    global _GLOBAL_OCR_ENGINE
    if _GLOBAL_OCR_ENGINE is None:
        _GLOBAL_OCR_ENGINE = OCREngine(engine_type="ctd", use_gpu=True, lang="en")
    return _GLOBAL_OCR_ENGINE


def load_pixiv_image(filename: str) -> np.ndarray:
    path = os.path.join(TEST_DATASET_DIR, filename)
    assert os.path.isfile(path), f"Pixiv test image not found: {path}"
    # Read binary bytes for Windows unicode path safety
    with open(path, "rb") as f:
        data = bytearray(f.read())
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to decode image: {path}"
    return img


_OCR_CACHE = {}

def run_or_load_ocr(filename: str, img: np.ndarray = None, force_rerun: bool = False):
    """
    Executes real OCREngine on the image, caching results in-memory during the test session
    to avoid redundant model inference across multiple test methods.
    """
    if not force_rerun and filename in _OCR_CACHE:
        return _OCR_CACHE[filename]

    if img is None:
        img = load_pixiv_image(filename)

    engine = get_ocr_engine()
    blocks = engine.detect_and_recognize(img)
    _OCR_CACHE[filename] = blocks
    return blocks



# ==============================================================================
# DIMENSION 1: Bubble Segmentation & Physical Isolation (F1, F2, F3, F4)
# ==============================================================================

class TestDimension1BubbleSegmentation:
    """
    Validates that tight adjacent speech bubbles and slanted annotations
    are physically separated into distinct TranslationBlocks and not merged.
    """

    def test_p002_adjacent_right_bubbles_strictly_separated(self):
        """
        On 88061806_p002.jpg, verify the two adjacent right bubbles:
          - Bubble A ("Hehe! Bold of you...") at (1227, 137)-(1477, 240)
          - Bubble B ("Now you have to wear...") at (1207, 311)-(1501, 556)
        are detected as 2 separate TranslationBlocks, strictly NOT merged into 1.
        """
        img = load_pixiv_image("88061806_p002.jpg")
        h_img, w_img = img.shape[:2]

        # 1. Direct Algorithmic Barrier Test in can_merge_pair
        # Simulate text line from Bubble A and text line from Bubble B
        line_a = {
            "xmin": 1230, "ymin": 150, "xmax": 1470, "ymax": 230,
            "text": "Hehe! Bold of you, jacking off in the maid costume.",
            "angle": 0.0, "line_count": 2
        }
        line_b = {
            "xmin": 1215, "ymin": 320, "xmax": 1490, "ymax": 530,
            "text": "Now you have to wear this slutty maid dress",
            "angle": 0.0, "line_count": 3
        }

        # can_merge_pair must detect the dark boundary barrier between y=230 and y=320
        # and strictly return False
        merge_allowed = can_merge_pair(line_a, line_b, w_img, h_img, image=img)
        assert not merge_allowed, (
            "can_merge_pair falsely allowed merging adjacent independent right bubbles across the dark boundary barrier!"
        )

        # 2. Full Pipeline Execution Check on real image
        blocks = run_or_load_ocr("88061806_p002.jpg", img=img)
        assert len(blocks) >= 8, f"Expected at least 8 blocks on p002, got {len(blocks)}"

        # Find block containing Bubble A text and block containing Bubble B text
        block_a = None
        block_b = None
        for b in blocks:
            text = (b.get("original_text") or "").lower()
            if any(k in text for k in ["bold of you", "hehe", "jacking"]):
                block_a = b
            if any(k in text for k in ["wear this slutty", "cage", "cumming", "furniture"]):
                block_b = b

        assert block_a is not None, "Failed to detect Bubble A ('Hehe! Bold of you...')"
        assert block_b is not None, "Failed to detect Bubble B ('Now you have to wear...')"
        assert block_a["id"] != block_b["id"], (
            f"Bubble A and Bubble B were erroneously merged into single TranslationBlock {block_a['id']}!"
        )
        assert block_a["ymax"] <= block_b["ymin"], (
            f"Bubble A (ymax={block_a['ymax']}%) overlaps vertically with Bubble B (ymin={block_b['ymin']}%)!"
        )

        # Verify NO block spans across both Bubble A (y ~ 140) and Bubble B (y ~ 500) on the right (x > 75%)
        violating_merged_blocks = []
        for b in blocks:
            xmin_px = (b["xmin"] / 100.0) * w_img
            ymin_px = (b["ymin"] / 100.0) * h_img
            xmax_px = (b["xmax"] / 100.0) * w_img
            ymax_px = (b["ymax"] / 100.0) * h_img
            text = (b.get("original_text") or "").lower()

            # Check if block is located in the right dialogue region
            if xmin_px > 0.70 * w_img:
                # Check for erroneous merging across the vertical barrier (y=240 to y=310)
                if ymin_px < 230 and ymax_px > 350:
                    # Also check if text from both bubbles is joined together
                    has_a = any(k in text for k in ["bold", "hehe", "jacking", "costume"])
                    has_b = any(k in text for k in ["wear", "slutty", "cage", "cumming", "furniture"])
                    if has_a and has_b:
                        violating_merged_blocks.append(b)

        assert len(violating_merged_blocks) == 0, (
            f"Detected erroneous merged block bridging Bubble A and Bubble B: {violating_merged_blocks}"
        )

    def test_p002_slanted_side_notes_isolated(self):
        """
        Verify slanted side notes on 88061806_p002.jpg:
          - "What a naughty maid!"
          - "And you ruined his sister's dress..."
        are isolated into separate blocks, not enveloped by main bubbles.
        """
        img = load_pixiv_image("88061806_p002.jpg")
        blocks = run_or_load_ocr("88061806_p002.jpg", img=img)

        found_slanted_note = False
        for b in blocks:
            text = (b.get("original_text") or "").lower()
            if "what" in text and ("naught" in text or "maid" in text):
                found_slanted_note = True
                w_pct = b["xmax"] - b["xmin"]
                assert w_pct < 20.0, f"Slanted note merged into large block (w={w_pct}%): {b}"

        assert found_slanted_note, "Did not detect slanted side note 'What a naughty maid!'"

    @pytest.mark.parametrize("filename", list(DATASET_SPECS.keys()))
    def test_all_10pages_minimum_bubble_count(self, filename):
        """
        Verifies that every one of the 10 real Pixiv pages meets the minimum bubble count.
        """
        spec = DATASET_SPECS[filename]
        img = load_pixiv_image(filename)
        blocks = run_or_load_ocr(filename, img=img)

        min_expected = spec["min_bubbles"]
        assert len(blocks) >= min_expected, (
            f"Page {filename} detected {len(blocks)} bubbles, which is less than expected minimum {min_expected}!"
        )


# ==============================================================================
# DIMENSION 2: OCR Accuracy & Language Routing (F5, F6, F7, F8)
# ==============================================================================

class TestDimension2OCRLanguageRouting:
    """
    Validates that English manga routes cleanly to EasyOCR, produces ZERO Japanese
    kana hallucinations, and preserves 2D reading order without word scrambling.
    """

    @pytest.mark.parametrize("filename", [
        "88061806_p000.jpg", "88061806_p001.jpg", "88061806_p002.jpg",
        "88061806_p003.jpg", "88061806_p004.jpg", "88061806_p005.jpg", "88061806_p006.jpg",
        "126464149_p000.jpg", "126464149_p001.jpg", "126464149_p002.jpg"
    ])
    def test_zero_japanese_kana_hallucinations(self, filename):
        """
        English manga must route to EasyOCR and produce ZERO Japanese kana
        (Hiragana [\u3040-\u309f] or Katakana [\u30a0-\u30ff]).
        """
        img = load_pixiv_image(filename)
        blocks = run_or_load_ocr(filename, img=img)

        kana_regex = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
        hallucinations = []

        for b in blocks:
            raw_text = b.get("original_text", "")
            matches = kana_regex.findall(raw_text)
            if matches:
                hallucinations.append((b.get("id"), raw_text, matches))

        assert len(hallucinations) == 0, (
            f"Page {filename} produced Japanese Kana hallucinations in English manga: {hallucinations}"
        )

    def test_easyocr_reading_order_and_word_integrity(self):
        """
        Verify that EasyOCR detects multi-line text and character labels in correct
        reading order without word scrambling (e.g. 'Friend A (19)', 'Minutes later').
        """
        # Test 1: p000 character badges
        img_p000 = load_pixiv_image("88061806_p000.jpg")
        blocks_p000 = run_or_load_ocr("88061806_p000.jpg", img=img_p000)
        all_text_p000 = " ".join(b.get("original_text", "") for b in blocks_p000).lower()

        # Should contain character names
        assert "friend" in all_text_p000, "Expected 'friend' in p000 text"
        assert "chris" in all_text_p000, "Expected 'chris' in p000 text"

        # Test 2: p004 top banner "Minutes later"
        img_p004 = load_pixiv_image("88061806_p004.jpg")
        blocks_p004 = run_or_load_ocr("88061806_p004.jpg", img=img_p004)
        top_banner_p004 = [
            b for b in blocks_p004
            if (b.get("ymin", 100) < 10.0 and "minute" in (b.get("original_text") or "").lower())
        ]
        assert len(top_banner_p004) > 0, "Failed to detect top banner 'Minutes later' on p004"
        banner_text = top_banner_p004[0].get("original_text", "").strip()
        # Words should not be scrambled
        assert "minutes" in banner_text.lower() and "later" in banner_text.lower(), (
            f"Banner text scrambled: '{banner_text}'"
        )


# ==============================================================================
# DIMENSION 3: Inpainting & Typography on Dark/Inverted Boxes (F9, F10, F11)
# ==============================================================================

class TestDimension3InpaintAndTypography:
    """
    Validates clean solid flat-fill on dark/inverted boxes without gray halos,
    and high-contrast white text typography rendering with dark outlines.
    """

    def test_dark_box_clean_inpainting(self):
        """
        Verify dark/inverted boxes ('2 days and 1 masturbation later', 'Friend A (19)')
        have clean background fill without gray halos or feather smudges.
        """
        img = load_pixiv_image("88061806_p002.jpg")
        h_img, w_img = img.shape[:2]

        # Top dark box: (34, 13)-(306, 84)
        x1, y1, x2, y2 = 34, 13, 306, 84
        dark_block = {
            "id": "dark_box_test",
            "xmin": round((x1 / w_img) * 100.0, 2),
            "ymin": round((y1 / h_img) * 100.0, 2),
            "xmax": round((x2 / w_img) * 100.0, 2),
            "ymax": round((y2 / h_img) * 100.0, 2),
            "type": "bubble",
            "bg_color": "#000000",
            "text_color": "#FFFFFF",
            "original_text": "2 days and 1 masturbation later"
        }

        inpaint_eng = InpaintEngine(mode="auto")
        erased = inpaint_eng.inpaint(img, [dark_block], bubble_dilation=3, feather_radius=4)
        assert erased is not None, "Inpainting returned None"

        # Crop the inner text area of the dark box (padding 4px inside to avoid outer border)
        inner_crop = erased[y1 + 4 : y2 - 4, x1 + 4 : x2 - 4]
        assert inner_crop.size > 0, "Empty crop for dark box"

        # Check inpainting cleanliness:
        # A clean black box in erased image must have low mean luminance (< 15)
        # and essentially zero white smudge pixels (> 60)
        mean_lum = float(np.mean(inner_crop))
        smudge_pixels = int(np.sum(inner_crop > 60))

        assert mean_lum < 15.0, (
            f"Dark box inpainting mean luminance too high ({mean_lum:.1f} >= 15.0), indicates gray smudge!"
        )
        assert smudge_pixels < 25, (
            f"Dark box contains {smudge_pixels} white smudge pixels (>60) after inpainting!"
        )

    def test_dark_box_typography_renders_white_text_with_outline(self):
        """
        Verify typography engine renders high-contrast white text (#FFFFFF) with
        dark outline on dark/inverted background blocks.
        """
        img = load_pixiv_image("88061806_p000.jpg")
        h_img, w_img = img.shape[:2]

        # Dark character tag box "Friend A (19)" at (265, 711)-(386, 783)
        x1, y1, x2, y2 = 265, 711, 386, 783
        dark_block = TranslationBlock(
            id="friend_a_tag",
            original_text="Friend A (19)",
            translated_text="朋友 A (19)",
            xmin=round((x1 / w_img) * 100.0, 2),
            ymin=round((y1 / h_img) * 100.0, 2),
            xmax=round((x2 / w_img) * 100.0, 2),
            ymax=round((y2 / h_img) * 100.0, 2),
            bg_color="#000000",
            text_color="#FFFFFF",
            type="bubble"
        )

        # Prepare solid black background crop
        erased_base = img.copy()
        erased_base[y1:y2, x1:x2] = (0, 0, 0)

        typo_eng = TypographyEngine()
        cfg = StyleConfig()

        rendered_img = typo_eng.render_page(erased_base, [dark_block], cfg)
        assert rendered_img is not None, "Typography rendering returned None"

        rendered_crop = rendered_img[y1:y2, x1:x2]

        # Verify rendered text is white (#FFFFFF):
        # A rendered white text block MUST have high-luminance pixels (> 180).
        # If text was mistakenly drawn black (#000000) on black box, bright pixels would be 0!
        bright_pixels = int(np.sum(rendered_crop > 180))
        assert bright_pixels > 20, (
            f"Rendered dark box has only {bright_pixels} bright pixels; expected white text (#FFFFFF)!"
        )

    def test_qr_code_preservation(self):
        """
        Verify that inpainting on pages with QR codes (126464149_p002.jpg)
        preserves the QR code with zero alteration.
        """
        img = load_pixiv_image("126464149_p002.jpg")
        h_img, w_img = img.shape[:2]

        blocks = run_or_load_ocr("126464149_p002.jpg", img=img)
        inpaint_eng = InpaintEngine(mode="auto")
        erased = inpaint_eng.inpaint(img, blocks)

        # Detect QR code in original image
        qr_detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img)
        if retval and points is not None and len(points) > 0:
            pts = np.asarray(points[0], dtype=np.int32)
            qx1, qy1 = np.min(pts[:, 0]), np.min(pts[:, 1])
            qx2, qy2 = np.max(pts[:, 0]), np.max(pts[:, 1])

            orig_qr = img[qy1:qy2, qx1:qx2]
            eras_qr = erased[qy1:qy2, qx1:qx2]

            # Inpaint must preserve QR area exactly (0 pixel diff)
            diff = np.max(np.abs(orig_qr.astype(int) - eras_qr.astype(int)))
            assert diff == 0, f"QR code region was modified during inpainting (max diff = {diff})!"


# ==============================================================================
# DIMENSION 4: Regression & Dataset Integrity (F12, F13)
# ==============================================================================

class TestDimension4DatasetIntegrity:
    """
    Validates dataset integrity across all 10 real images and ensures
    CLI / Pytest runner invocation compatibility.
    """

    @pytest.mark.parametrize("filename,spec", list(DATASET_SPECS.items()))
    def test_all_10_images_exist_and_match_spec(self, filename, spec):
        """
        Verifies that all 10 images exist in D:\\baidu\\download\\baidu\\pixiv\\test
        and match expected image dimensions.
        """
        img = load_pixiv_image(filename)
        h, w = img.shape[:2]
        expected_w, expected_h = spec["res"]

        assert (w, h) == (expected_w, expected_h), (
            f"Image {filename} resolution mismatch: expected ({expected_w}, {expected_h}), got ({w}, {h})"
        )


# ==============================================================================
# STANDALONE RUNNER ENTRY POINT
# ==============================================================================

def run_all_pixiv_e2e_tests():
    """
    Standalone runner providing execution summary table.
    """
    print("=" * 80)
    print("  PIXIV 10-PAGE REAL MANGA TRANSLATION E2E TEST SUITE")
    print("=" * 80)
    print(f"Dataset Path : {TEST_DATASET_DIR}")
    print(f"Total Pages  : {len(DATASET_SPECS)}")
    print(f"Timestamp    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = time.time()

    # Run pytest on this file
    this_file = os.path.abspath(__file__)
    exit_code = pytest.main(["-v", "-s", this_file])

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    if exit_code == 0:
        print(f"  [SUCCESS] All 4 Dimensions Passed across 10 Pixiv Pages! ({elapsed:.2f}s)")
    else:
        print(f"  [FAILURE] Test suite failed with exit code {exit_code} ({elapsed:.2f}s)")
    print("=" * 80)
    return exit_code


if __name__ == "__main__":
    sys.exit(run_all_pixiv_e2e_tests())
