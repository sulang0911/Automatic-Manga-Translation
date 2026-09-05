"""
tests/unit/test_qr_bubble_adaptive.py
Unit tests verifying QR code detection, spurious OCR noise rejection,
zero-pixel inpainting protection, and bubble-aware adaptive line clustering.
"""
import os
import pytest
import numpy as np
import cv2

from app.core.models import TranslationBlock
from app.core.ocr.base import can_merge_pair, merge_adjacent_boxes, compute_bubble_labels
from app.core.inpaint.opencv_engine import OpenCVInpainter


# ==============================================================================
# Helper functions for QR code generation and filtering
# ==============================================================================

def create_synthetic_qr_image(content: str = "https://example.com/test_qr", qr_size: int = 200) -> np.ndarray:
    """Generates a synthetic image containing a decodable QR code."""
    encoder = cv2.QRCodeEncoder.create()
    raw_qr = encoder.encode(content)
    # Scale up with nearest neighbor to preserve crisp edges
    scaled_qr = cv2.resize(raw_qr, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)
    # Add quiet zone border (at least 4 modules)
    border = int(qr_size * 0.15)
    qr_with_border = cv2.copyMakeBorder(
        scaled_qr, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255
    )
    # Convert to 3-channel BGR
    return cv2.cvtColor(qr_with_border, cv2.COLOR_GRAY2BGR)


def filter_spurious_ocr_boxes_by_bbox(boxes, qr_bbox, threshold_ratio=0.60):
    """Reference filter: discards OCR boxes that fall inside or overlap the QR bounding box."""
    qx1, qy1, qx2, qy2 = qr_bbox
    filtered = []
    for b in boxes:
        bx1 = min(b["xmin"], b["xmax"])
        by1 = min(b["ymin"], b["ymax"])
        bx2 = max(b["xmin"], b["xmax"])
        by2 = max(b["ymin"], b["ymax"])
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        b_area = bw * bh

        # Check center containment
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        if qx1 <= cx <= qx2 and qy1 <= cy <= qy2:
            continue

        # Check area overlap
        ix1 = max(bx1, qx1)
        iy1 = max(by1, qy1)
        ix2 = min(bx2, qx2)
        iy2 = min(by2, qy2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            if (inter_area / b_area) >= threshold_ratio:
                continue

        filtered.append(b)
    return filtered


# compute_bubble_labels is imported from app.core.ocr.base


# ==============================================================================
# 1. QR Code Detection via OpenCV
# ==============================================================================

class TestQRCodeDetectionOpenCV:
    """Tests for OpenCV-based QR code detection on synthetic and real images."""

    def test_qr_detection_synthetic_image(self):
        """OpenCV QRCodeDetector detects and decodes a generated synthetic QR code."""
        test_url = "https://example.com/test_qr"
        qr_img = create_synthetic_qr_image(content=test_url, qr_size=240)

        # Place onto a larger manga page canvas (500x500) at offset (100, 120)
        canvas = np.full((600, 600, 3), 240, dtype=np.uint8)
        h_qr, w_qr = qr_img.shape[:2]
        canvas[120:120+h_qr, 100:100+w_qr] = qr_img

        detector = cv2.QRCodeDetector()
        val, pts, _ = detector.detectAndDecode(canvas)

        assert val == test_url, f"Expected decoded URL '{test_url}', got '{val}'"
        assert pts is not None and len(pts) > 0
        p = pts.reshape(-1, 2)
        xmin, ymin = p[:, 0].min(), p[:, 1].min()
        xmax, ymax = p[:, 0].max(), p[:, 1].max()

        # Bounding box should be within the placed area [100..100+w_qr, 120..120+h_qr]
        assert xmin >= 90 and ymin >= 110
        assert xmax <= 100 + w_qr + 10 and ymax <= 120 + h_qr + 10

    def test_qr_detection_real_sample_image(self):
        """OpenCV QRCodeDetector identifies the QR code in exported_chapter/media_1788518641910.jpg."""
        sample_path = "exported_chapter/media_1788518641910.jpg"
        if not os.path.exists(sample_path):
            pytest.skip(f"Real sample {sample_path} not found in workspace.")

        img = cv2.imread(sample_path)
        assert img is not None, "Failed to load sample image."

        detector = cv2.QRCodeDetector()
        val, pts, _ = detector.detectAndDecode(img)

        assert "nimbletail.com" in val.lower(), f"Unexpected QR payload: {val}"
        assert pts is not None and len(pts) > 0
        p = pts.reshape(-1, 2)
        x0, y0 = p[:, 0].min(), p[:, 1].min()
        x1, y1 = p[:, 0].max(), p[:, 1].max()

        # Ground truth: QR is located approximately at (46, 264, 129, 346)
        assert abs(x0 - 46) <= 8, f"xmin {x0} deviates from expected 46"
        assert abs(y0 - 264) <= 8, f"ymin {y0} deviates from expected 264"
        assert abs(x1 - 129) <= 8, f"xmax {x1} deviates from expected 129"
        assert abs(y1 - 346) <= 8, f"ymax {y1} deviates from expected 346"


# ==============================================================================
# 2. Spurious OCR Noise Filtering inside QR Code Region
# ==============================================================================

class TestSpuriousOCRNoiseFiltering:
    """Tests for filtering out false-positive OCR fragments inside detected QR regions."""

    def test_spurious_boxes_inside_qr_are_discarded(self):
        """Boxes inside the QR code boundaries must be filtered out, preserving external dialogue."""
        qr_bbox = (50, 200, 160, 310)

        boxes = [
            # Spurious OCR detections caused by QR finder patterns & black-and-white grid dots
            {"xmin": 60, "ymin": 210, "xmax": 110, "ymax": 235, "text": "#ittA8n", "conf": 0.45},
            {"xmin": 70, "ymin": 240, "xmax": 145, "ymax": 265, "text": "DrrnrD", "conf": 0.50},
            {"xmin": 55, "ymin": 270, "xmax": 150, "ymax": 300, "text": "Genri", "conf": 0.40},
            # Legitimate dialogue box located elsewhere on the page
            {"xmin": 300, "ymin": 120, "xmax": 520, "ymax": 155, "text": "Wait for me!", "conf": 0.98},
            {"xmin": 300, "ymin": 165, "xmax": 520, "ymax": 195, "text": "Don't go yet!", "conf": 0.95},
        ]

        cleaned = filter_spurious_ocr_boxes_by_bbox(boxes, qr_bbox)

        assert len(cleaned) == 2
        assert cleaned[0]["text"] == "Wait for me!"
        assert cleaned[1]["text"] == "Don't go yet!"

    def test_partial_overlap_filtering(self):
        """A box whose area is mostly (> 60%) inside the QR region should be filtered."""
        qr_bbox = (100, 100, 200, 200)

        # 80% inside QR
        box_mostly_in = {"xmin": 120, "ymin": 120, "xmax": 220, "ymax": 180, "text": "Noise1"}
        # 10% inside QR (should NOT be filtered)
        box_barely_in = {"xmin": 190, "ymin": 80, "xmax": 290, "ymax": 140, "text": "Valid Speech"}

        cleaned = filter_spurious_ocr_boxes_by_bbox([box_mostly_in, box_barely_in], qr_bbox)
        assert len(cleaned) == 1
        assert cleaned[0]["text"] == "Valid Speech"


# ==============================================================================
# 3. QR Code Preservation during Inpainting Mask Creation
# ==============================================================================

class TestInpaintingQRPreservation:
    """Tests verifying 0-pixel modification of QR code areas during inpainting."""

    def test_inpainting_mask_zero_pixel_change_in_qr_area_synthetic(self):
        """
        When inpainting adjacent dialogue text, the QR code area must remain 100% bit-identical
        (zero pixel difference) and retain its decodability.
        """
        # Create 500x600 canvas
        canvas = np.full((500, 600, 3), 245, dtype=np.uint8)

        # Generate and paste QR code at (40, 150, 180, 290)
        qr_img = create_synthetic_qr_image("https://example.com/author_qr", qr_size=100)
        qh, qw = qr_img.shape[:2]
        qx1, qy1 = 40, 150
        qx2, qy2 = qx1 + qw, qy1 + qh
        canvas[qy1:qy2, qx1:qx2] = qr_img
        qr_original_crop = canvas[qy1:qy2, qx1:qx2].copy()

        # Draw a speech bubble with text to the right: (250, 100, 520, 220)
        cv2.rectangle(canvas, (250, 100), (520, 220), (255, 255, 255), -1)
        cv2.putText(canvas, "Dialogue Line 1", (270, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Dialogue Line 2", (270, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Dialogue translation block (normalized coordinates)
        w_img, h_img = 600, 500
        block = TranslationBlock.from_pixel_box(
            250, 100, 520, 220,
            img_width=w_img, img_height=h_img,
            original_text="Dialogue Line 1\nDialogue Line 2"
        )

        inpainter = OpenCVInpainter(method="telea")
        erased = inpainter.inpaint(canvas, [block])
        assert erased is not None

        # Verify QR code crop has ZERO pixel modification
        erased_qr_crop = erased[qy1:qy2, qx1:qx2]
        pixel_diff = np.sum(np.abs(erased_qr_crop.astype(int) - qr_original_crop.astype(int)))
        assert pixel_diff == 0, f"QR area was modified by inpainting: diff sum = {pixel_diff}"
        assert np.array_equal(erased_qr_crop, qr_original_crop), "QR code pixels must be identical"

        # Verify QR code remains decodable after inpainting
        det = cv2.QRCodeDetector()
        val, pts, _ = det.detectAndDecode(erased)
        assert val == "https://example.com/author_qr", "QR code lost decodability after inpainting"

    def test_inpainting_preserves_real_sample_qr_region(self):
        """Verify inpainting preserves the exact pixels of the QR code in media_1788518641910.jpg."""
        sample_path = "exported_chapter/media_1788518641910.jpg"
        if not os.path.exists(sample_path):
            pytest.skip(f"Real sample {sample_path} not found.")

        img = cv2.imread(sample_path)
        det = cv2.QRCodeDetector()
        val, pts, _ = det.detectAndDecode(img)
        assert pts is not None and len(pts) > 0

        p = pts.reshape(-1, 2)
        qx1, qy1 = int(p[:, 0].min()), int(p[:, 1].min())
        qx2, qy2 = int(p[:, 0].max()), int(p[:, 1].max())
        original_qr = img[qy1:qy2, qx1:qx2].copy()

        # Define a speech bubble block outside the QR area
        h_img, w_img = img.shape[:2]
        # Text block far to the right of the QR code
        block = TranslationBlock.from_pixel_box(
            min(w_img - 200, qx2 + 50), 50, w_img - 20, 150,
            img_width=w_img, img_height=h_img,
            original_text="Credit Info"
        )

        inpainter = OpenCVInpainter(method="telea")
        erased = inpainter.inpaint(img, [block])
        assert erased is not None

        erased_qr = erased[qy1:qy2, qx1:qx2]
        assert np.array_equal(erased_qr, original_qr), "Sample QR region must remain unaltered"


# ==============================================================================
# 4. Adaptive Bubble Clustering Tests
# ==============================================================================

class TestAdaptiveBubbleClustering:
    """
    Tests verifying adaptive line spacing merging (1.8 ~ 2.2x line_h) inside bubbles,
    separation of independent bubbles, and 100% backward compatibility of coordinate defaults.
    """

    def test_coordinate_only_default_preserves_backward_compatibility(self):
        """
        Under default coordinate-only parameters (v_thresh_ratio=0.025), two lines separated
        by 1.0x line height (> 0.70 * line_h ceiling) must remain SEPARATE.
        Guarantees 100% backward compatibility with baseline test suite.
        """
        # line_h = 30, gap = 30px (1.0x line_h).
        # 30px > 0.70 * 30 = 21px ceiling.
        b1 = {"xmin": 120, "ymin": 100, "xmax": 280, "ymax": 130, "text": "Top bubble"}
        b2 = {"xmin": 120, "ymin": 160, "xmax": 280, "ymax": 190, "text": "Bottom bubble"}

        can_merge = can_merge_pair(b1, b2, img_w=1000, img_h=1000, v_thresh_ratio=0.025)
        assert can_merge is False, "Default threshold must not merge 1.0x line gap without adaptive context"

        merged = merge_adjacent_boxes([b1, b2], img_w=1000, img_h=1000, v_thresh_ratio=0.025)
        assert len(merged) == 2, "Default merge must produce 2 separate blocks"

    def test_adaptive_spacing_merges_lines_with_large_gap(self):
        """
        When adaptive spacing is enabled (e.g. v_thresh_ratio allowing 1.8 ~ 2.2x line_h),
        well-aligned lines with line spacing 1.8 ~ 2.0x line_h successfully merge.
        """
        # line_h = 25, gap = 48px (48 / 25 = 1.92x line_h)
        b1 = {"xmin": 200, "ymin": 100, "xmax": 450, "ymax": 125, "text": "Because you are"}
        b2 = {"xmin": 202, "ymin": 173, "xmax": 448, "ymax": 198, "text": "the chosen one."}

        # v_scale = 0.075 / 0.025 = 3.0 -> max_gap = 0.70 * 25 * 3.0 = 52.5px > 48px
        can_merge = can_merge_pair(b1, b2, img_w=1000, img_h=1000, v_thresh_ratio=0.075)
        assert can_merge is True

        merged = merge_adjacent_boxes([b1, b2], img_w=1000, img_h=1000, v_thresh_ratio=0.075)
        assert len(merged) == 1
        assert merged[0]["text"] == "Because you are\nthe chosen one."
        assert merged[0]["line_count"] == 2

    def test_sharing_bubble_closure_discrimination(self):
        """
        Connected-component morphology correctly identifies whether two text lines share
        a continuous speech bubble closure or reside in separate bordered bubbles.
        """
        h, w = 600, 800
        canvas = np.full((h, w, 3), 200, dtype=np.uint8)  # Gray manga background

        # Case A: Two lines inside the SAME large speech bubble (white closure with dark border)
        cv2.ellipse(canvas, (400, 200), (220, 130), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, (400, 200), (220, 130), 0, 0, 360, (0, 0, 0), 3)

        # Case B: Two separate independent speech bubbles with dark borders between them
        cv2.rectangle(canvas, (100, 400), (350, 480), (255, 255, 255), -1)
        cv2.rectangle(canvas, (100, 400), (350, 480), (0, 0, 0), 3)

        cv2.rectangle(canvas, (100, 500), (350, 580), (255, 255, 255), -1)
        cv2.rectangle(canvas, (100, 500), (350, 580), (0, 0, 0), 3)

        labels = compute_bubble_labels(canvas)

        # Inside Case A bubble: Line 1 center at (400, 160), Line 2 center at (400, 240)
        lbl_a1 = labels[160, 400]
        lbl_a2 = labels[240, 400]
        assert lbl_a1 > 0 and lbl_a2 > 0
        assert lbl_a1 == lbl_a2, "Lines in the same speech bubble must share the same closure label"

        # Inside Case B bubbles: Line 1 in top bubble (225, 440), Line 2 in bottom bubble (225, 540)
        lbl_b1 = labels[440, 225]
        lbl_b2 = labels[540, 225]
        assert lbl_b1 > 0 and lbl_b2 > 0
        assert lbl_b1 != lbl_b2, "Lines in distinct speech bubbles must have different closure labels"

    def test_merge_adjacent_boxes_large_bubble_2x_spacing(self):
        """
        Test merge_adjacent_boxes with image containing a large speech bubble where
        lines have 2.0x line height spacing: confirms lines merge into 1 single block with line_count = 2.
        """
        h, w = 600, 800
        canvas = np.full((h, w, 3), 200, dtype=np.uint8)  # Gray background
        # Large white speech bubble with dark border
        cv2.ellipse(canvas, (400, 300), (260, 180), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, (400, 300), (260, 180), 0, 0, 360, (0, 0, 0), 3)

        # Two lines with 2.0x line height spacing inside the bubble
        # line_h = 30, gap = 60px (60 / 30 = 2.0x)
        b1 = {"xmin": 260, "ymin": 210, "xmax": 540, "ymax": 240, "text": "Because you are", "conf": 0.96}
        b2 = {"xmin": 262, "ymin": 300, "xmax": 538, "ymax": 330, "text": "the chosen one.", "conf": 0.94}

        merged = merge_adjacent_boxes([b1, b2], img_w=w, img_h=h, image=canvas)
        assert len(merged) == 1, f"Expected 1 merged block, got {len(merged)}"
        assert merged[0]["line_count"] == 2
        assert merged[0]["text"] == "Because you are\nthe chosen one."
        assert merged[0]["xmin"] == 260
        assert merged[0]["ymin"] == 210
        assert merged[0]["xmax"] == 540
        assert merged[0]["ymax"] == 330

    def test_merge_adjacent_boxes_separated_by_barrier_or_independent_bubbles(self):
        """
        Test merge_adjacent_boxes where lines are separated by a barrier or independent bubbles:
        confirms they stay in 2 distinct blocks.
        """
        h, w = 600, 800
        canvas = np.full((h, w, 3), 200, dtype=np.uint8)  # Gray background

        # Bubble 1 (top): white rectangle with black border
        cv2.rectangle(canvas, (200, 80), (600, 180), (255, 255, 255), -1)
        cv2.rectangle(canvas, (200, 80), (600, 180), (0, 0, 0), 3)

        # Bubble 2 (bottom): white rectangle with black border
        cv2.rectangle(canvas, (200, 220), (600, 320), (255, 255, 255), -1)
        cv2.rectangle(canvas, (200, 220), (600, 320), (0, 0, 0), 3)

        # Line 1 inside Bubble 1 (line_h = 30)
        b1 = {"xmin": 240, "ymin": 110, "xmax": 560, "ymax": 140, "text": "Top bubble dialogue", "conf": 0.95}
        # Line 2 inside Bubble 2 (line_h = 30), separated by 110px and dark bubble borders
        b2 = {"xmin": 240, "ymin": 250, "xmax": 560, "ymax": 280, "text": "Bottom bubble dialogue", "conf": 0.93}

        merged = merge_adjacent_boxes([b1, b2], img_w=w, img_h=h, image=canvas)
        assert len(merged) == 2, f"Expected 2 separate blocks across independent bubbles, got {len(merged)}"
        assert merged[0]["text"] == "Top bubble dialogue"
        assert merged[1]["text"] == "Bottom bubble dialogue"
        assert merged[0]["line_count"] == 1
        assert merged[1]["line_count"] == 1

    def test_qr_code_barrier_prevents_merging(self):
        """
        Test QR code barrier prevention: two text boxes on opposite sides of a QR code
        are never merged together.
        """
        from app.core.ocr.qr_filter import QRRegion

        # QR code placed between two dialogue boxes
        qr_regs = [
            QRRegion(
                kind="qrcode",
                polygon=[[200, 100], [300, 100], [300, 200], [200, 200]],
                bbox=(200, 100, 300, 200),
                bbox_normalized=(20.0, 10.0, 30.0, 20.0),
                decoded_text="https://example.com"
            )
        ]

        b_left = {"xmin": 50, "ymin": 120, "xmax": 180, "ymax": 160, "text": "Support and join us"}
        b_right = {"xmin": 320, "ymin": 120, "xmax": 450, "ymax": 160, "text": "on discord now!"}

        # Verify can_merge_pair directly rejects bridging across QR code
        assert can_merge_pair(b_left, b_right, img_w=1000, img_h=1000, qr_regions=qr_regs) is False

        # Verify merge_adjacent_boxes keeps them separate
        merged = merge_adjacent_boxes([b_left, b_right], img_w=1000, img_h=1000, qr_regions=qr_regs)
        assert len(merged) == 2, "Must not merge boxes bridging across a protected QR code"
        assert merged[0]["text"] == "Support and join us"
        assert merged[1]["text"] == "on discord now!"

    def test_vertical_japanese_multi_column_bubble_loose_spacing(self):
        """
        Test vertical Japanese multi-column bubble with loose column spacing merges properly.
        Right column comes first in reading order, followed by left column.
        """
        h, w = 600, 500
        canvas = np.full((h, w, 3), 210, dtype=np.uint8)  # Gray background
        # Vertical oval speech bubble (center (250, 300), radius_x 160, radius_y 240)
        cv2.ellipse(canvas, (250, 300), (160, 240), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, (250, 300), (160, 240), 0, 0, 360, (0, 0, 0), 3)

        # Two vertical columns inside bubble:
        # Col 1 (Right): x in [290..320] (col_w = 30), y in [150..350]
        # Col 2 (Left):  x in [200..230] (col_w = 30), y in [150..350]
        # Column gap: 290 - 230 = 60px (60 / 30 = 2.0x column width)
        # Default col gap would be 0.70 * 30 = 21px (would NOT merge without bubble awareness)
        b_right = {
            "xmin": 290, "ymin": 150, "xmax": 320, "ymax": 350,
            "text": "お前は", "conf": 0.98
        }
        b_left = {
            "xmin": 200, "ymin": 150, "xmax": 230, "ymax": 350,
            "text": "選ばれし者だ", "conf": 0.96
        }

        # Passed in arbitrary order (e.g. left first, then right)
        merged = merge_adjacent_boxes([b_left, b_right], img_w=w, img_h=h, image=canvas)
        assert len(merged) == 1, f"Expected 1 merged Japanese vertical dialogue block, got {len(merged)}"
        assert merged[0]["line_count"] == 2
        # Japanese reading flow: right column first, newline, left column second
        assert merged[0]["text"] == "お前は\n選ばれし者だ"
