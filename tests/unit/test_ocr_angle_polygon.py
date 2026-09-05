"""
tests/unit/test_ocr_angle_polygon.py
Unit tests for OCR polygon extraction, orientation angle calculation,
TranslationBlock geometric methods, and angle-gated box clustering.
"""
import pytest
import numpy as np
import math
from typing import List

from app.core.models import TranslationBlock
from app.core.ocr.base import calculate_polygon_angle, can_merge_pair, merge_adjacent_boxes


# ==============================================================================
# 1. calculate_polygon_angle Tests
# ==============================================================================

class TestCalculatePolygonAngle:
    """Tests for calculate_polygon_angle with various quadrilaterals and polygons."""

    def test_horizontal_quadrilateral_returns_zero(self):
        """A purely horizontal text line quadrilateral should return 0.0 degrees."""
        # Top-left, top-right, bottom-right, bottom-left
        pts = np.array([
            [50.0, 100.0],
            [250.0, 100.0],
            [250.0, 130.0],
            [50.0, 130.0]
        ], dtype=np.float32)
        angle = calculate_polygon_angle(pts)
        assert angle == 0.0

    def test_clockwise_15_degree_quadrilateral(self):
        """
        In image coordinates (Y down), a clockwise rotation lowers the right end (dy > 0).
        A 15-degree clockwise quadrilateral should return ~15.0 degrees.
        """
        deg = 15.0
        rad = math.radians(deg)
        length = 150.0
        height = 25.0

        x0, y0 = 50.0, 100.0
        dx = length * math.cos(rad)
        dy = length * math.sin(rad)

        # Vector perpendicular to baseline for height (downwards)
        hx = -height * math.sin(rad)
        hy = height * math.cos(rad)

        pts = np.array([
            [x0, y0],
            [x0 + dx, y0 + dy],
            [x0 + dx + hx, y0 + dy + hy],
            [x0 + hx, y0 + hy]
        ], dtype=np.float32)

        angle = calculate_polygon_angle(pts)
        assert abs(angle - 15.0) <= 0.5

    def test_counter_clockwise_15_degree_quadrilateral(self):
        """
        In image coordinates (Y down), a counter-clockwise tilt raises the right end (dy < 0).
        A -15-degree quadrilateral should return ~ -15.0 degrees.
        """
        deg = -15.0
        rad = math.radians(deg)
        length = 150.0
        height = 25.0

        x0, y0 = 60.0, 120.0
        dx = length * math.cos(rad)
        dy = length * math.sin(rad)

        hx = -height * math.sin(rad)
        hy = height * math.cos(rad)

        pts = np.array([
            [x0, y0],
            [x0 + dx, y0 + dy],
            [x0 + dx + hx, y0 + dy + hy],
            [x0 + hx, y0 + hy]
        ], dtype=np.float32)

        angle = calculate_polygon_angle(pts)
        assert abs(angle - (-15.0)) <= 0.5

    def test_45_degree_quadrilateral(self):
        """A 45-degree tilted quadrilateral should return ~45.0 degrees."""
        deg = 45.0
        rad = math.radians(deg)
        length = 100.0
        height = 20.0

        x0, y0 = 100.0, 100.0
        dx = length * math.cos(rad)
        dy = length * math.sin(rad)

        hx = -height * math.sin(rad)
        hy = height * math.cos(rad)

        pts = np.array([
            [x0, y0],
            [x0 + dx, y0 + dy],
            [x0 + dx + hx, y0 + dy + hy],
            [x0 + hx, y0 + hy]
        ], dtype=np.float32)

        angle = calculate_polygon_angle(pts)
        assert abs(angle - 45.0) <= 0.5

    def test_small_angle_noise_deadband(self):
        """Angles with absolute value < 15.0 degrees should be clamped to 0.0."""
        # Tilt angle = 1.5 degrees
        rad = math.radians(1.5)
        length = 100.0
        pts = np.array([
            [50.0, 100.0],
            [50.0 + length * math.cos(rad), 100.0 + length * math.sin(rad)],
            [50.0 + length * math.cos(rad), 125.0 + length * math.sin(rad)],
            [50.0, 125.0]
        ], dtype=np.float32)

        assert calculate_polygon_angle(pts) == 0.0

        # Tilt angle = -2.0 degrees
        rad_neg = math.radians(-2.0)
        pts_neg = np.array([
            [50.0, 100.0],
            [50.0 + length * math.cos(rad_neg), 100.0 + length * math.sin(rad_neg)],
            [50.0 + length * math.cos(rad_neg), 125.0 + length * math.sin(rad_neg)],
            [50.0, 125.0]
        ], dtype=np.float32)

        assert calculate_polygon_angle(pts_neg) == 0.0

        # Tilt angle = 10.0 degrees (below 15.0 degree threshold)
        rad_10 = math.radians(10.0)
        pts_10 = np.array([
            [50.0, 100.0],
            [50.0 + length * math.cos(rad_10), 100.0 + length * math.sin(rad_10)],
            [50.0 + length * math.cos(rad_10), 125.0 + length * math.sin(rad_10)],
            [50.0, 125.0]
        ], dtype=np.float32)

        assert calculate_polygon_angle(pts_10) == 0.0

    def test_general_polygon_fallback(self):
        """Non-4-point polygons (e.g. 6-point contour) should use oriented minAreaRect fallback."""
        # 6-point hexagon-like text contour tilted roughly at 30 degrees
        center_x, center_y = 200.0, 200.0
        rect_w, rect_h = 120.0, 30.0
        rad = math.radians(30.0)

        # Generate 6 points along the rotated rectangle perimeter
        offsets = [
            (-rect_w / 2, -rect_h / 2),
            (0.0, -rect_h / 2),
            (rect_w / 2, -rect_h / 2),
            (rect_w / 2, rect_h / 2),
            (0.0, rect_h / 2),
            (-rect_w / 2, rect_h / 2),
        ]
        pts = []
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        for ox, oy in offsets:
            rx = center_x + ox * cos_t - oy * sin_t
            ry = center_y + ox * sin_t + oy * cos_t
            pts.append([rx, ry])

        pts_arr = np.array(pts, dtype=np.float32)
        angle = calculate_polygon_angle(pts_arr)
        assert abs(angle - 30.0) <= 2.5

    def test_invalid_and_degenerate_inputs(self):
        """Degenerate inputs should safely return 0.0 without exceptions."""
        assert calculate_polygon_angle(None) == 0.0
        assert calculate_polygon_angle([]) == 0.0
        assert calculate_polygon_angle([[10, 10]]) == 0.0
        # Zero length line
        assert calculate_polygon_angle([[10, 10], [10, 10]]) == 0.0


# ==============================================================================
# 2. TranslationBlock to_pixel_polygon & get_effective_angle Tests
# ==============================================================================

class TestTranslationBlockPolygonAngleMethods:
    """Tests for TranslationBlock polygon and angle helper methods."""

    def test_get_effective_angle_precedence(self):
        """get_effective_angle should prioritize angle_override over angle."""
        block = TranslationBlock(angle=-14.5)
        assert block.get_effective_angle() == -14.5

        # When override is set, it takes precedence
        block.angle_override = 18.0
        assert block.get_effective_angle() == 18.0

        # When angle_override is 0.0, it still overrides non-zero angle
        block.angle_override = 0.0
        assert block.get_effective_angle() == 0.0

        # Default block has 0.0 effective angle
        empty_block = TranslationBlock()
        assert empty_block.get_effective_angle() == 0.0

    def test_to_pixel_polygon_with_explicit_polygon(self):
        """When polygon is explicitly provided, to_pixel_polygon returns integer vertices."""
        raw_poly = [[100, 150], [300, 160], [298, 200], [98, 190]]
        block = TranslationBlock(
            xmin=10.0, ymin=15.0, xmax=30.0, ymax=20.0,
            polygon=raw_poly
        )
        pixel_poly = block.to_pixel_polygon(img_w=1000, img_h=1000)
        assert pixel_poly == raw_poly

    def test_to_pixel_polygon_synthesizes_horizontal_box_when_no_polygon(self):
        """When polygon is None and effective angle is near zero, synthesizes 4-corner box."""
        block = TranslationBlock.from_pixel_box(
            xmin=100, ymin=200, xmax=400, ymax=260,
            img_width=1000, img_height=1000,
            angle=0.0
        )
        assert block.polygon is None
        pixel_poly = block.to_pixel_polygon(img_w=1000, img_h=1000)
        assert pixel_poly is not None
        assert len(pixel_poly) == 4
        assert pixel_poly == [[100, 200], [400, 200], [400, 260], [100, 260]]

    def test_to_pixel_polygon_synthesizes_rotated_box_when_angle_present(self):
        """When polygon is None and effective angle is significant, synthesizes rotated rect."""
        block = TranslationBlock.from_pixel_box(
            xmin=200, ymin=300, xmax=500, ymax=360,
            img_width=1000, img_height=1000,
            angle=15.0
        )
        pixel_poly = block.to_pixel_polygon(img_w=1000, img_h=1000)
        assert pixel_poly is not None
        assert len(pixel_poly) == 4
        # Center of synthesized rect should match center of bounding box
        cx = sum(p[0] for p in pixel_poly) / 4.0
        cy = sum(p[1] for p in pixel_poly) / 4.0
        assert abs(cx - 350.0) <= 2.0
        assert abs(cy - 330.0) <= 2.0

    def test_to_pixel_polygon_invalid_dimensions_returns_none(self):
        """Returns None if image dimensions are zero or negative when polygon is missing."""
        block = TranslationBlock(xmin=10.0, ymin=10.0, xmax=50.0, ymax=50.0)
        assert block.to_pixel_polygon(img_w=0, img_h=100) is None
        assert block.to_pixel_polygon(img_w=100, img_h=-10) is None

    def test_serialization_round_trip_preserves_polygon_and_angle(self):
        """to_dict and from_dict preserve polygon, angle, and angle_override."""
        poly = [[50, 60], [180, 70], [178, 110], [48, 100]]
        block = TranslationBlock(
            xmin=5.0, ymin=6.0, xmax=18.0, ymax=11.0,
            polygon=poly,
            angle=-14.2,
            angle_override=12.0
        )
        data = block.to_dict()
        assert data["polygon"] == poly
        assert data["angle"] == -14.2
        assert data["angle_override"] == 12.0

        restored = TranslationBlock.from_dict(data)
        assert restored.polygon == poly
        assert restored.angle == -14.2
        assert restored.angle_override == 12.0
        assert restored.get_effective_angle() == 12.0


# ==============================================================================
# 3. can_merge_pair Angle Gating Tests
# ==============================================================================

class TestCanMergePairAngleGating:
    """Tests for angle mismatch rejection and compatibility in can_merge_pair."""

    def test_compatible_angles_within_10_degrees_can_merge(self):
        """Boxes with |angle1 - angle2| <= 10.0 degrees should be allowed to merge."""
        # Two vertically stacked lines with small angle difference (delta = 5.0 deg <= 10.0)
        b1 = {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125, "text": "Line 1", "angle": -15.0}
        b2 = {"xmin": 100, "ymin": 135, "xmax": 300, "ymax": 160, "text": "Line 2", "angle": -10.0}

        can_merge = can_merge_pair(b1, b2, img_w=1000, img_h=1000)
        assert can_merge is True

    def test_incompatible_angles_exceeding_10_degrees_rejected(self):
        """Boxes with |angle1 - angle2| > 10.0 degrees must be rejected from merging."""
        # Line 1 is tilted at -16.0 deg, Line 2 is horizontal (0.0 deg). Delta = 16.0 > 10.0
        b1 = {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125, "text": "Line 1", "angle": -16.0}
        b2 = {"xmin": 100, "ymin": 135, "xmax": 300, "ymax": 160, "text": "Line 2", "angle": 0.0}

        can_merge = can_merge_pair(b1, b2, img_w=1000, img_h=1000)
        assert can_merge is False

    def test_angle_gating_boundary_condition(self):
        """Exactly 10.0 degrees is permitted, 10.5 degrees is rejected."""
        base = {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125, "text": "L1", "angle": 5.0}

        # Delta = 10.0 -> allowed
        b_at_limit = {"xmin": 100, "ymin": 135, "xmax": 300, "ymax": 160, "text": "L2", "angle": 15.0}
        assert can_merge_pair(base, b_at_limit, img_w=1000, img_h=1000) is True

        # Delta = 10.5 -> rejected
        b_beyond_limit = {"xmin": 100, "ymin": 135, "xmax": 300, "ymax": 160, "text": "L2", "angle": 15.5}
        assert can_merge_pair(base, b_beyond_limit, img_w=1000, img_h=1000) is False

    def test_missing_or_none_angle_backward_compatibility(self):
        """When angle is not specified or None, angle gating is skipped and spatial merge applies."""
        b1 = {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125, "text": "Line 1"}
        b2 = {"xmin": 100, "ymin": 135, "xmax": 300, "ymax": 160, "text": "Line 2"}
        assert can_merge_pair(b1, b2, img_w=1000, img_h=1000) is True

        b1_none = {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125, "text": "Line 1", "angle": None}
        b2_val = {"xmin": 100, "ymin": 135, "xmax": 300, "ymax": 160, "text": "Line 2", "angle": 15.0}
        assert can_merge_pair(b1_none, b2_val, img_w=1000, img_h=1000) is True


# ==============================================================================
# 4. OCR Polygon & Angle Propagation in merge_adjacent_boxes Tests
# ==============================================================================

class TestMergeAdjacentBoxesPropagation:
    """Tests for polygon and angle aggregation during box merging."""

    def test_single_box_preserves_polygon_and_angle(self):
        """A single box should retain its original polygon and angle through merge_adjacent_boxes."""
        poly = [[120, 150], [280, 160], [278, 195], [118, 185]]
        boxes = [{
            "xmin": 118, "ymin": 150, "xmax": 280, "ymax": 195,
            "text": "Single tilted line",
            "angle": 14.5,
            "polygon": poly,
            "conf": 0.95
        }]
        merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
        assert len(merged) == 1
        assert merged[0]["angle"] == 14.5
        assert merged[0]["polygon"] == poly
        assert merged[0]["text"] == "Single tilted line"

    def test_merged_boxes_propagate_median_angle_and_polygon(self):
        """Merging multiple lines computes median angle and synthesizes/aggregates polygon."""
        boxes = [
            {
                "xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125,
                "text": "Upper line",
                "angle": -15.0,
                "polygon": [[100, 100], [300, 110], [298, 135], [98, 125]],
                "conf": 0.98
            },
            {
                "xmin": 102, "ymin": 135, "xmax": 304, "ymax": 160,
                "text": "Lower line",
                "angle": -14.0,
                "polygon": [[102, 135], [304, 145], [302, 170], [100, 160]],
                "conf": 0.94
            }
        ]
        merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
        assert len(merged) == 1
        assert merged[0]["text"] == "Upper line\nLower line"
        assert merged[0]["line_count"] == 2
        # Median of -15.0 and -14.0 is -14.5
        assert abs(merged[0]["angle"] - (-14.5)) <= 0.1
        # Merged polygon must exist and have 4 vertices
        assert "polygon" in merged[0]
        assert merged[0]["polygon"] is not None
        assert len(merged[0]["polygon"]) == 4

    def test_incompatible_angle_boxes_remain_separated(self):
        """Boxes with incompatible angles must not merge into a single TranslationBlock."""
        boxes = [
            {
                "xmin": 100, "ymin": 100, "xmax": 300, "ymax": 125,
                "text": "Tilted Dialogue",
                "angle": -18.0,
                "conf": 0.95
            },
            {
                "xmin": 100, "ymin": 135, "xmax": 300, "ymax": 160,
                "text": "Horizontal Caption",
                "angle": 0.0,
                "conf": 0.92
            }
        ]
        merged = merge_adjacent_boxes(boxes, img_w=1000, img_h=1000)
        # Because |-18.0 - 0.0| = 18.0 > 10.0, they must remain 2 separate blocks
        assert len(merged) == 2
        assert merged[0]["text"] == "Tilted Dialogue"
        assert merged[0]["angle"] == -18.0
        assert merged[1]["text"] == "Horizontal Caption"
        assert merged[1]["angle"] == 0.0
