import pytest
import numpy as np
import cv2
from app.core.ocr.base import can_merge_pair, merge_adjacent_boxes
from desktop.core.ocr_engine import mask_crop_with_lines


class TestBubbleOverlapBleed:
    def test_multiline_block_does_not_swallow_overlapping_outside_box(self):
        b1_bubble = {
            'xmin': 246, 'ymin': 1028, 'xmax': 430, 'ymax': 1136,
            'text': 'Is this new dress also your sisters?',
            'line_count': 3,
            'lines': [
                [[267, 1031], [414, 1031], [414, 1061], [267, 1061]],
                [[268, 1067], [403, 1067], [403, 1097], [268, 1097]],
                [[249, 1103], [427, 1100], [427, 1130], [249, 1133]],
            ],
            'angle': 0.0
        }
        b2_outside = {
            'xmin': 360, 'ymin': 1123, 'xmax': 461, 'ymax': 1171,
            'text': 'qbout the',
            'line_count': 1,
            'angle': 0.0
        }
        assert not can_merge_pair(b1_bubble, b2_outside, 1536, 1536)
        assert not can_merge_pair(b2_outside, b1_bubble, 1536, 1536)

    def test_same_line_outside_fragments_still_merge_properly(self):
        b_what = {
            'xmin': 309, 'ymin': 1148, 'xmax': 370, 'ymax': 1181,
            'text': 'What',
            'line_count': 1,
            'angle': 0.0
        }
        b_about = {
            'xmin': 360, 'ymin': 1146, 'xmax': 461, 'ymax': 1175,
            'text': 'about the',
            'line_count': 1,
            'angle': 0.0
        }
        assert can_merge_pair(b_what, b_about, 1536, 1536)
        merged = merge_adjacent_boxes([b_what, b_about], 1536, 1536)
        assert len(merged) == 1
        assert 'What about the' in merged[0]['text']

    def test_row_box_deduplication_prevents_repeated_words(self):
        b1 = {
            'xmin': 564, 'ymin': 770, 'xmax': 670, 'ymax': 806,
            'text': 'F-Fuckl',
            'conf': 0.95,
            'angle': 0.0
        }
        b2 = {
            'xmin': 563, 'ymin': 768, 'xmax': 672, 'ymax': 806,
            'text': 'F-Fuckl',
            'conf': 0.88,
            'angle': 0.0
        }
        merged = merge_adjacent_boxes([b1, b2], 1536, 1536)
        assert len(merged) == 1
        assert merged[0]['text'] == 'F-Fuckl'
        assert 0.90 <= float(merged[0]['conf']) <= 1.0

    def test_mask_crop_with_lines_clears_corners(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        box = {
            'xmin': 0, 'ymin': 0, 'xmax': 100, 'ymax': 100,
            'lines': [
                [[10, 10], [90, 10], [90, 40], [10, 40]]
            ],
            'bg_color': '#FFFFFF'
        }
        masked = mask_crop_with_lines(crop, box, bx1=0, by1=0)
        assert masked is not None
        assert np.array_equal(masked[80, 80], [255, 255, 255])
        assert np.array_equal(masked[25, 25], [0, 0, 0])

    def test_side_by_side_multiline_columns_do_not_merge_across_panel_gutter(self):
        # Dark CG 2-panel page with a bright white vertical panel divider (x=490..510)
        img = np.full((1000, 1000, 3), 40, dtype=np.uint8)
        img[:, 490:510] = 255

        # Left column (10 lines, x=150..460, y=350..950) and Right column (8 lines, x=530..920, y=620..960)
        left_lines = [[[150, 350 + i * 60], [460, 350 + i * 60], [460, 385 + i * 60], [150, 385 + i * 60]] for i in range(10)]
        right_lines = [[[530, 620 + i * 42], [920, 620 + i * 42], [920, 650 + i * 42], [530, 650 + i * 42]] for i in range(8)]
        b_left = {
            "xmin": 150, "ymin": 350, "xmax": 460, "ymax": 950,
            "text": "\n".join(f"L{i}" for i in range(10)), "conf": 0.95,
            "lines": left_lines, "line_count": 10,
        }
        b_right = {
            "xmin": 530, "ymin": 620, "xmax": 920, "ymax": 960,
            "text": "\n".join(f"R{i}" for i in range(8)), "conf": 0.95,
            "lines": right_lines, "line_count": 8,
        }
        assert not can_merge_pair(b_left, b_right, 1000, 1000, image=img)
        merged = merge_adjacent_boxes([b_left, b_right], img_w=1000, img_h=1000, image=img)
        assert len(merged) == 2

    def test_tight_single_line_continuation_merges_with_multiline_block(self):
        # Simulate top-right 7-line block + flush-left bottom continuation orphan (1 line)
        b_top = {
            "xmin": 100, "ymin": 50, "xmax": 500, "ymax": 330,
            "text": "\n".join(f"Line{i}" for i in range(7)), "conf": 0.95,
            "lines": [[[100, 50 + i * 40], [500, 50 + i * 40], [500, 80 + i * 40], [100, 80 + i * 40]] for i in range(7)],
            "line_count": 7,
        }
        b_orphan = {
            "xmin": 120, "ymin": 335, "xmax": 460, "ymax": 365,
            "text": "OrphanTail", "conf": 0.95,
            "lines": [[[120, 335], [460, 335], [460, 365], [120, 365]]],
            "line_count": 1,
        }
        assert can_merge_pair(b_top, b_orphan, 1000, 1000)
        merged = merge_adjacent_boxes([b_top, b_orphan], img_w=1000, img_h=1000)
        assert len(merged) == 1
        assert merged[0]["line_count"] == 8
