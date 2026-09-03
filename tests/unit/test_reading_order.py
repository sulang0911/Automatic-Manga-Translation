"""
tests/unit/test_reading_order.py
Unit tests for 2D spatial reading order algorithms:
- Manga RTL (Right-to-Left, Top-to-Bottom panel tiers)
- Webtoon TTB (Top-to-Bottom continuous vertical flow)
- Western LTR (Left-to-Right, Top-to-Bottom)
"""
import pytest
from app.core.models import TranslationBlock, ReadingOrderMode
from app.core.ocr.reading_order import sort_reading_order


def test_empty_and_single_block_reading_order():
    assert sort_reading_order([]) == []

    single = [TranslationBlock(id="single", xmin=10, ymin=10, xmax=30, ymax=30)]
    result = sort_reading_order(single)
    assert len(result) == 1
    assert result[0].reading_order == 1


def test_manga_rtl_reading_order():
    # 3 bubbles in upper panel tier: Right bubble should come first
    b_left = TranslationBlock(id="left", xmin=10.0, ymin=10.0, xmax=30.0, ymax=25.0)
    b_mid = TranslationBlock(id="mid", xmin=40.0, ymin=12.0, xmax=60.0, ymax=26.0)
    b_right = TranslationBlock(id="right", xmin=70.0, ymin=11.0, xmax=90.0, ymax=24.0)

    # 1 bubble in lower panel tier
    b_lower = TranslationBlock(id="lower", xmin=50.0, ymin=60.0, xmax=80.0, ymax=75.0)

    blocks = [b_left, b_lower, b_right, b_mid]
    ordered = sort_reading_order(blocks, mode=ReadingOrderMode.MANGA_RTL.value)

    # Expected order: right -> mid -> left -> lower
    assert [b.id for b in ordered] == ["right", "mid", "left", "lower"]
    assert [b.reading_order for b in ordered] == [1, 2, 3, 4]


def test_webtoon_ttb_reading_order():
    # Continuous vertical strip: sort strictly by ascending Y, secondary ascending X
    b1 = TranslationBlock(id="b1", xmin=20.0, ymin=5.0, xmax=50.0, ymax=10.0)
    b2 = TranslationBlock(id="b2", xmin=10.0, ymin=30.0, xmax=40.0, ymax=35.0)
    b3 = TranslationBlock(id="b3", xmin=60.0, ymin=15.0, xmax=90.0, ymax=20.0)
    b4 = TranslationBlock(id="b4", xmin=20.0, ymin=30.0, xmax=50.0, ymax=35.0)

    blocks = [b4, b2, b3, b1]
    ordered = sort_reading_order(blocks, mode=ReadingOrderMode.WEBTOON_TTB.value)

    # Expected order: b1 (ymin=5), b3 (ymin=15), b2 (ymin=30, xmin=10), b4 (ymin=30, xmin=20)
    assert [b.id for b in ordered] == ["b1", "b3", "b2", "b4"]
    assert [b.reading_order for b in ordered] == [1, 2, 3, 4]


def test_western_ltr_reading_order():
    # Western comic: Left to Right within horizontal panel tier
    b_left = TranslationBlock(id="w_left", xmin=10.0, ymin=10.0, xmax=30.0, ymax=25.0)
    b_mid = TranslationBlock(id="w_mid", xmin=40.0, ymin=12.0, xmax=60.0, ymax=26.0)
    b_right = TranslationBlock(id="w_right", xmin=70.0, ymin=11.0, xmax=90.0, ymax=24.0)

    blocks = [b_right, b_left, b_mid]
    ordered = sort_reading_order(blocks, mode=ReadingOrderMode.WESTERN_LTR.value)

    assert [b.id for b in ordered] == ["w_left", "w_mid", "w_right"]
    assert [b.reading_order for b in ordered] == [1, 2, 3]


def test_multi_tier_complex_manga_page():
    # Tier 1 (top): right (80%), left (20%)
    t1_r = TranslationBlock(id="t1_r", xmin=70.0, ymin=5.0, xmax=90.0, ymax=18.0)
    t1_l = TranslationBlock(id="t1_l", xmin=10.0, ymin=6.0, xmax=35.0, ymax=20.0)

    # Tier 2 (middle): single narration box
    t2_m = TranslationBlock(id="t2_m", xmin=30.0, ymin=40.0, xmax=70.0, ymax=50.0)

    # Tier 3 (bottom): right (75%), mid (45%), left (15%)
    t3_r = TranslationBlock(id="t3_r", xmin=65.0, ymin=75.0, xmax=85.0, ymax=90.0)
    t3_m = TranslationBlock(id="t3_m", xmin=40.0, ymin=76.0, xmax=60.0, ymax=88.0)
    t3_l = TranslationBlock(id="t3_l", xmin=10.0, ymin=74.0, xmax=30.0, ymax=89.0)

    shuffled = [t3_m, t1_l, t3_r, t2_m, t3_l, t1_r]
    ordered = sort_reading_order(shuffled, mode=ReadingOrderMode.MANGA_RTL.value)

    assert [b.id for b in ordered] == ["t1_r", "t1_l", "t2_m", "t3_r", "t3_m", "t3_l"]
