"""
tests/unit/test_models.py
Unit tests for data models, coordinate conversions, and configuration serialization.
"""
import pytest
from app.core.models import (
    TranslationBlock, StyleConfig, MangaPage,
    BlockType, TextDirection, ReadingOrderMode,
    TextColorMode, BgColorMode, StrokeMode, PageStatus
)


def test_translation_block_initialization_and_clamping():
    # Test valid coordinates
    block = TranslationBlock(xmin=10.0, ymin=20.0, xmax=50.0, ymax=60.0, original_text="Hello")
    assert block.xmin == 10.0
    assert block.ymin == 20.0
    assert block.xmax == 50.0
    assert block.ymax == 60.0
    assert block.original_text == "Hello"
    assert block.type == BlockType.BUBBLE.value

    # Test out-of-bounds coordinates clamping to [0.0, 100.0]
    bad_block = TranslationBlock(xmin=-10.0, ymin=150.0, xmax=200.0, ymax=10.0)
    assert bad_block.xmin == 0.0
    assert bad_block.ymin == 100.0
    assert bad_block.xmax == 100.0
    assert bad_block.ymax == 100.0  # ymax is clamped and >= ymin


def test_translation_block_coordinate_conversions():
    # 1000x2000 image dimensions
    block = TranslationBlock.from_pixel_box(
        xmin=100, ymin=200, xmax=500, ymax=800,
        img_width=1000, img_height=2000,
        original_text="Coordinate Test"
    )
    assert block.xmin == 10.0
    assert block.ymin == 10.0
    assert block.xmax == 50.0
    assert block.ymax == 40.0

    rect = block.to_pixel_rect(img_width=1000, img_height=2000)
    assert rect == (100, 200, 400, 600)  # (x, y, w, h)

    box = block.to_pixel_box(img_width=1000, img_height=2000)
    assert box == (100, 200, 500, 800)  # (xmin, ymin, xmax, ymax)


def test_translation_block_center_and_aspect_ratio():
    block = TranslationBlock(xmin=20.0, ymin=10.0, xmax=60.0, ymax=90.0)
    assert block.center_normalized() == (40.0, 50.0)

    # width = 40, height = 80 -> aspect = 0.5
    assert pytest.approx(block.aspect_ratio(), 0.01) == 0.5
    assert block.is_vertical_candidate() is True

    # Horizontal block: width = 60, height = 20
    h_block = TranslationBlock(xmin=10.0, ymin=10.0, xmax=70.0, ymax=30.0)
    assert pytest.approx(h_block.aspect_ratio(), 0.01) == 3.0
    assert h_block.is_vertical_candidate() is False


def test_translation_block_serialization_roundtrip():
    original = TranslationBlock(
        id="blk_1234",
        original_text="こんにちは",
        translated_text="Hello",
        xmin=15.5,
        ymin=25.5,
        xmax=45.5,
        ymax=65.5,
        text_color="#112233",
        bg_color="#EFEFEF",
        font_size_pct=3.5,
        type=BlockType.BUBBLE.value,
        direction=TextDirection.VERTICAL.value,
        font_family_override="SimHei",
        font_size_override=18.0
    )
    data = original.to_dict()
    assert isinstance(data, dict)
    assert data["id"] == "blk_1234"
    assert data["original_text"] == "こんにちは"
    assert data["font_family_override"] == "SimHei"

    reconstructed = TranslationBlock.from_dict(data)
    assert reconstructed.id == original.id
    assert reconstructed.original_text == original.original_text
    assert reconstructed.xmin == original.xmin
    assert reconstructed.font_size_override == original.font_size_override


def test_style_config_defaults_and_serialization():
    style = StyleConfig()
    assert style.font_family == "Microsoft YaHei"
    assert style.auto_fit_font_size is True
    assert style.min_font_size == 8
    assert style.reading_direction == ReadingOrderMode.MANGA_RTL.value

    style_dict = style.to_dict()
    assert "font_family" in style_dict
    assert "line_spacing" in style_dict

    style_dict["font_family"] = "SimSun"
    reconstructed = StyleConfig.from_dict(style_dict)
    assert reconstructed.font_family == "SimSun"


def test_manga_page_model_and_blocks():
    page = MangaPage(
        file_path="C:/manga/ch1/001.png",
        file_name="001.png",
        width=1200,
        height=1800,
        status=PageStatus.PENDING.value
    )
    assert page.status == PageStatus.PENDING.value
    assert len(page.blocks) == 0

    b1 = TranslationBlock(id="b1", original_text="Text 1")
    b2 = TranslationBlock(id="b2", original_text="Text 2")
    page.blocks = [b1, b2]

    d = page.to_dict()
    assert len(d["blocks"]) == 2
    assert d["blocks"][0]["id"] == "b1"

    restored = MangaPage.from_dict(d)
    assert len(restored.blocks) == 2
    assert restored.blocks[0].original_text == "Text 1"
    assert restored.file_name == "001.png"
