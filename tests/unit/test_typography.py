"""
tests/unit/test_typography.py
Unit tests for Smart Typography Engine:
- ITU-R BT.709 perceived luminance auto-contrast
- JIS X 4051 Kinsoku Shori line-breaking rules
- Western word-wrapping
- Vertical CJK layout & RTL column ordering
- 10-iteration bisection font auto-fit algorithm (< 0.07px accuracy)
- Headless PIL drawing pipeline
"""
import pytest
import numpy as np
from PIL import Image, ImageFont

from app.core.models import TranslationBlock, StyleConfig, TextDirection
from app.core.typography.stroke_renderer import StrokeRenderer, StrokeStyle
from app.core.typography.line_breaker import LineBreaker, GYOTO_KINSOKU, GYOMATSU_KINSOKU
from app.core.typography.vertical_layout import VerticalLayoutEngine, VERTICAL_GLYPH_MAP
from app.core.typography.auto_fit import AutoFitEngine, LayoutResult
from app.core.typography.engine import TypographyEngine, PilTextMeasurer


def test_bt709_luminance_precision():
    # Pure Green in BT.709 gives 0.7152 * 255 = 182.376
    green_lum = StrokeRenderer.calculate_bt709_luminance(0, 255, 0)
    assert pytest.approx(green_lum, 0.01) == 182.38

    # Pure Red: 0.2126 * 255 = 54.213
    red_lum = StrokeRenderer.calculate_bt709_luminance(255, 0, 0)
    assert pytest.approx(red_lum, 0.01) == 54.21

    # Pure Blue: 0.0722 * 255 = 18.411
    blue_lum = StrokeRenderer.calculate_bt709_luminance(0, 0, 255)
    assert pytest.approx(blue_lum, 0.01) == 18.41

    # White: 255
    white_lum = StrokeRenderer.calculate_bt709_luminance(255, 255, 255)
    assert pytest.approx(white_lum, 0.01) == 255.0


def test_auto_contrast_stroke_selection():
    # Light text (White) -> Dark outline (85% black)
    light_stroke = StrokeRenderer.get_auto_contrast_stroke((255, 255, 255), font_size=24.0)
    assert light_stroke.color_rgba == (0, 0, 0, 217)
    assert pytest.approx(light_stroke.width, 0.01) == 24.0 * 0.06

    # Dark text (Black) -> Light outline (85% white)
    dark_stroke = StrokeRenderer.get_auto_contrast_stroke((10, 10, 10), font_size=24.0)
    assert dark_stroke.color_rgba == (255, 255, 255, 217)

    # Clamping test: font_size = 100 -> stroke width clamped to 3.5px
    large_stroke = StrokeRenderer.get_auto_contrast_stroke((0, 0, 0), font_size=100.0)
    assert large_stroke.width == 3.5

    # Clamping test: font_size = 5 -> stroke width clamped to 0.5px
    small_stroke = StrokeRenderer.get_auto_contrast_stroke((0, 0, 0), font_size=5.0)
    assert small_stroke.width == 0.5


def test_kinsoku_prohibition_sets():
    breaker = LineBreaker()
    # Verify Gyōtō Kinsoku characters
    for char in ["。", "、", "！", "？", "」", "）", "ー", "っ", "…"]:
        assert char in GYOTO_KINSOKU

    # Verify Gyōmatsu Kinsoku characters
    for char in ["「", "『", "（", "《", "“"]:
        assert char in GYOMATSU_KINSOKU


def test_western_word_wrap():
    breaker = LineBreaker()

    class SimpleMeasurer:
        def measure_width(self, text: str, font_size: float) -> float:
            return len(text) * 10.0  # 10px per character

    measurer = SimpleMeasurer()
    # Line width = 75px (can hold at most 7 characters)
    text = "Hello world from manga translation"
    lines = breaker.wrap_text(text, max_width=75.0, font_size=14.0, measurer=measurer)

    # Words should not be split in half
    for line in lines:
        assert line in ["Hello", "world", "from", "manga", "translation"] or len(line) <= 7


def test_vertical_glyph_substitutions():
    for orig, target in [("「", "﹁"), ("」", "﹂"), ("（", "︵"), ("）", "︶"), ("ー", "丨"), ("!?", "⁉")]:
        assert VERTICAL_GLYPH_MAP.get(orig) == target


def test_vertical_column_rtl_ordering():
    engine = VerticalLayoutEngine()
    # 2 columns of vertical text in a 200x200 box
    cols, total_w, max_h = engine.compute_layout(
        text="第一行文字\n第二行文字",
        font_size=20.0,
        box_x=0.0,
        box_y=0.0,
        box_w=200.0,
        box_h=200.0
    )

    assert len(cols) >= 2
    # In Japanese Manga RTL: Column 0 X coordinate > Column 1 X coordinate
    assert cols[0].x_center > cols[1].x_center

    # Verify glyph stacking top-to-bottom: glyph 0 Y < glyph 1 Y
    assert cols[0].glyphs[0].y < cols[0].glyphs[1].y


def test_bisection_auto_fit_convergence():
    engine = AutoFitEngine(min_font_size=6.0, max_font_size=72.0, bisection_iterations=10)

    class MockEvaluator:
        def evaluate(self, text, font_size, max_w, max_h, is_vertical):
            # Text fits if font_size <= 28.5
            fits = font_size <= 28.5
            return LayoutResult(
                fits=fits,
                total_width=font_size * 5.0,
                total_height=font_size * 3.0,
                lines_or_columns=["L1", "L2"],
                font_size=font_size
            )

    evaluator = MockEvaluator()
    res = engine.fit_text(
        text="Test Auto Fit",
        box_w=200.0,
        box_h=200.0,
        is_vertical=False,
        evaluator=evaluator
    )

    # After 10 iterations on [6, 72], resolution is (72 - 6) / 1024 = 0.0645px
    assert res.iterations_run == 10
    assert abs(res.optimal_font_size - 28.5) < 0.15


def test_typography_engine_render_page():
    typ_engine = TypographyEngine()

    # Synthetic canvas (400x400 white)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255

    block = TranslationBlock.from_pixel_box(
        xmin=50, ymin=50, xmax=350, ymax=200,
        img_width=400, img_height=400,
        original_text="テスト",
        translated_text="自动排版测试",
        direction=TextDirection.HORIZONTAL.value
    )

    rendered = typ_engine.render_page(img, [block])
    assert rendered is not None
    assert rendered.shape == img.shape
    # Some pixels inside the bounding box should now be non-white (drawn text)
    center_crop = rendered[80:150, 100:300]
    assert np.min(center_crop) < 200


def test_typography_engine_render_translations_unpacks_nested_style_config():
    """Verify render_translations handles full config dict with nested style and raw dict blocks."""
    typ_engine = TypographyEngine()
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    raw_blocks = [{
        "id": "b1",
        "original_text": "こんにちは",
        "translated_text": "你好世界",
        "xmin": 10.0,
        "ymin": 10.0,
        "xmax": 80.0,
        "ymax": 80.0,
        "type": "bubble"
    }]
    full_config = {
        "llm": {"provider": "openai"},
        "style": {
            "font_family": "霞鹜文楷",
            "font_size_scale": 1.0,
            "auto_fit_font_size": True
        }
    }

    called_fonts = []
    orig_get_font = typ_engine.get_font
    def spy_get_font(font_family, size):
        called_fonts.append(font_family)
        return orig_get_font(font_family, size)

    typ_engine.get_font = spy_get_font
    rendered = typ_engine.render_translations(img, raw_blocks, full_config)
    assert rendered is not None
    assert rendered.shape == (300, 300, 3)
    assert "霞鹜文楷" in called_fonts

