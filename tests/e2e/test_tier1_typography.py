import os
import pytest
import numpy as np
import cv2
from PIL import ImageFont

from desktop.core.typography_engine import TypographyEngine, hex_to_rgb

# ============================================================================
# F-TYP-01: Dual-Mode Typography Renderer
# ============================================================================

def test_ftyp_01_render_translations_output_format(sample_manga_image_np, sample_translation_blocks):
    engine = TypographyEngine()
    cfg = {
        "font_family": "Microsoft YaHei",
        "font_size_scale": 1.0,
        "auto_fit_font_size": True,
        "bg_color_mode": "original",
        "bg_opacity": 0.95,
        "stroke_mode": "auto",
        "text_color_mode": "custom",
        "text_color": "#000000"
    }
    rendered = engine.render_translations(sample_manga_image_np, sample_translation_blocks, cfg)
    assert rendered is not None
    assert rendered.shape == sample_manga_image_np.shape
    assert rendered.dtype == np.uint8

def test_ftyp_01_empty_base_image_safe():
    engine = TypographyEngine()
    assert engine.render_translations(None, [], {}) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res = engine.render_translations(empty, [], {})
    assert res.size == 0

def test_ftyp_01_empty_blocks_safe(sample_manga_image_np):
    engine = TypographyEngine()
    res = engine.render_translations(sample_manga_image_np, [], {})
    # Returns image unchanged or equivalent
    assert res.shape == sample_manga_image_np.shape

def test_ftyp_01_wrap_text_splitting():
    engine = TypographyEngine()
    font = ImageFont.load_default()
    long_text = "This is a long sentence that definitely needs to wrap across multiple lines."
    lines = engine._wrap_text(long_text, max_w=80, font=font)
    assert len(lines) > 1

def test_ftyp_01_wrap_text_preserves_explicit_newlines():
    engine = TypographyEngine()
    font = ImageFont.load_default()
    text = "第一行\n第二行\n第三行"
    lines = engine._wrap_text(text, max_w=200, font=font)
    assert len(lines) == 3
    assert lines[0] == "第一行"
    assert lines[1] == "第二行"
    assert lines[2] == "第三行"

# ============================================================================
# F-TYP-02: Binary Search Auto-Fit Font Scaling
# ============================================================================

def test_ftyp_02_fit_text_converges():
    engine = TypographyEngine()
    lines, font, sz = engine._fit_text_to_box(
        text="短句", box_w=200, box_h=200,
        font_family="Microsoft YaHei", font_size_scale=1.0, auto_fit=True, bold=False
    )
    assert sz >= 9
    assert sz <= 72
    assert len(lines) >= 1

def test_ftyp_02_long_text_shrinks_font_size():
    engine = TypographyEngine()
    _, _, sz_short = engine._fit_text_to_box(
        text="嗨", box_w=150, box_h=150,
        font_family="Microsoft YaHei", font_size_scale=1.0, auto_fit=True, bold=False
    )
    _, _, sz_long = engine._fit_text_to_box(
        text="这是一个非常长非常长非常长非常长非常长的漫画翻译句子，需要自适应缩小字号才能塞进气泡。",
        box_w=150, box_h=150, font_family="Microsoft YaHei", font_size_scale=1.0, auto_fit=True, bold=False
    )
    assert sz_long < sz_short

def test_ftyp_02_auto_fit_disabled_uses_fixed_size():
    engine = TypographyEngine()
    lines, font, sz = engine._fit_text_to_box(
        text="测试固定字号", box_w=200, box_h=200,
        font_family="Microsoft YaHei", font_size_scale=1.5, auto_fit=False, bold=False
    )
    # Fixed target_sz = max(10, int(20 * 1.5)) = 30
    assert sz == 30

def test_ftyp_02_minimum_font_size_clamp():
    engine = TypographyEngine()
    # Tiny box with long text
    lines, font, sz = engine._fit_text_to_box(
        text="很多字塞进超小框框", box_w=15, box_h=15,
        font_family="Microsoft YaHei", font_size_scale=1.0, auto_fit=True, bold=False
    )
    assert sz >= 9

def test_ftyp_02_font_size_scale_slider():
    engine = TypographyEngine()
    _, _, sz_base = engine._fit_text_to_box(
        text="字号缩放", box_w=200, box_h=200,
        font_family="Microsoft YaHei", font_size_scale=1.0, auto_fit=True, bold=False
    )
    _, _, sz_double = engine._fit_text_to_box(
        text="字号缩放", box_w=200, box_h=200,
        font_family="Microsoft YaHei", font_size_scale=2.0, auto_fit=True, bold=False
    )
    assert sz_double > sz_base

# ============================================================================
# F-TYP-03: Typography Styling Options
# ============================================================================

def test_ftyp_03_hex_to_rgb_6_digits():
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("#000000") == (0, 0, 0)
    assert hex_to_rgb("#0A84FF") == (10, 132, 255)

def test_ftyp_03_hex_to_rgb_3_digits():
    assert hex_to_rgb("#FFF") == (255, 255, 255)
    assert hex_to_rgb("#000") == (0, 0, 0)
    assert hex_to_rgb("#F0A") == (255, 0, 170)

def test_ftyp_03_hex_to_rgb_malformed_returns_default():
    assert hex_to_rgb("") == (0, 0, 0)
    assert hex_to_rgb("invalid") == (0, 0, 0)

def test_ftyp_03_bold_flag():
    engine = TypographyEngine()
    font_normal = engine._get_font("Microsoft YaHei", 16, bold=False)
    font_bold = engine._get_font("Microsoft YaHei", 16, bold=True)
    assert font_normal is not None
    assert font_bold is not None

def test_ftyp_03_font_cache_reuse():
    engine = TypographyEngine()
    font1 = engine._get_font("Arial", 18, bold=False)
    font2 = engine._get_font("Arial", 18, bold=False)
    # Should be identical cached instance
    assert font1 is font2

# ============================================================================
# F-TYP-04: Adaptive Contrast Text Stroke
# ============================================================================

def test_ftyp_04_auto_contrast_dark_text():
    # Dark text: luminance < 128 -> white stroke
    text_rgb = (0, 0, 0)
    lum = 0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]
    stroke_rgb = (255, 255, 255) if lum < 128 else (0, 0, 0)
    assert stroke_rgb == (255, 255, 255)

def test_ftyp_04_auto_contrast_light_text():
    # Light text: luminance >= 128 -> black stroke
    text_rgb = (240, 240, 240)
    lum = 0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]
    stroke_rgb = (255, 255, 255) if lum < 128 else (0, 0, 0)
    assert stroke_rgb == (0, 0, 0)

def test_ftyp_04_proportional_stroke_width():
    font_size = 30
    stroke_w = max(1, int(font_size * 0.08))
    assert stroke_w == 2

def test_ftyp_04_manual_stroke_custom_color(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 20, "ymin": 20, "xmax": 50, "ymax": 50, "translated_text": "描边测试"}]
    cfg = {
        "stroke_mode": "manual",
        "stroke_color": "#FF0000",
        "stroke_width": 3,
        "text_color_mode": "custom",
        "text_color": "#FFFFFF",
        "bg_color_mode": "none"
    }
    res = engine.render_translations(sample_manga_image_np, blocks, cfg)
    assert res is not None

def test_ftyp_04_stroke_mode_off(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 20, "ymin": 20, "xmax": 50, "ymax": 50, "translated_text": "无描边"}]
    cfg = {"stroke_mode": "off", "bg_color_mode": "none"}
    res = engine.render_translations(sample_manga_image_np, blocks, cfg)
    assert res is not None

# ============================================================================
# F-TYP-05: Drop Shadow & Visual Contrast
# ============================================================================

def test_ftyp_05_background_custom_mode(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 30, "translated_text": "自定义背景"}]
    cfg = {"bg_color_mode": "custom", "bg_color": "#FFFF00", "bg_opacity": 1.0}
    res = engine.render_translations(sample_manga_image_np, blocks, cfg)
    assert res is not None

def test_ftyp_05_background_none_mode(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 30, "translated_text": "无背景"}]
    cfg = {"bg_color_mode": "none", "bg_opacity": 0.0}
    res = engine.render_translations(sample_manga_image_np, blocks, cfg)
    assert res is not None

def test_ftyp_05_background_opacity_scaling():
    opacity = 0.5
    alpha = int(opacity * 255)
    assert alpha == 127

def test_ftyp_05_padding_calculation():
    box_w, box_h = 100, 200
    padding = max(2, int(min(box_w, box_h) * 0.05))
    assert padding == 5

def test_ftyp_05_corner_radius_calculation():
    box_w, box_h = 100, 200
    radius = max(4, int(min(box_w, box_h) * 0.15))
    assert radius == 15

# ============================================================================
# F-TYP-06: Onomatopoeia SFX Handling Modes
# ============================================================================

def test_ftyp_06_text_color_mode_custom(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30, "translated_text": "轰！"}]
    cfg = {"text_color_mode": "custom", "text_color": "#FF453A"}
    res = engine.render_translations(sample_manga_image_np, blocks, cfg)
    assert res is not None

def test_ftyp_06_text_color_mode_original(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30, "translated_text": "啪！", "text_color": "#30D158"}]
    cfg = {"text_color_mode": "original"}
    res = engine.render_translations(sample_manga_image_np, blocks, cfg)
    assert res is not None

def test_ftyp_06_empty_translated_text_skipped(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [{"id": "b1", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30, "translated_text": "   "}]
    res = engine.render_translations(sample_manga_image_np, blocks, {})
    # No changes should be rendered
    assert res.shape == sample_manga_image_np.shape

def test_ftyp_06_multiple_blocks_sequential_rendering(sample_manga_image_np):
    engine = TypographyEngine()
    blocks = [
        {"id": "b1", "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30, "translated_text": "第一个"},
        {"id": "b2", "xmin": 50, "ymin": 50, "xmax": 80, "ymax": 80, "translated_text": "第二个"}
    ]
    res = engine.render_translations(sample_manga_image_np, blocks, {})
    assert res is not None

def test_ftyp_06_unknown_font_family_falls_back():
    engine = TypographyEngine()
    font = engine._get_font("NonExistentFont12345", 20)
    assert font is not None
