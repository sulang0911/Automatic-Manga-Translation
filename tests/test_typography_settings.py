"""
tests/test_typography_settings.py
Unit tests verifying:
1. Default font is cute manga font ('霞鹜文楷') and bold is enabled by default.
2. StrokeRenderer and TypographyEngine bold font rasterization.
3. PageStyleDialog independent single-page configuration.
4. SettingsDialog typography page and "apply to all" flag.
5. MainWindow single-page and global re-render logic.
"""
import sys
import os
import numpy as np
import pytest

from PyQt6.QtWidgets import QApplication

from app.core.config import AppConfig
from app.core.models import StyleConfig, TranslationBlock, TextColorMode, StrokeMode
from app.core.typography.engine import TypographyEngine
from app.core.typography.stroke_renderer import StrokeRenderer


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_default_typography_models_and_config():
    """Verify cute font and bold defaults across StyleConfig and AppConfig."""
    cfg = StyleConfig()
    assert cfg.font_family == "霞鹜文楷"
    assert cfg.font_bold is True

    app_cfg = AppConfig()
    assert app_cfg.style.font_family == "霞鹜文楷"
    assert app_cfg.style.font_bold is True


def test_typography_engine_bold_rendering():
    """Verify typography engine renders bold text without clipping or exception."""
    engine = TypographyEngine()
    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)

    block = TranslationBlock(
        id="test_b1",
        xmin=10.0,
        ymin=10.0,
        xmax=90.0,
        ymax=90.0,
        original_text="テストです！",
        translated_text="这是可爱的加粗测试文字！",
    )

    style = StyleConfig(
        font_family="霞鹜文楷",
        font_bold=True,
        stroke_mode=StrokeMode.AUTO.value,
        stroke_width=2.5,
    )

    out_img = engine.render_page(canvas, [block], style)
    assert out_img is not None
    assert out_img.shape == (300, 300, 3)
    # Ensure canvas was drawn on (not all white)
    assert not np.all(out_img == 255)


def test_page_style_dialog_lifecycle(qapp):
    """Verify PageStyleDialog creates correctly, handles overrides and resets."""
    from app.ui.settings.page_style_dialog import PageStyleDialog

    global_style = StyleConfig(font_family="霞鹜文楷", font_bold=True, font_size_scale=1.0)
    item_data = {"id": "p1", "path": "sample_page_01.png"}

    # Case 1: Initialized with no override (inherits global)
    dlg = PageStyleDialog(item_data, global_style, current_page_style=None)
    assert dlg.override_cb.isChecked() is False
    assert dlg.bold_cb.isChecked() is True
    assert dlg.font_combo.isEnabled() is False

    # Toggle override
    dlg.override_cb.setChecked(True)
    assert dlg.font_combo.isEnabled() is True
    assert dlg.bold_cb.isEnabled() is True

    # Change settings
    dlg.bold_cb.setChecked(False)
    dlg.scale_slider.setValue(15)  # 1.5x
    dlg._on_apply()

    assert dlg.applied_style is not None
    assert dlg.applied_style.font_bold is False
    assert dlg.applied_style.font_size_scale == 1.5

    # Case 2: Reset to global
    dlg2 = PageStyleDialog(item_data, global_style, current_page_style=dlg.applied_style)
    assert dlg2.override_cb.isChecked() is True
    dlg2._reset_to_global()
    assert dlg2.override_cb.isChecked() is False
    assert dlg2.bold_cb.isChecked() is True
    dlg2._on_apply()
    assert dlg2.applied_style is None  # Inherit global


def test_settings_dialog_typography_page(qapp):
    """Verify SettingsDialog has typography tab, default bold, and apply-all trigger."""
    from app.ui.settings.settings_dialog import SettingsDialog

    app_cfg = AppConfig()
    dlg = SettingsDialog(config=app_cfg)

    # Check navigation items
    items = [dlg.nav_list.item(i).text() for i in range(dlg.nav_list.count())]
    assert "📝 译文文字设置" in items
    typo_index = items.index("📝 译文文字设置")

    # Switch to typography tab
    dlg.nav_list.setCurrentRow(typo_index)
    assert dlg.typo_bold_cb.isChecked() is True
    assert "霞鹜文楷" in dlg.typo_font_combo.currentText()

    # Click apply all
    dlg._on_apply_all()
    assert dlg.re_render_all_requested is True
    assert dlg.config.style.font_bold is True


def test_main_window_re_render_all_and_single_page(qapp, tmp_path):
    """Verify MainWindow re-rendering all pages vs single page isolation."""
    import cv2
    from app.ui.main_window import MainWindow

    win = MainWindow()

    # Create dummy images
    img1_path = str(tmp_path / "page_01.png")
    img2_path = str(tmp_path / "page_02.png")
    cv2.imwrite(img1_path, np.full((200, 200, 3), 255, dtype=np.uint8))
    cv2.imwrite(img2_path, np.full((200, 200, 3), 255, dtype=np.uint8))

    win.page_list.add_paths([img1_path, img2_path])
    assert len(win.page_list.items_data) == 2

    # Provide blocks for both
    b1 = TranslationBlock(id="b1", xmin=10, ymin=10, xmax=90, ymax=90, translated_text="Page 1 Text")
    b2 = TranslationBlock(id="b2", xmin=10, ymin=10, xmax=90, ymax=90, translated_text="Page 2 Text")
    win.page_list.items_data[0]["blocks"] = [b1]
    win.page_list.items_data[1]["blocks"] = [b2]

    # Test single page re-render on page 1 with custom style
    custom_style = StyleConfig(font_family="幼圆", font_bold=False)
    win.page_list.items_data[0]["style"] = custom_style
    win._re_render_single_page(win.page_list.items_data[0], custom_style)
    assert win.page_list.items_data[0].get("translated_img") is not None
    # Page 2 should not be re-rendered yet
    assert win.page_list.items_data[1].get("translated_img") is None

    # Test re-render all pages
    win._re_render_all_pages()
    assert win.page_list.items_data[0].get("translated_img") is not None
    assert win.page_list.items_data[1].get("translated_img") is not None

    win.close()

