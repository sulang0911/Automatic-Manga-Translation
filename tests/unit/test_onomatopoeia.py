"""
tests/unit/test_onomatopoeia.py
Unit tests verifying that onomatopoeia blocks (e.g. orange-framed tone words and SFX)
are translated, inpainted, and rendered by default without being hidden.
"""
import numpy as np
import pytest
from app.core.models import StyleConfig, OnomatopoeiaMode, TranslationBlock
from app.core.typography.engine import TypographyEngine
from app.ui.settings.page_style_dialog import PageStyleDialog
from app.ui.settings.settings_dialog import SettingsDialog
from app.core.config import AppConfig


def test_onomatopoeia_default_mode_is_normal():
    """Confirms StyleConfig defaults to OnomatopoeiaMode.NORMAL."""
    cfg = StyleConfig()
    assert cfg.onomatopoeia_mode == OnomatopoeiaMode.NORMAL.value
    assert cfg.onomatopoeia_mode == "normal"


def test_onomatopoeia_rendered_by_default():
    """
    Confirms onomatopoeia text blocks (orange-framed SFX / tone words)
    are rendered onto the image by default instead of being ignored/hidden.
    """
    engine = TypographyEngine()
    # Create white canvas
    canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
    blocks = [
        {
            "id": "sfx_1",
            "xmin": 20.0,
            "ymin": 20.0,
            "xmax": 80.0,
            "ymax": 80.0,
            "type": "onomatopoeia",
            "original_text": "ドキドキ",
            "translated_text": "砰砰心跳",
            "text_color": "#FF0000"
        }
    ]

    # Render with default config (should NOT ignore)
    rendered = engine.render_translations(canvas, blocks)

    # Pixel comparison: some pixels must change from 255, 255, 255 due to rendering
    diff = np.abs(rendered.astype(np.int32) - canvas.astype(np.int32))
    assert np.sum(diff) > 0, "Onomatopoeia text must be rendered and visible by default!"


def test_onomatopoeia_ignored_when_explicitly_configured():
    """
    Confirms that if the user explicitly switches onomatopoeia_mode to 'ignore',
    it is skipped and no pixels in canvas change.
    """
    engine = TypographyEngine()
    canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
    blocks = [
        {
            "id": "sfx_1",
            "xmin": 20.0,
            "ymin": 20.0,
            "xmax": 80.0,
            "ymax": 80.0,
            "type": "onomatopoeia",
            "original_text": "ドキドキ",
            "translated_text": "砰砰心跳",
        }
    ]

    cfg = StyleConfig(onomatopoeia_mode=OnomatopoeiaMode.IGNORE.value)
    rendered = engine.render_translations(canvas, blocks, cfg)

    # No change when explicitly ignored
    diff = np.abs(rendered.astype(np.int32) - canvas.astype(np.int32))
    assert np.sum(diff) == 0, "When onomatopoeia_mode is 'ignore', no text should be rendered."


def test_onomatopoeia_transparent_mode_renders():
    """
    Confirms that 'transparent' mode also renders onomatopoeia.
    """
    engine = TypographyEngine()
    canvas = np.ones((200, 200, 3), dtype=np.uint8) * 255
    blocks = [
        {
            "id": "sfx_1",
            "xmin": 20.0,
            "ymin": 20.0,
            "xmax": 80.0,
            "ymax": 80.0,
            "type": "onomatopoeia",
            "original_text": "ドキドキ",
            "translated_text": "砰砰心跳",
            "text_color": "#000000"
        }
    ]

    cfg = StyleConfig(onomatopoeia_mode=OnomatopoeiaMode.TRANSPARENT.value)
    rendered = engine.render_translations(canvas, blocks, cfg)
    diff = np.abs(rendered.astype(np.int32) - canvas.astype(np.int32))
    assert np.sum(diff) > 0


def test_page_style_dialog_onomatopoeia_combo(qapp):
    """
    Verifies PageStyleDialog includes onomatopoeia configuration
    defaulting to 'normal'.
    """
    dialog = PageStyleDialog(item_data={"path": "page_01.png"}, global_style=StyleConfig())
    assert dialog.onoma_mode_combo is not None
    assert dialog.onoma_mode_combo.currentData() == OnomatopoeiaMode.NORMAL.value

    # Test toggling override and applying
    dialog.override_cb.setChecked(True)
    dialog.onoma_mode_combo.setCurrentIndex(1)  # transparent
    dialog._on_apply()

    assert dialog.applied_style is not None
    assert dialog.applied_style.onomatopoeia_mode == OnomatopoeiaMode.TRANSPARENT.value
    dialog.close()


def test_settings_dialog_onomatopoeia_combo(qapp, tmp_path):
    """
    Verifies SettingsDialog loads and saves onomatopoeia_mode to config.
    """
    cfg = AppConfig()
    cfg.style.onomatopoeia_mode = OnomatopoeiaMode.NORMAL.value
    dialog = SettingsDialog(config=cfg)
    assert dialog.typo_onoma_combo is not None
    assert dialog.typo_onoma_combo.currentData() == OnomatopoeiaMode.NORMAL.value

    # Change to transparent and save
    dialog.typo_onoma_combo.setCurrentIndex(1)
    dialog._save_to_config()
    assert dialog.config.style.onomatopoeia_mode == OnomatopoeiaMode.TRANSPARENT.value
    dialog.close()
