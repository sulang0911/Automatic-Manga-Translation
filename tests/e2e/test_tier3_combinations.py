import os
import pytest
import numpy as np
import cv2
from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine
from desktop.core.translation_engine import TranslationEngine
from desktop.core.typography_engine import TypographyEngine
from desktop.core.config_manager import ConfigManager
from desktop.ui.main_window import MainWindow
from desktop.ui.canvas_view import CanvasView

# ============================================================================
# Tier 3: Cross-Feature Combinations (Pairwise Interactions)
# ============================================================================

def test_tier3_01_ocr_to_inpaint_pipeline(sample_manga_image_np):
    """Pairwise Interaction 1: OCR output blocks directly feed into InpaintEngine."""
    ocr_eng = OCREngine(use_gpu=False)
    # Mock OCR output to avoid downloading models
    detected_blocks = [
        {
            "id": "combo1", "original_text": "セリフ", "translated_text": "",
            "xmin": 12.5, "ymin": 12.5, "xmax": 43.75, "ymax": 37.5,
            "bg_color": "#FFFFFF", "text_color": "#000000", "type": "bubble"
        }
    ]
    inpaint_eng = InpaintEngine(mode="opencv_telea")
    erased = inpaint_eng.inpaint(sample_manga_image_np, detected_blocks)
    assert erased is not None
    assert erased.shape == sample_manga_image_np.shape

def test_tier3_02_inpaint_to_typography_pipeline(sample_manga_image_np, sample_translation_blocks):
    """Pairwise Interaction 2: Inpainted background serves as canvas for typography."""
    inpaint_eng = InpaintEngine(mode="opencv_telea")
    erased = inpaint_eng.inpaint(sample_manga_image_np, sample_translation_blocks)

    typo_eng = TypographyEngine()
    rendered = typo_eng.render_translations(erased, sample_translation_blocks, {"bg_color_mode": "original"})
    assert rendered is not None
    assert rendered.shape == sample_manga_image_np.shape

def test_tier3_03_translation_to_typography_pipeline(sample_manga_image_np):
    """Pairwise Interaction 3: Translation output fed directly into typography layout."""
    trans_eng = TranslationEngine(api_key="")  # Demo mode
    raw_blocks = [{"id": "t1", "original_text": "おはよう", "translated_text": "", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 30}]
    translated_blocks = trans_eng.translate_blocks(raw_blocks)
    assert translated_blocks[0]["translated_text"] != ""

    typo_eng = TypographyEngine()
    rendered = typo_eng.render_translations(sample_manga_image_np, translated_blocks, {})
    assert rendered is not None

def test_tier3_04_inspector_live_edit_to_canvas_rerender(qapp, sample_manga_image_np, sample_translation_blocks):
    """Pairwise Interaction 4: Inspector edit modifies data and triggers canvas re-render."""
    win = MainWindow()
    win.current_image_data = {
        "id": "combo4",
        "path": "test.png",
        "blocks": sample_translation_blocks,
        "erased_img": sample_manga_image_np.copy(),
        "translated_img": None
    }
    # User edits translation in inspector
    win.inspector_panel.set_blocks(sample_translation_blocks)
    win.inspector_panel.select_block_by_id("b1001")
    win.inspector_panel.trans_text_edit.setText("测试联动重绘")

    # Verify canvas translated image updated
    assert win.current_image_data["translated_img"] is not None
    assert win.canvas_view.translated_cv is not None
    win.close()

def test_tier3_05_queue_selection_to_canvas_display(qapp, sample_manga_image_file, sample_manga_image_np):
    """Pairwise Interaction 5: Selecting queue item displays it on canvas."""
    win = MainWindow()
    win.queue_panel.add_paths([sample_manga_image_file])
    # Item should be auto-selected
    assert win.current_image_data is not None
    assert win.canvas_view.original_cv is not None
    assert win.canvas_view.original_cv.shape == sample_manga_image_np.shape
    win.close()

def test_tier3_06_settings_to_typography_config_propagation(qapp, sample_manga_image_np, sample_translation_blocks, isolated_config):
    """Pairwise Interaction 6: ConfigManager updates affect typography rendering style."""
    cfg = isolated_config
    cfg.set("text_color", "#0A84FF")
    cfg.set("font_size_scale", 1.8)

    typo_eng = TypographyEngine()
    rendered = typo_eng.render_translations(sample_manga_image_np, sample_translation_blocks, cfg.data)
    assert rendered is not None
    assert rendered.shape == sample_manga_image_np.shape

def test_tier3_07_split_slider_and_bubble_overlays(qapp, sample_manga_image_np, sample_translation_blocks):
    """Pairwise Interaction 7: Split-slider mode maintains bubble overlay positions."""
    canvas = CanvasView()
    canvas.set_data(
        sample_manga_image_np,
        translated_cv=sample_manga_image_np.copy(),
        blocks=sample_translation_blocks
    )
    canvas.set_view_mode("split_slider")
    canvas.set_split_position(0.7)

    assert canvas.view_mode == "split_slider"
    assert canvas.split_position == 0.7
    assert len(canvas.bubble_items) == len(sample_translation_blocks)
    canvas.close()

def test_tier3_08_batch_queue_to_file_exporter(temp_dir, sample_manga_image_file, monkeypatch):
    """Pairwise Interaction 8: Batch worker processes queue and auto-exports images."""
    from desktop.core.batch_worker import BatchWorker
    monkeypatch.setattr(OCREngine, "detect_and_recognize", lambda *args, **kwargs: [
        {"id": "b1", "original_text": "a", "translated_text": "译", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 30}
    ])
    monkeypatch.setattr(InpaintEngine, "inpaint", lambda self, img, blocks, *args, **kwargs: img)

    items = [{"id": "item_export", "path": sample_manga_image_file}]
    worker = BatchWorker(items, {"api_key": ""}, export_dir=temp_dir)
    worker.run()

    exported_files = [f for f in os.listdir(temp_dir) if f.endswith("_translated.png")]
    assert len(exported_files) >= 1

def test_tier3_09_pipeline_error_to_toast_notification(qapp):
    """Pairwise Interaction 9: Pipeline worker error automatically shows toast."""
    win = MainWindow()
    win.current_image_data = {"id": "err_test", "path": "test.png"}
    win._on_worker_error("联动错误提示")
    assert "联动错误提示" in win.toast.msg_label.text()
    win.close()

def test_tier3_10_headless_cpu_ocr_and_telea_inpaint_fallback(sample_manga_image_np):
    """Pairwise Interaction 10: Graceful combination of CPU OCR and OpenCV inpainting."""
    ocr_eng = OCREngine(use_gpu=False, engine_type="paddle")
    ocr_eng._paddle_ocr = type("MockPaddle", (), {"ocr": lambda s, img: []})()
    blocks = ocr_eng.detect_and_recognize(sample_manga_image_np)

    inpaint_eng = InpaintEngine(mode="opencv_telea")
    erased = inpaint_eng.inpaint(sample_manga_image_np, blocks)
    assert erased.shape == sample_manga_image_np.shape
