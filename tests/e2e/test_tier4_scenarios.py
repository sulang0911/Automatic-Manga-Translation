import os
import pytest
import numpy as np
import cv2
from PIL import Image

from desktop.core.batch_worker import BatchWorker
from desktop.core.pipeline_worker import PipelineWorker
from desktop.core.ocr_engine import OCREngine
from desktop.core.inpaint_engine import InpaintEngine
from desktop.core.translation_engine import TranslationEngine
from desktop.core.typography_engine import TypographyEngine
from desktop.ui.main_window import MainWindow

# ============================================================================
# Tier 4: Real-World Scenarios
# ============================================================================

def test_tier4_01_full_chapter_batch_ingestion_and_processing(sample_chapter_dir, temp_dir, monkeypatch):
    """
    Scenario 1: Full chapter folder ingestion.
    - Ingest folder with 3 images and helper files (.txt, .DS_Store).
    - Batch worker processes all pages sequentially.
    - Exports translated images to target directory.
    """
    monkeypatch.setattr(OCREngine, "detect_and_recognize", lambda *args, **kwargs: [
        {"id": "b1", "original_text": "テスト", "translated_text": "", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 30}
    ])
    monkeypatch.setattr(InpaintEngine, "inpaint", lambda self, img, blocks, *args, **kwargs: img)

    # Collect valid images
    images = [
        {"id": f"page_{idx}", "path": os.path.join(sample_chapter_dir, f)}
        for idx, f in enumerate(sorted(os.listdir(sample_chapter_dir)))
        if f.endswith(".png")
    ]
    assert len(images) == 3

    export_dir = os.path.join(temp_dir, "batch_out")
    os.makedirs(export_dir, exist_ok=True)

    worker = BatchWorker(images, {"api_key": ""}, export_dir=export_dir)
    completed_ids = []
    worker.sig_item_completed.connect(lambda img_id, res: completed_ids.append(img_id))
    worker.run()

    assert len(completed_ids) == 3
    exported_pngs = [f for f in os.listdir(export_dir) if f.endswith(".png")]
    assert len(exported_pngs) == 3

def test_tier4_02_interactive_proofreading_and_canvas_update(qapp, sample_manga_image_np):
    """
    Scenario 2: User interactive proofreading.
    - Load manga page.
    - Click bubble to inspect.
    - Edit text and font in inspector.
    - Move bubble bounding box.
    - Canvas re-renders with new text and coordinates.
    """
    win = MainWindow()
    initial_blocks = [
        {
            "id": "proof_1",
            "original_text": "お腹が空いた…",
            "translated_text": "肚子饿了…",
            "xmin": 15.0, "ymin": 15.0, "xmax": 40.0, "ymax": 35.0,
            "bg_color": "#FFFFFF", "text_color": "#000000", "type": "bubble"
        }
    ]
    win.current_image_data = {
        "id": "proof_img",
        "path": "page.png",
        "blocks": initial_blocks,
        "erased_img": sample_manga_image_np.copy(),
        "translated_img": None
    }
    win.canvas_view.set_data(sample_manga_image_np, blocks=initial_blocks)
    win.inspector_panel.set_blocks(initial_blocks)

    # 1. User selects bubble
    win.inspector_panel.select_block_by_id("proof_1")
    assert win.inspector_panel.selected_block["id"] == "proof_1"

    # 2. User modifies translation text
    win.inspector_panel.trans_text_edit.setText("肚子好饿啊，想吃拉面！")
    assert initial_blocks[0]["translated_text"] == "肚子好饿啊，想吃拉面！"

    # 3. User modifies geometry
    bubble_item = win.canvas_view.bubble_items[0]
    bubble_item.setPos(200.0, 250.0)
    bubble_item.itemChange(bubble_item.GraphicsItemChange.ItemPositionHasChanged, bubble_item.pos())

    # 4. Trigger re-render
    win._re_render_current_page()
    assert win.canvas_view.translated_cv is not None
    win.close()

def test_tier4_03_chapter_to_multipage_pdf_compilation(temp_dir, sample_manga_image_np):
    """
    Scenario 3: Chapter compilation into multi-page PDF document.
    - Render 3 pages.
    - Compile into a single PDF volume.
    - Verify with pypdfium2.
    """
    import pypdfium2 as pdfium
    pdf_path = os.path.join(temp_dir, "volume_01.pdf")

    # Generate 3 translated pages
    typo_eng = TypographyEngine()
    pages = []
    for i in range(1, 4):
        blocks = [{"id": f"b{i}", "xmin": 20, "ymin": 20, "xmax": 50, "ymax": 40, "translated_text": f"第 {i} 页内容"}]
        rendered = typo_eng.render_translations(sample_manga_image_np, blocks, {})
        pil_img = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        pages.append(pil_img)

    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:])
    assert os.path.exists(pdf_path)

    pdf = pdfium.PdfDocument(pdf_path)
    assert len(pdf) == 3
    pdf.close()

def test_tier4_04_offline_demo_mode_translation(sample_manga_image_np):
    """
    Scenario 4: Fully offline demo mode translation.
    - No API key configured.
    - Pipeline runs end-to-end using local mock/demo translation.
    - Returns localized dialogue overlay with zero network requests.
    """
    blocks = [
        {"id": "demo1", "original_text": "おはよう", "translated_text": "", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 30}
    ]
    trans_eng = TranslationEngine(api_key="")  # No key
    translated_blocks = trans_eng.translate_blocks(blocks)
    assert "【译】おはよう" in translated_blocks[0]["translated_text"]

    typo_eng = TypographyEngine()
    rendered = typo_eng.render_translations(sample_manga_image_np, translated_blocks, {})
    assert rendered.shape == sample_manga_image_np.shape

def test_tier4_05_batch_processing_cancellation_and_cleanup(sample_chapter_dir, monkeypatch):
    """
    Scenario 5: Batch processing cancellation.
    - Start batch queue with multiple items.
    - Cancel after first item.
    - Worker halts gracefully without crashing.
    """
    monkeypatch.setattr(OCREngine, "detect_and_recognize", lambda *args, **kwargs: [])
    monkeypatch.setattr(InpaintEngine, "inpaint", lambda self, img, blocks, *args, **kwargs: img)

    items = [
        {"id": f"p{i}", "path": os.path.join(sample_chapter_dir, f)}
        for i, f in enumerate(sorted(os.listdir(sample_chapter_dir))) if f.endswith(".png")
    ]
    worker = BatchWorker(items, {"api_key": ""})

    # Cancel immediately
    worker.cancel()
    finished = []
    worker.sig_batch_finished.connect(lambda s, f: finished.append((s, f)))
    worker.run()

    # Success count should be 0 because cancelled immediately
    assert len(finished) == 1
    assert finished[0][0] == 0
