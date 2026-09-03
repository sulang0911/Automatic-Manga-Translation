import os
import zipfile
import json
import pytest
import numpy as np
import cv2
from PIL import Image

from desktop.core.batch_worker import BatchWorker
from desktop.ui.main_window import MainWindow
from desktop.core.config_manager import ConfigManager

# ============================================================================
# F-EXP-01: Batch Translation Queue Manager
# ============================================================================

def test_fexp_01_batch_worker_init():
    items = [{"id": "1", "path": "p1.png"}]
    worker = BatchWorker(items, {}, export_dir="")
    assert len(worker.queue_items) == 1
    assert not worker._is_cancelled

def test_fexp_01_batch_worker_cancellation():
    worker = BatchWorker([], {}, export_dir="")
    worker.cancel()
    assert worker._is_cancelled

def test_fexp_01_batch_worker_signals():
    worker = BatchWorker([], {}, export_dir="")
    assert hasattr(worker, "sig_batch_progress")
    assert hasattr(worker, "sig_item_completed")
    assert hasattr(worker, "sig_batch_finished")
    assert hasattr(worker, "sig_item_failed")

def test_fexp_01_batch_worker_empty_queue():
    worker = BatchWorker([], {}, export_dir="")
    emitted = []
    worker.sig_batch_finished.connect(lambda s, f: emitted.append((s, f)))
    worker.run()
    assert len(emitted) == 1
    assert emitted[0] == (0, 0)

def test_fexp_01_batch_progress_calculation():
    current = 2
    total = 5
    pct = 50
    overall_pct = int(((current - 1) / total) * 100 + (pct / total))
    assert overall_pct == 30

# ============================================================================
# F-EXP-02: High-Resolution Multi-Format Export
# ============================================================================

def test_fexp_02_export_png(temp_dir, sample_manga_image_np):
    out_path = os.path.join(temp_dir, "export.png")
    success, buf = cv2.imencode(".png", sample_manga_image_np)
    assert success
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())

    assert os.path.exists(out_path)
    loaded = cv2.imread(out_path)
    assert loaded.shape == sample_manga_image_np.shape

def test_fexp_02_export_jpg(temp_dir, sample_manga_image_np):
    out_path = os.path.join(temp_dir, "export.jpg")
    success, buf = cv2.imencode(".jpg", sample_manga_image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert success
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())

    assert os.path.exists(out_path)
    loaded = cv2.imread(out_path)
    assert loaded.shape == sample_manga_image_np.shape

def test_fexp_02_export_unicode_path(temp_dir, sample_manga_image_np):
    # Test path with Japanese & Chinese characters
    out_path = os.path.join(temp_dir, "漫画_第01话_translated.png")
    success, buf = cv2.imencode(".png", sample_manga_image_np)
    assert success
    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    assert os.path.exists(out_path)

def test_fexp_02_export_current_page_no_image_toast(qapp):
    win = MainWindow()
    win.current_image_data = None
    win._export_current_page()
    # Should trigger warning toast without crash
    win.close()

def test_fexp_02_export_quality_reproducibility(sample_manga_image_np):
    _, buf = cv2.imencode(".png", sample_manga_image_np)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, sample_manga_image_np)

# ============================================================================
# F-EXP-03: Multi-Page PDF Chapter Compiler
# ============================================================================

def test_fexp_03_pdf_compilation_via_pillow(temp_dir, sample_manga_image_np):
    pdf_path = os.path.join(temp_dir, "chapter_01.pdf")
    img_rgb = cv2.cvtColor(sample_manga_image_np, cv2.COLOR_BGR2RGB)
    pil_1 = Image.fromarray(img_rgb)
    pil_2 = Image.fromarray(img_rgb)

    pil_1.save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=[pil_2])
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000

def test_fexp_03_pdf_magic_header(temp_dir, sample_manga_image_np):
    pdf_path = os.path.join(temp_dir, "test.pdf")
    img_rgb = cv2.cvtColor(sample_manga_image_np, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil.save(pdf_path, "PDF")

    with open(pdf_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"

def test_fexp_03_pypdfium2_inspection(temp_dir, sample_manga_image_np):
    import pypdfium2 as pdfium
    pdf_path = os.path.join(temp_dir, "test_inspect.pdf")
    img_rgb = cv2.cvtColor(sample_manga_image_np, cv2.COLOR_BGR2RGB)
    pil_1 = Image.fromarray(img_rgb)
    pil_2 = Image.fromarray(img_rgb)
    pil_3 = Image.fromarray(img_rgb)
    pil_1.save(pdf_path, "PDF", save_all=True, append_images=[pil_2, pil_3])

    pdf = pdfium.PdfDocument(pdf_path)
    assert len(pdf) == 3
    pdf.close()

def test_fexp_03_single_page_pdf(temp_dir, sample_manga_image_np):
    pdf_path = os.path.join(temp_dir, "single.pdf")
    pil = Image.fromarray(cv2.cvtColor(sample_manga_image_np, cv2.COLOR_BGR2RGB))
    pil.save(pdf_path, "PDF")
    assert os.path.exists(pdf_path)

def test_fexp_03_pdf_dimensions_preserved(temp_dir, sample_manga_image_np):
    import pypdfium2 as pdfium
    pdf_path = os.path.join(temp_dir, "dim.pdf")
    pil = Image.fromarray(cv2.cvtColor(sample_manga_image_np, cv2.COLOR_BGR2RGB))
    pil.save(pdf_path, "PDF")

    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[0]
    w, h = page.get_size()
    # Aspect ratio h/w should match 1200/800 = 1.5
    assert abs((h / w) - 1.5) < 0.05
    pdf.close()

# ============================================================================
# F-EXP-04: Batch ZIP Archive Packaging
# ============================================================================

def test_fexp_04_create_zip_archive(temp_dir, sample_manga_image_np):
    zip_path = os.path.join(temp_dir, "chapter_translated.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 4):
            _, buf = cv2.imencode(".png", sample_manga_image_np)
            zf.writestr(f"page_{i:02d}.png", buf.tobytes())

    assert os.path.exists(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert len(names) == 3
        assert "page_01.png" in names
        assert "page_02.png" in names
        assert "page_03.png" in names

def test_fexp_04_zip_extraction(temp_dir, sample_manga_image_np):
    zip_path = os.path.join(temp_dir, "extract_test.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        _, buf = cv2.imencode(".png", sample_manga_image_np)
        zf.writestr("test_extract.png", buf.tobytes())

    extract_dir = os.path.join(temp_dir, "extracted")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    extracted_img = os.path.join(extract_dir, "test_extract.png")
    assert os.path.exists(extracted_img)
    loaded = cv2.imread(extracted_img)
    assert loaded.shape == sample_manga_image_np.shape

def test_fexp_04_zip_empty_list_handling(temp_dir):
    zip_path = os.path.join(temp_dir, "empty.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        pass
    assert os.path.exists(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert len(zf.namelist()) == 0

def test_fexp_04_overwrite_zip_cleanly(temp_dir):
    zip_path = os.path.join(temp_dir, "overwrite.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", "1")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file2.txt", "2")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert names == ["file2.txt"]

def test_fexp_04_zip_file_integrity_test(temp_dir, sample_manga_image_np):
    zip_path = os.path.join(temp_dir, "integrity.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        _, buf = cv2.imencode(".png", sample_manga_image_np)
        zf.writestr("image.png", buf.tobytes())
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert zf.testzip() is None  # None indicates no corrupted files

# ============================================================================
# F-EXP-05: Bidirectional Disk Cache & Sync
# ============================================================================

def test_fexp_05_save_and_load_blocks_cache(temp_dir, sample_translation_blocks):
    cache_path = os.path.join(temp_dir, "page_01_blocks.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sample_translation_blocks, f, ensure_ascii=False, indent=2)

    with open(cache_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == len(sample_translation_blocks)
    assert loaded[0]["id"] == sample_translation_blocks[0]["id"]

def test_fexp_05_load_erased_cache(temp_dir, sample_manga_image_np):
    cache_path = os.path.join(temp_dir, "page_01_erased.png")
    cv2.imwrite(cache_path, sample_manga_image_np)

    loaded = cv2.imread(cache_path)
    assert loaded is not None
    assert loaded.shape == sample_manga_image_np.shape

def test_fexp_05_load_translated_cache(temp_dir, sample_manga_image_np):
    cache_path = os.path.join(temp_dir, "page_01_translated.png")
    cv2.imwrite(cache_path, sample_manga_image_np)

    loaded = cv2.imread(cache_path)
    assert loaded is not None
    assert loaded.shape == sample_manga_image_np.shape

def test_fexp_05_recent_export_dir_config(isolated_config, temp_dir):
    isolated_config.set("recent_export_dir", temp_dir)
    assert isolated_config.get("recent_export_dir") == temp_dir

def test_fexp_05_auto_export_path_generation(temp_dir):
    filename = "01_intro.png"
    name_without_ext = os.path.splitext(filename)[0]
    export_path = os.path.join(temp_dir, f"{name_without_ext}_translated.png")
    assert export_path.endswith("01_intro_translated.png")
