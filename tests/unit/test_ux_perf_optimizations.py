"""
tests/unit/test_ux_perf_optimizations.py
Regression & verification tests covering P0, P1, and P2 UX/performance optimizations:
- P0: In-memory single-page PDF export, ZIP_STORED compressed images, selective canvas pixmap decode,
      and double-rerender suppression.
- P1: Coalescing async cache writes with linearizable flush/read, shared erased snapshot memory,
      and private bounded AsyncThumbnailManager pool (400 LRU).
- P2: Lazy PyTorch imports at UI startup, batched directory cache index, streaming PyMuPDF PDF
      export, lazy-init + parallel cache-only BatchWorker export, and async PageListWidget ingestion.
"""
import os
import sys
import json
import zipfile
import subprocess
import numpy as np
import cv2
import pytest
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThreadPool, QEventLoop, QTimer

from app.core.cache.cache_manager import MangaCacheManager, safe_cv2_imwrite
from app.core.models import TranslationBlock, StyleConfig
from app.core.pipeline.exporter import MangaExporter
from app.core.undo_manager import PageSnapshot, UndoManager
from app.ui.canvas.view import MangaCanvasView
from app.ui.widgets.thumbnail_loader import AsyncThumbnailManager
from app.ui.sidebar.page_list import PageListWidget
from app.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_sample_image(path: str, w: int = 80, h: int = 100, color=(200, 150, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    ext = os.path.splitext(path)[1].lower() or ".png"
    safe_cv2_imwrite(path, img, ext=ext)
    return img


# ============================================================================
# P0 Tests
# ============================================================================

def test_p0_export_single_image_pdf_in_memory(tmp_path):
    import pymupdf

    img = np.full((120, 90, 3), 180, dtype=np.uint8)
    out_pdf = str(tmp_path / "single.pdf")
    assert MangaExporter.export_single_image_pdf(img, out_pdf) is True
    assert os.path.exists(out_pdf)

    doc = pymupdf.open(out_pdf)
    try:
        assert doc.page_count == 1
    finally:
        doc.close()


def test_p0_package_chapter_zip_uses_stored_for_images(tmp_path):
    p_png = str(tmp_path / "p01.png")
    p_webp = str(tmp_path / "p02.webp")
    p_txt = str(tmp_path / "notes.txt")
    _make_sample_image(p_png)
    _make_sample_image(p_webp)
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write("hello " * 100)

    out_zip = str(tmp_path / "chapter.zip")
    assert MangaExporter.package_chapter_zip([p_png, p_webp, p_txt], out_zip) is True

    with zipfile.ZipFile(out_zip, "r") as zf:
        by_name = {info.filename: info for info in zf.infolist()}
        assert by_name["p01.png"].compress_type == zipfile.ZIP_STORED
        assert by_name["p02.webp"].compress_type == zipfile.ZIP_STORED
        assert by_name["notes.txt"].compress_type == zipfile.ZIP_DEFLATED


def test_p0_update_translated_image_skips_unused_pixmaps(qapp, monkeypatch):
    import app.ui.canvas.view as view_mod

    view = MangaCanvasView()
    orig = np.zeros((64, 64, 3), dtype=np.uint8)
    erased = np.ones((64, 64, 3), dtype=np.uint8) * 128
    trans1 = np.ones((64, 64, 3), dtype=np.uint8) * 200
    trans2 = np.ones((64, 64, 3), dtype=np.uint8) * 220

    view.set_data(original_cv=orig, translated_cv=trans1, erased_cv=erased, blocks=[])
    view.set_view_mode("translated")

    calls = []
    real_conv = view_mod.cvimg_to_qpixmap

    def _counted_conv(arr):
        calls.append(arr.shape)
        return real_conv(arr)

    monkeypatch.setattr(view_mod, "cvimg_to_qpixmap", _counted_conv)
    view.update_translated_image(trans2)

    assert len(calls) == 1
    view.close()


def test_p0_rerender_suppresses_pending_debounce_timer(qapp, tmp_path):
    p = str(tmp_path / "p01.png")
    img = _make_sample_image(p)
    win = MainWindow()
    try:
        blk = TranslationBlock(id="b1", original_text="hello", translated_text="你好", xmin=10, ymin=10, xmax=50, ymax=50)
        win.current_image_data = {
            "id": "id_1",
            "path": p,
            "original_img": img,
            "erased_img": img.copy(),
            "translated_img": None,
            "blocks": [blk],
            "style": StyleConfig(),
        }
        win.canvas_view.set_data(original_cv=img, erased_cv=img.copy(), blocks=[blk])

        win._on_block_updated_from_inspector({"id": "b1", "translated_text": "新文本"})
        assert win._rerender_timer.isActive()
        assert win.current_image_data.get("_dirty_blocks") is True

        win._re_render_current_page(async_cache_save=True)
        assert not win._rerender_timer.isActive()
        assert not win.current_image_data.get("_dirty_blocks", False)
    finally:
        win.close()


# ============================================================================
# P1 Tests
# ============================================================================

def test_p1_async_cache_write_coalesces_and_linearizes(tmp_path):
    img_path = str(tmp_path / "page_01.png")
    img = _make_sample_image(img_path)
    mgr = MangaCacheManager()

    for step in ("v1", "v2", "v3"):
        b = TranslationBlock(id="b1", original_text="hi", translated_text=step, xmin=10, ymin=10, xmax=40, ymax=40)
        mgr.save_page_cache_async(img_path, erased_img=img, blocks=[b], rendered_img=img)

    loaded = mgr.load_page_cache(img_path, load_images=False)
    assert len(loaded["blocks"]) == 1
    assert loaded["blocks"][0].translated_text == "v3"


def test_p1_snapshot_copy_erased_false_shares_memory():
    erased = np.zeros((32, 32, 3), dtype=np.uint8)
    blocks = [TranslationBlock(id="b1", original_text="a", translated_text="b")]

    snap_shared = PageSnapshot.create("p1.png", blocks, erased_img=erased, copy_erased=False)
    assert snap_shared.erased_img is erased

    snap_copied = PageSnapshot.create("p1.png", blocks, erased_img=erased, copy_erased=True)
    assert snap_copied.erased_img is not erased

    mgr = UndoManager(max_depth=5)
    mgr.push(snap_shared)
    curr = PageSnapshot.create("p1.png", blocks, erased_img=erased, copy_erased=False)
    undone = mgr.undo(curr)
    assert undone is not None
    assert undone.erased_img is erased
    redone = mgr.redo(undone)
    assert redone is not None
    assert redone.erased_img is erased


def test_p1_thumbnail_manager_private_pool(qapp):
    mgr = AsyncThumbnailManager.instance()
    assert mgr._pool is not QThreadPool.globalInstance()
    assert mgr._max_cache == 400
    assert mgr._pool.maxThreadCount() >= 2


# ============================================================================
# P2 Tests
# ============================================================================

def test_p2_lazy_imports_do_not_load_torch():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    code = (
        "import sys; "
        "import app.ui.main_window; "
        "assert 'torch' not in sys.modules, 'torch eagerly imported'; "
        "assert 'easyocr' not in sys.modules, 'easyocr eagerly imported'; "
        "assert 'paddle' not in sys.modules, 'paddle eagerly imported'; "
        "print('OK')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"
    assert "OK" in res.stdout


def test_p2_batch_cache_status_matches_individual(tmp_path):
    p_done = str(tmp_path / "p01.png")
    p_erased = str(tmp_path / "p02.png")
    p_none = str(tmp_path / "p03.png")
    img = _make_sample_image(p_done)
    _make_sample_image(p_erased)
    _make_sample_image(p_none)

    mgr = MangaCacheManager()
    blk = TranslationBlock(id="b1", original_text="hello", translated_text="你好", xmin=10, ymin=10, xmax=40, ymax=40)
    mgr.save_page_cache(
        p_done,
        erased_img=img,
        blocks=[blk],
        rendered_img=img,
        page_style=StyleConfig(max_font_size=24, line_spacing=1.5),
    )
    mgr.save_page_cache(p_erased, erased_img=img)

    batch = mgr.batch_cache_status([p_done, p_erased, p_none])
    assert batch[p_done]["fully_translated"] is True
    assert batch[p_done]["erased"] is True
    assert batch[p_done]["rendered"] is True
    assert batch[p_done]["page_style"]["max_font_size"] == 24

    assert batch[p_erased]["fully_translated"] is False
    assert batch[p_erased]["erased"] is True
    assert batch[p_erased]["blocks"] is False

    assert batch[p_none]["fully_translated"] is False
    assert batch[p_none]["erased"] is False


def test_p2_compile_chapter_pdf_streaming_mixed_formats(tmp_path):
    import pymupdf

    p1 = str(tmp_path / "page1.jpg")
    p2 = str(tmp_path / "page2.png")
    p10 = str(tmp_path / "page10.webp")
    _make_sample_image(p1, w=100, h=100)
    _make_sample_image(p2, w=100, h=200)
    _make_sample_image(p10, w=100, h=300)

    out_pdf = str(tmp_path / "chapter.pdf")
    assert MangaExporter.compile_chapter_pdf([p10, p2, p1], out_pdf) is True

    doc = pymupdf.open(out_pdf)
    try:
        assert doc.page_count == 3
        heights = [round(doc[i].rect.height) for i in range(3)]
        assert heights[0] < heights[1] < heights[2]
    finally:
        doc.close()


@patch("app.core.pipeline.batch_worker.TranslationManager")
@patch("app.core.pipeline.batch_worker.InpaintEngine")
@patch("app.core.pipeline.batch_worker.OCREngine")
def test_p2_batch_worker_skips_ml_init_on_cache_export(
    mock_ocr_cls, mock_inpaint_cls, mock_trans_cls, tmp_path
):
    from app.core.pipeline.batch_worker import BatchWorker

    mgr = MangaCacheManager()
    queue_items = []
    for i in range(1, 5):
        p = str(tmp_path / f"page_{i:02d}.png")
        img = _make_sample_image(p)
        blk = TranslationBlock(id="b1", original_text="hi", translated_text="你好", xmin=10, ymin=10, xmax=30, ymax=30)
        mgr.save_page_cache(p, erased_img=img, blocks=[blk], rendered_img=img)
        queue_items.append({"id": f"id_{i}", "path": p})

    export_dir = str(tmp_path / "out")
    os.makedirs(export_dir, exist_ok=True)
    worker = BatchWorker(queue_items, {"ocr_engine": "ctd"}, export_dir=export_dir)

    finished = []
    worker.sig_batch_finished.connect(lambda s, f: finished.append((s, f)))
    worker.run()

    assert finished == [(4, 0)]
    assert mock_ocr_cls.call_count == 0
    assert mock_inpaint_cls.call_count == 0
    assert mock_trans_cls.get_instance.call_count == 0
    for i in range(1, 5):
        assert os.path.exists(os.path.join(export_dir, f"page_{i:02d}.png"))


def test_p2_page_list_add_paths_async(qapp, tmp_path):
    p1 = str(tmp_path / "page_1.png")
    p2 = str(tmp_path / "page_10.png")
    p3 = str(tmp_path / "page_2.png")
    _make_sample_image(p1)
    _make_sample_image(p2)
    _make_sample_image(p3)

    zip_img = str(tmp_path / "in_zip.png")
    _make_sample_image(zip_img)
    zip_path = str(tmp_path / "extra.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(zip_img, arcname="page_3.png")
    os.remove(zip_img)

    sub = tmp_path / "chapter"
    sub.mkdir()
    os.replace(p1, str(sub / "page_1.png"))
    os.replace(p2, str(sub / "page_10.png"))
    os.replace(p3, str(sub / "page_2.png"))

    widget = PageListWidget()
    loop = QEventLoop()
    results = []

    def _done(cnt):
        results.append(cnt)
        loop.quit()

    QTimer.singleShot(5000, loop.quit)
    widget.add_paths_async([str(sub), zip_path], on_finished=_done)
    loop.exec()

    try:
        assert results == [4]
        assert len(widget.items_data) == 4
    finally:
        widget.cleanup_temp_dirs()
        widget.close()


def test_p2_chapter_archive_export_worker_pdf_and_zip(qapp, tmp_path):
    import pymupdf
    from app.ui.main_window import _ChapterArchiveExportWorker

    mgr = MangaCacheManager()
    items = []
    for i in (1, 2):
        p = str(tmp_path / f"p_{i}.png")
        img = _make_sample_image(p, w=60, h=80 * i)
        blk = TranslationBlock(id="b1", original_text="a", translated_text="乙", xmin=10, ymin=10, xmax=30, ymax=30)
        mgr.save_page_cache(p, erased_img=img, blocks=[blk], rendered_img=img)
        items.append({"id": f"id_{i}", "path": p, "status": "completed"})

    out_pdf = str(tmp_path / "out.pdf")
    w_pdf = _ChapterArchiveExportWorker(items, {}, out_pdf, "pdf")
    pdf_res = []
    w_pdf.sig_finished.connect(lambda ok, msg: pdf_res.append((ok, msg)))
    w_pdf.run()
    assert pdf_res and pdf_res[0][0] is True
    doc = pymupdf.open(out_pdf)
    try:
        assert doc.page_count == 2
    finally:
        doc.close()

    out_zip = str(tmp_path / "out.zip")
    w_zip = _ChapterArchiveExportWorker(items, {}, out_zip, "zip")
    zip_res = []
    w_zip.sig_finished.connect(lambda ok, msg: zip_res.append((ok, msg)))
    w_zip.run()
    assert zip_res and zip_res[0][0] is True
    with zipfile.ZipFile(out_zip, "r") as zf:
        assert len(zf.namelist()) == 2
        assert all(not name.lower().endswith(".bmp") for name in zf.namelist())

    # After invalidating rendered cache (e.g. global typography change), multi-threaded
    # _ChapterArchiveExportWorker must rebuild in strict page order and still keep ZIP entries as PNG.
    mgr.invalidate_rendered_cache(items[0]["path"])
    mgr.invalidate_rendered_cache(items[1]["path"])

    out_pdf_re = str(tmp_path / "out_re.pdf")
    w_pdf_re = _ChapterArchiveExportWorker(items, {}, out_pdf_re, "pdf")
    pdf_re_res = []
    w_pdf_re.sig_finished.connect(lambda ok, msg: pdf_re_res.append((ok, msg)))
    w_pdf_re.run()
    assert pdf_re_res and pdf_re_res[0][0] is True
    doc_re = pymupdf.open(out_pdf_re)
    try:
        assert doc_re.page_count == 2
        assert round(doc_re[0].rect.height) < round(doc_re[1].rect.height)
    finally:
        doc_re.close()

    # Clear again and test ZIP rebuild directly from BMP erased cache -> ZIP must contain PNG, never BMP
    mgr.invalidate_rendered_cache(items[0]["path"])
    mgr.invalidate_rendered_cache(items[1]["path"])
    out_zip_re = str(tmp_path / "out_re.zip")
    w_zip_re = _ChapterArchiveExportWorker(items, {}, out_zip_re, "zip")
    zip_re_res = []
    w_zip_re.sig_finished.connect(lambda ok, msg: zip_re_res.append((ok, msg)))
    w_zip_re.run()
    assert zip_re_res and zip_re_res[0][0] is True
    with zipfile.ZipFile(out_zip_re, "r") as zf:
        names = sorted(zf.namelist())
        assert len(names) == 2
        assert all(n.lower().endswith(".png") for n in names)


def test_chapter_zip_preserves_source_image_formats(qapp, tmp_path):
    """ZIP/CBZ chapter export preserves each page's original format like batch export."""
    from app.ui.main_window import _ChapterArchiveExportWorker

    mgr = MangaCacheManager()
    orig_fmt = mgr.get_cache_format()
    try:
        mgr.set_cache_format("bmp")
        exts = [".jpg", ".jpeg", ".webp", ".png", ".bmp"]
        items = []
        for idx, ext in enumerate(exts, start=1):
            p = str(tmp_path / f"page_{idx}{ext}")
            img = _make_sample_image(p, w=48, h=64)
            blk = TranslationBlock(id=f"b{idx}", original_text="a", translated_text="译", xmin=6, ymin=6, xmax=28, ymax=28)
            mgr.save_page_cache(p, erased_img=img, blocks=[blk], rendered_img=img)
            items.append({"id": f"id_{idx}", "path": p, "status": "completed"})

        # Inject a legacy .rendered.webp alongside page_1.jpg (simulating old cache) to verify
        # ZIP still emits .jpg rather than leaking stale .webp extension.
        mgr.invalidate_rendered_cache(items[0]["path"])
        mgr.set_cache_format("webp")
        mgr.save_page_cache(items[0]["path"], rendered_img=_make_sample_image(items[0]["path"], w=48, h=64))
        mgr.set_cache_format("bmp")

        out_zip = str(tmp_path / "mixed_cached.zip")
        worker = _ChapterArchiveExportWorker(items, {}, out_zip, "zip")
        res = []
        worker.sig_finished.connect(lambda ok, msg: res.append((ok, msg)))
        worker.run()
        assert res and res[0][0] is True

        expected_names = [f"{i:04d}_page_{i}{ext}" for i, ext in enumerate(exts, start=1)]
        with zipfile.ZipFile(out_zip, "r") as zf:
            infos = {info.filename: info for info in zf.infolist()}
            assert sorted(infos.keys()) == sorted(expected_names)
            assert infos["0005_page_5.bmp"].compress_type == zipfile.ZIP_DEFLATED
            for name in expected_names[:-1]:
                assert infos[name].compress_type == zipfile.ZIP_STORED
            for name in expected_names:
                decoded = cv2.imdecode(np.frombuffer(zf.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
                assert decoded is not None and decoded.shape[:2] == (64, 48)

        # Invalidate rendered cache and rebuild from erased BMP:
        # 1) export_compressed=False -> preserves original formats (.jpg/.jpeg/.webp/.png/.bmp)
        for it in items:
            mgr.invalidate_rendered_cache(it["path"])

        out_zip_rebuilt = str(tmp_path / "mixed_rebuilt.zip")
        worker_re = _ChapterArchiveExportWorker(
            items, {"style": {"export_compressed": False}}, out_zip_rebuilt, "zip"
        )
        res_re = []
        worker_re.sig_finished.connect(lambda ok, msg: res_re.append((ok, msg)))
        worker_re.run()
        assert res_re and res_re[0][0] is True

        with zipfile.ZipFile(out_zip_rebuilt, "r") as zf:
            assert sorted(zf.namelist()) == sorted(expected_names)
            for name in expected_names:
                decoded = cv2.imdecode(np.frombuffer(zf.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
                assert decoded is not None and decoded.shape[:2] == (64, 48)

        # 2) export_compressed=True -> fixed export format becomes .webp for every entry
        expected_webp_names = [f"{i:04d}_page_{i}.webp" for i in range(1, len(exts) + 1)]
        out_zip_comp = str(tmp_path / "mixed_compressed_webp.zip")
        worker_comp = _ChapterArchiveExportWorker(
            items, {"style": {"export_compressed": True}}, out_zip_comp, "zip"
        )
        res_comp = []
        worker_comp.sig_finished.connect(lambda ok, msg: res_comp.append((ok, msg)))
        worker_comp.run()
        assert res_comp and res_comp[0][0] is True

        with zipfile.ZipFile(out_zip_comp, "r") as zf:
            infos = {info.filename: info for info in zf.infolist()}
            assert sorted(infos.keys()) == sorted(expected_webp_names)
            for name in expected_webp_names:
                assert infos[name].compress_type == zipfile.ZIP_STORED
                decoded = cv2.imdecode(np.frombuffer(zf.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
                assert decoded is not None and decoded.shape[:2] == (64, 48)

        assert MangaExporter.resolve_export_format("scan.tif") == (".png", "PNG")
        assert MangaExporter.resolve_export_format("scan.jpg", compressed=True) == (".webp", "WEBP")
        assert MangaExporter.resolve_export_format("scan.png", compressed=True) == (".webp", "WEBP")
        assert MangaExporter.is_export_compressed({"style": {"export_compressed": True}}) is True
        assert MangaExporter.is_export_compressed({}) is False
    finally:
        mgr.set_cache_format(orig_fmt)


def test_compressed_mode_forces_webp_batch_export_and_reuses_webp_cache(tmp_path, monkeypatch):
    """When export_compressed=True, batch export targets .webp and copies matching .rendered.webp without re-encoding."""
    from app.core.pipeline.batch_worker import BatchWorker

    mgr = MangaCacheManager()
    orig_fmt = mgr.get_cache_format()
    try:
        mgr.set_cache_format("webp")
        src_dir = tmp_path / "src"
        sub_dir = src_dir / "ch1"
        sub_dir.mkdir(parents=True)
        out_dir = tmp_path / "out"

        p1 = str(sub_dir / "01.png")
        p2 = str(sub_dir / "02.jpg")
        img1 = _make_sample_image(p1, w=40, h=50)
        img2 = _make_sample_image(p2, w=40, h=50)
        blk = TranslationBlock(id="b1", original_text="a", translated_text="译", xmin=4, ymin=4, xmax=20, ymax=20)
        mgr.save_page_cache(p1, erased_img=img1, blocks=[blk], rendered_img=img1)
        mgr.save_page_cache(p2, erased_img=img2, blocks=[blk], rendered_img=img2)

        assert MangaExporter.compute_export_path(p1, str(out_dir), root_dir=str(src_dir), compressed=True) == os.path.normpath(
            str(out_dir / "ch1" / "01.webp")
        )
        with pytest.raises(ValueError):
            MangaExporter.compute_export_path(p1, str(sub_dir), rel_path="01.png", compressed=True)

        # Verify zero re-encode when .rendered.webp cache hits in export_compressed=True mode
        def _forbid_encode(*args, **kwargs):
            raise AssertionError("export_hierarchical_image must not re-encode matching .rendered.webp cache")

        monkeypatch.setattr(MangaExporter, "export_hierarchical_image", staticmethod(_forbid_encode))

        items = [
            {"id": "1", "path": p1, "status": "completed", "root_dir": str(src_dir)},
            {"id": "2", "path": p2, "status": "completed", "root_dir": str(src_dir)},
        ]
        worker = BatchWorker(items, {"style": {"export_compressed": True}}, export_dir=str(out_dir), root_dir=str(src_dir))
        finished = []
        worker.sig_batch_finished.connect(lambda s, f: finished.append((s, f)))
        worker.run()

        assert finished == [(2, 0)]
        assert os.path.exists(out_dir / "ch1" / "01.webp")
        assert os.path.exists(out_dir / "ch1" / "02.webp")
        assert not os.path.exists(out_dir / "ch1" / "01.png")
        assert not os.path.exists(out_dir / "ch1" / "02.jpg")
    finally:
        mgr.set_cache_format(orig_fmt)


def test_compressed_mode_encodes_webp_from_bmp_cache_and_single_page_dialog(qapp, tmp_path, monkeypatch):
    """BMP cache with export_compressed=True exports .webp; single-page save defaults to WebP."""
    from app.core.pipeline.batch_worker import BatchWorker

    mgr = MangaCacheManager()
    orig_fmt = mgr.get_cache_format()
    try:
        mgr.set_cache_format("bmp")
        p = str(tmp_path / "scan.png")
        img = _make_sample_image(p, w=40, h=50)
        blk = TranslationBlock(id="b1", original_text="a", translated_text="译", xmin=4, ymin=4, xmax=20, ymax=20)
        mgr.save_page_cache(p, erased_img=img, blocks=[blk], rendered_img=img)

        out_dir = tmp_path / "out_bmp_to_webp"
        worker = BatchWorker([{"id": "1", "path": p, "status": "completed"}], {"style": {"export_compressed": True}}, export_dir=str(out_dir))
        res = []
        worker.sig_batch_finished.connect(lambda s, f: res.append((s, f)))
        worker.run()
        assert res == [(1, 0)]
        assert os.path.exists(out_dir / "scan.webp")
        assert not os.path.exists(out_dir / "scan.png")

        win = MainWindow()
        try:
            win.current_image_data = {"id": "1", "path": p, "blocks": [blk]}
            win.canvas_view.translated_cv = img
            seen = []
            monkeypatch.setattr(
                "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
                lambda _p, _t, name, filt: (seen.append((name, filt)) or ("", "")),
            )
            win.config.style.export_compressed = True
            win._export_current_page()
            win.config.style.export_compressed = False
            win._export_current_page()
            assert seen[0][0] == "manga_page_translated.webp"
            assert seen[0][1].startswith("WebP Image (*.webp)")
            assert seen[1][0] == "manga_page_translated.png"
            assert seen[1][1].startswith("PNG Image (*.png)")
        finally:
            win.close()
            win.deleteLater()
            qapp.processEvents()
    finally:
        mgr.set_cache_format(orig_fmt)


def test_settings_dialog_cache_format_combo(qapp, tmp_path, monkeypatch):
    from app.core.config import AppConfig
    from app.core.cache.cache_manager import get_cache_manager
    from app.ui.settings.settings_dialog import SettingsDialog

    cache_mgr = get_cache_manager()
    orig_fmt = cache_mgr.get_cache_format()
    try:
        cfg = AppConfig()
        orig_save = cfg.save
        monkeypatch.setattr(cfg, "save", lambda _p: orig_save(str(tmp_path / "cfg.json")))
        dlg = SettingsDialog(config=cfg)
        assert dlg.cache_format_combo.currentData() == "bmp"
        dlg.cache_format_combo.setCurrentIndex(1)  # webp
        dlg._on_save()
        assert cfg.cache_format == "webp"
        assert cache_mgr.get_cache_format() == "webp"
    finally:
        cache_mgr.set_cache_format(orig_fmt)
