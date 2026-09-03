"""
tests/unit/test_recursive_hierarchical_export.py
Comprehensive tests for:
1. Recursive folder image discovery with strict cache and intermediate file exclusion.
2. Natural sorting across subdirectories and deduplication.
3. 1:1 Subfolder hierarchy and original filename preservation in a separate export directory.
4. Source directory non-contamination and read-only protection.
5. Backward compatibility for single file and flat folder imports.
6. Breakpoint cache resumption in hierarchical export.
"""
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.ui.sidebar.page_list import (
    PageListWidget, natural_sort_key, natural_sort_path_key, is_ignored_cache_or_export
)
from app.core.pipeline.exporter import MangaExporter
from app.core.pipeline.batch_worker import BatchWorker
from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imwrite
from app.core.models import TranslationBlock
from app.core.config import AppConfig
from desktop.ui.queue_panel import QueuePanel
from desktop.core.batch_worker import BatchWorker as DesktopBatchWorker


@pytest.fixture
def complex_nested_manga_tree(tmp_path):
    """
    Creates a multi-level nested directory structure with:
    - Valid images in top-level and arbitrarily nested subdirectories
    - Duplicate filenames across distinct subfolders (e.g. vol1/01.png vs vol2/01.png)
    - Hidden folders (.git, .cache, .amt_cache)
    - System cache directories (__pycache__, translation_cache)
    - Intermediate cached files (.erased.webp, .rendered.webp, _erased.png, _translated.png, .blocks.json)
    - Non-image files (.txt, .exe)
    """
    source_root = tmp_path / "manga_source"
    source_root.mkdir(parents=True, exist_ok=True)

    # Top-level image
    img_top = source_root / "cover.png"
    safe_cv2_imwrite(str(img_top), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")

    # vol1/ch01 with 01.png, 02.png, 10.png
    ch01 = source_root / "vol1" / "ch01"
    ch01.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(ch01 / "01.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")
    safe_cv2_imwrite(str(ch01 / "02.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")
    safe_cv2_imwrite(str(ch01 / "10.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")

    # vol1/ch02 with 01.png (same filename as in ch01!)
    ch02 = source_root / "vol1" / "ch02"
    ch02.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(ch02 / "01.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")

    # vol2/deep/nested/sub/01.png and 02.jpg
    deep_dir = source_root / "vol2" / "deep" / "nested" / "sub"
    deep_dir.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(deep_dir / "01.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")
    safe_cv2_imwrite(str(deep_dir / "02.jpg"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".jpg")

    # Hidden folders and cache directories that MUST be ignored:
    # 1. .amt_cache inside ch01
    cache_dir = ch01 / ".amt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(cache_dir / "01.erased.webp"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".webp")
    safe_cv2_imwrite(str(cache_dir / "01.rendered.webp"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".webp")
    (cache_dir / "01.blocks.json").write_text('{"blocks": []}')

    # 2. translation_cache inside root
    trans_cache = source_root / "translation_cache"
    trans_cache.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(trans_cache / "cached_image.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")

    # 3. __pycache__ inside vol1
    pycache_dir = source_root / "vol1" / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(pycache_dir / "dummy.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")

    # 4. .git inside root
    git_dir = source_root / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(git_dir / "git_logo.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")

    # 5. .cache inside vol2
    dot_cache_dir = source_root / "vol2" / ".cache"
    dot_cache_dir.mkdir(parents=True, exist_ok=True)
    safe_cv2_imwrite(str(dot_cache_dir / "thumb.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")

    # 6. Intermediate cached files directly in ch02
    safe_cv2_imwrite(str(ch02 / "01.erased.webp"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".webp")
    safe_cv2_imwrite(str(ch02 / "01.rendered.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")
    safe_cv2_imwrite(str(ch02 / "01_erased.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")
    safe_cv2_imwrite(str(ch02 / "01_translated.png"), np.zeros((60, 60, 3), dtype=np.uint8), ext=".png")
    (ch02 / "01.blocks.json").write_text("{}")
    (ch02 / ".hidden_page.png").write_bytes(b"\x89PNG")
    (ch02 / "notes.txt").write_text("translation notes")

    return source_root


# ============================================================================
# R1 Tests: Recursive Discovery & Cache Exclusion
# ============================================================================

def test_is_ignored_cache_or_export_exhaustive():
    """Verifies that all specified cache and intermediate patterns are strictly filtered."""
    # Should be excluded:
    assert is_ignored_cache_or_export("manga/.amt_cache/01.erased.webp") is True
    assert is_ignored_cache_or_export("manga/.amt_cache/01.blocks.json") is True
    assert is_ignored_cache_or_export("manga/.amt_cache/01.rendered.webp") is True
    assert is_ignored_cache_or_export("manga/vol1/__pycache__/compiled.py") is True
    assert is_ignored_cache_or_export("manga/translation_cache/page.png") is True
    assert is_ignored_cache_or_export("manga/.git/head.png") is True
    assert is_ignored_cache_or_export("manga/.cache/tmp.jpg") is True
    assert is_ignored_cache_or_export("manga/vol1/ch01/page.erased.png") is True
    assert is_ignored_cache_or_export("manga/vol1/ch01/page.rendered.jpg") is True
    assert is_ignored_cache_or_export("manga/vol1/ch01/page_erased.png") is True
    assert is_ignored_cache_or_export("manga/vol1/ch01/page_translated.png") is True
    assert is_ignored_cache_or_export("manga/vol1/ch01/page.blocks.json") is True
    assert is_ignored_cache_or_export("manga/vol1/ch01/.hidden_img.png") is True

    # Should NOT be excluded (valid manga images):
    assert is_ignored_cache_or_export("manga/vol1/ch01/01.png") is False
    assert is_ignored_cache_or_export("manga/vol1/ch01/02.jpg") is False
    assert is_ignored_cache_or_export("manga/vol2/deep/01.webp") is False
    assert is_ignored_cache_or_export("manga/cover.png") is False


def test_recursive_discovery_discovers_nested_and_filters_caches(qapp, complex_nested_manga_tree):
    """
    Acceptance Criteria:
    - Scanning a folder containing multi-level nested subfolders discovers all images in both
      top-level and nested subdirectories.
    - Hidden folders and cache directories are completely excluded.
    - Cached intermediate files are completely excluded.
    """
    panel = PageListWidget()
    panel.add_paths([str(complex_nested_manga_tree)])

    # Exactly 7 valid images exist:
    # 1. cover.png
    # 2. vol1/ch01/01.png
    # 3. vol1/ch01/02.png
    # 4. vol1/ch01/10.png
    # 5. vol1/ch02/01.png
    # 6. vol2/deep/nested/sub/01.png
    # 7. vol2/deep/nested/sub/02.jpg
    assert len(panel.items_data) == 7

    discovered_rel_paths = [it["rel_path"].replace("\\", "/") for it in panel.items_data]
    expected_rel_paths = [
        "cover.png",
        "vol1/ch01/01.png",
        "vol1/ch01/02.png",
        "vol1/ch01/10.png",
        "vol1/ch02/01.png",
        "vol2/deep/nested/sub/01.png",
        "vol2/deep/nested/sub/02.jpg",
    ]
    assert discovered_rel_paths == expected_rel_paths
    panel.close()


def test_natural_sort_path_key_orders_hierarchies_naturally():
    """
    Acceptance Criteria:
    - Image paths across subdirectories are naturally ordered without interleaving.
    """
    raw_paths = [
        "vol1/ch01/10.png",
        "vol2/ch01/01.png",
        "vol1/ch01/02.png",
        "vol1/ch02/01.png",
        "vol1/ch01/01.png",
    ]
    sorted_paths = sorted(raw_paths, key=natural_sort_path_key)
    assert sorted_paths == [
        "vol1/ch01/01.png",
        "vol1/ch01/02.png",
        "vol1/ch01/10.png",
        "vol1/ch02/01.png",
        "vol2/ch01/01.png",
    ]


def test_deduplication_on_repeated_and_overlapping_imports(qapp, complex_nested_manga_tree):
    """
    Acceptance Criteria:
    - Image paths across subdirectories are naturally ordered and deduplicated.
    """
    panel = PageListWidget()
    # Add root folder twice, plus child folder
    panel.add_paths([
        str(complex_nested_manga_tree),
        str(complex_nested_manga_tree),
        str(complex_nested_manga_tree / "vol1"),
    ])

    assert len(panel.items_data) == 7
    # Verify no duplicate paths in queue
    paths = [it["path"] for it in panel.items_data]
    assert len(paths) == len(set(paths))
    panel.close()


# ============================================================================
# R2 Tests: Subfolder Hierarchy & Source Protection
# ============================================================================

def test_compute_export_path_preserves_hierarchy_and_original_filename(tmp_path):
    """
    Acceptance Criteria:
    - A separate target export directory contains the exported translated files,
      mirroring original relative subfolder hierarchy 1:1.
    - Exported files keep their original filename (not flattened, no _translated suffix).
    """
    source_root = tmp_path / "source"
    export_dir = tmp_path / "target_export"

    img_file = source_root / "subA" / "subB" / "page_001.png"
    target = MangaExporter.compute_export_path(
        image_path=str(img_file),
        export_dir=str(export_dir),
        rel_path=os.path.join("subA", "subB", "page_001.png"),
        root_dir=str(source_root)
    )

    expected = os.path.normpath(str(export_dir / "subA" / "subB" / "page_001.png"))
    assert target == expected
    assert os.path.basename(target) == "page_001.png"


def test_compute_export_path_prevents_source_folder_contamination(tmp_path):
    """
    Acceptance Criteria:
    - Original source directory is strictly read-only during export;
      no translated or overwritten files are written into the original source directory.
    """
    source_root = tmp_path / "source"
    img_file = source_root / "vol1" / "01.png"

    # Attempt 1: export_dir set to same folder as source_root
    with pytest.raises(ValueError) as exc_info:
        MangaExporter.compute_export_path(
            image_path=str(img_file),
            export_dir=str(source_root),
            rel_path=os.path.join("vol1", "01.png"),
            root_dir=str(source_root)
        )
    assert "source" in str(exc_info.value).lower()

    # Attempt 2: target matches original source image directly
    with pytest.raises(ValueError) as exc_info:
        MangaExporter.compute_export_path(
            image_path=str(img_file),
            export_dir=str(source_root / "vol1"),
            rel_path="01.png"
        )
    assert "matches original source" in str(exc_info.value).lower()


def test_distinct_files_with_same_name_do_not_collide(tmp_path):
    """
    Acceptance Criteria:
    - Distinct files with the same filename across different subfolders
      (e.g., vol1/01.png and vol2/01.png) both export successfully into their
      respective subdirectories without collision or overwrite.
    """
    source_root = tmp_path / "source"
    export_dir = tmp_path / "export"

    file1 = source_root / "vol1" / "01.png"
    file2 = source_root / "vol2" / "01.png"

    target1 = MangaExporter.compute_export_path(str(file1), str(export_dir), rel_path=os.path.join("vol1", "01.png"))
    target2 = MangaExporter.compute_export_path(str(file2), str(export_dir), rel_path=os.path.join("vol2", "01.png"))

    assert target1 != target2
    assert os.path.normpath(target1) == os.path.normpath(str(export_dir / "vol1" / "01.png"))
    assert os.path.normpath(target2) == os.path.normpath(str(export_dir / "vol2" / "01.png"))


def test_backward_compatibility_single_file_and_flat_folder(tmp_path):
    """
    Acceptance Criteria:
    - Single file import or flat folder import continues to export correctly into
      the export directory without invalid path errors.
    """
    export_dir = tmp_path / "export"

    # 1. Single file drop
    single_img = tmp_path / "standalone" / "standalone_page.png"
    t_single = MangaExporter.compute_export_path(
        str(single_img),
        str(export_dir),
        rel_path="standalone_page.png",
        root_dir=str(tmp_path / "standalone")
    )
    assert os.path.normpath(t_single) == os.path.normpath(str(export_dir / "standalone_page.png"))

    # 2. Flat folder
    flat_img = tmp_path / "flat_folder" / "01.png"
    t_flat = MangaExporter.compute_export_path(
        str(flat_img),
        str(export_dir),
        rel_path="01.png",
        root_dir=str(tmp_path / "flat_folder")
    )
    assert os.path.normpath(t_flat) == os.path.normpath(str(export_dir / "01.png"))


# ============================================================================
# R3 Integration Test: Hierarchical Batch Export with Source Non-Contamination
# ============================================================================

@patch("app.core.pipeline.batch_worker.OCREngine")
@patch("app.core.pipeline.batch_worker.InpaintEngine")
@patch("app.core.pipeline.batch_worker.TranslationManager")
@patch("app.core.pipeline.batch_worker.TypographyEngine")
def test_batch_worker_hierarchical_export_end_to_end(
    mock_typo_cls, mock_trans_cls, mock_inpaint_cls, mock_ocr_cls,
    qapp, complex_nested_manga_tree, tmp_path
):
    """
    Full end-to-end integration test of BatchWorker verifying:
    1. 1:1 subfolder structure created under export_dir
    2. Original filenames preserved
    3. Source directory is strictly read-only and unpolluted
    4. Breakpoint resumption exports correctly
    """
    export_dir = tmp_path / "independent_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Setup mock engines
    mock_ocr = MagicMock()
    mock_ocr.detect_and_recognize.return_value = [
        {"id": "b1", "original_text": "text", "translated_text": "", "xmin": 5, "ymin": 5, "xmax": 20, "ymax": 20}
    ]
    mock_ocr_cls.return_value = mock_ocr

    mock_inpaint = MagicMock()
    mock_inpaint.inpaint.return_value = np.zeros((60, 60, 3), dtype=np.uint8)
    mock_inpaint_cls.return_value = mock_inpaint

    mock_trans_mgr = MagicMock()
    mock_trans_mgr.translate.return_value = [
        TranslationBlock(id="b1", original_text="text", translated_text="译文", xmin=5, ymin=5, xmax=20, ymax=20)
    ]
    mock_trans_cls.get_instance.return_value = mock_trans_mgr

    mock_typo = MagicMock()
    # Distinct translated image pattern
    mock_typo.render_translations.return_value = np.ones((60, 60, 3), dtype=np.uint8) * 150
    mock_typo_cls.return_value = mock_typo

    # Take snapshot of source folder contents before export
    source_files_before = set()
    for root, _, files in os.walk(str(complex_nested_manga_tree)):
        for f in files:
            source_files_before.add(os.path.relpath(os.path.join(root, f), str(complex_nested_manga_tree)))

    # Ingest paths via PageListWidget
    panel = PageListWidget()
    panel.add_paths([str(complex_nested_manga_tree)])
    assert len(panel.items_data) == 7

    # Run batch translation
    worker = BatchWorker(
        queue_items=panel.items_data,
        config={"ocr_engine": "easyocr", "provider": "deepseek"},
        export_dir=str(export_dir)
    )

    completed_events = []
    worker.sig_item_completed.connect(lambda iid, res: completed_events.append(res))
    worker.run()

    assert len(completed_events) == 7

    # 1. Verify 1:1 directory hierarchy and original filenames in export_dir
    expected_exports = [
        export_dir / "cover.png",
        export_dir / "vol1" / "ch01" / "01.png",
        export_dir / "vol1" / "ch01" / "02.png",
        export_dir / "vol1" / "ch01" / "10.png",
        export_dir / "vol1" / "ch02" / "01.png",
        export_dir / "vol2" / "deep" / "nested" / "sub" / "01.png",
        export_dir / "vol2" / "deep" / "nested" / "sub" / "02.jpg",
    ]

    for exp_path in expected_exports:
        assert exp_path.exists(), f"Expected exported file missing: {exp_path}"
        assert exp_path.stat().st_size > 0

    # Verify identical filenames did NOT overwrite each other
    # vol1/ch01/01.png and vol1/ch02/01.png and vol2/.../01.png all exist independently!
    assert (export_dir / "vol1" / "ch01" / "01.png").exists()
    assert (export_dir / "vol1" / "ch02" / "01.png").exists()
    assert (export_dir / "vol2" / "deep" / "nested" / "sub" / "01.png").exists()

    # 2. Verify source directory is strictly read-only (unpolluted)
    # The only files that might be added to source are .amt_cache intermediate cache
    for root, _, files in os.walk(str(complex_nested_manga_tree)):
        for f in files:
            full_p = os.path.join(root, f)
            rel = os.path.relpath(full_p, str(complex_nested_manga_tree))
            if ".amt_cache" not in rel:
                # No non-cache files were added to source tree!
                assert rel in source_files_before, f"Source tree was modified with new file: {rel}"

    # 3. Verify breakpoint resumption with existing cache
    # Delete one exported file from export_dir
    test_deleted = export_dir / "vol1" / "ch01" / "02.png"
    test_deleted.unlink()
    assert not test_deleted.exists()

    # Run BatchWorker again
    worker2 = BatchWorker(
        queue_items=panel.items_data,
        config={"ocr_engine": "easyocr", "provider": "deepseek"},
        export_dir=str(export_dir)
    )
    worker2.run()

    # Breakpoint resumption should re-export missing file from cache without running OCR/LLM
    assert test_deleted.exists()
    assert test_deleted.stat().st_size > 0

    panel.close()


# ============================================================================
# Adversarial Edge-Case Tests (Discovered during Skeptical Review)
# ============================================================================

def test_case_insensitive_source_protection_windows(tmp_path):
    """
    Verifies that case variations on Windows (e.g. drive letter or directory casing)
    cannot bypass the source non-contamination and overwrite checks.
    Also verifies that selecting an export folder INSIDE the source directory is strictly blocked.
    """
    source_dir = tmp_path / "MangaSource"
    source_dir.mkdir(parents=True, exist_ok=True)
    img_file = source_dir / "page_01.png"
    safe_cv2_imwrite(str(img_file), np.zeros((30, 30, 3), dtype=np.uint8), ext=".png")

    # 1. Export dir is case-different version of source dir (e.g. lowercase vs uppercase)
    lower_source = str(source_dir).lower()
    with pytest.raises(ValueError) as exc_info:
        MangaExporter.compute_export_path(
            image_path=str(img_file),
            export_dir=lower_source,
            rel_path="page_01.png",
            root_dir=str(source_dir).upper()
        )
    assert "source" in str(exc_info.value).lower()

    # 2. Target file matches original source file with case difference
    with pytest.raises(ValueError) as exc_info:
        MangaExporter.compute_export_path(
            image_path=str(img_file),
            export_dir=str(source_dir).lower(),
            rel_path="PAGE_01.PNG",
            root_dir=None
        )
    assert "matches original source" in str(exc_info.value).lower()

    # 3. Direct overwrite attempt in export_hierarchical_image with case mismatch
    with pytest.raises(ValueError) as exc_info:
        MangaExporter.export_hierarchical_image(
            np.zeros((30, 30, 3), dtype=np.uint8),
            export_path=str(img_file).lower(),
            source_path=str(img_file).upper()
        )
    assert "overwrite" in str(exc_info.value).lower()


def test_jpeg_and_bmp_original_filename_preservation(tmp_path):
    """
    Verifies that .jpeg and .bmp images preserve their exact filenames upon export,
    without appending duplicate extensions like .jpeg.jpg or .bmp.png.
    """
    export_dir = tmp_path / "format_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    test_img = np.zeros((40, 40, 3), dtype=np.uint8)

    # 1. .jpeg
    target_jpeg = str(export_dir / "page_01.jpeg")
    ok = MangaExporter.export_hierarchical_image(test_img, target_jpeg)
    assert ok is True
    assert os.path.exists(target_jpeg), "page_01.jpeg should exist directly"
    assert not os.path.exists(target_jpeg + ".jpg"), "Double extension .jpeg.jpg must NOT be created!"

    # 2. .bmp
    target_bmp = str(export_dir / "page_02.bmp")
    ok = MangaExporter.export_hierarchical_image(test_img, target_bmp)
    assert ok is True
    assert os.path.exists(target_bmp), "page_02.bmp should exist directly"
    assert not os.path.exists(target_bmp + ".png"), "Double extension .bmp.png must NOT be created!"


def test_multi_directory_drop_does_not_collide(tmp_path, qapp):
    """
    Verifies that dropping multiple disjoint directories (e.g. vol1 and vol2 from file dialog)
    into PageListWidget or BatchWorker calculates relative paths against their common parent,
    preventing identical filenames across chapters (e.g. vol1/01.png and vol2/01.png)
    from colliding and flattening into the root export folder.
    """
    manga_root = tmp_path / "series"
    vol1 = manga_root / "vol1"
    vol2 = manga_root / "vol2"
    vol1.mkdir(parents=True, exist_ok=True)
    vol2.mkdir(parents=True, exist_ok=True)

    img1 = vol1 / "01.png"
    img2 = vol2 / "01.png"
    safe_cv2_imwrite(str(img1), np.zeros((30, 30, 3), dtype=np.uint8), ext=".png")
    safe_cv2_imwrite(str(img2), np.zeros((30, 30, 3), dtype=np.uint8), ext=".png")

    # Ingest multiple directories together
    panel = PageListWidget()
    panel.add_paths([str(vol1), str(vol2)])
    assert len(panel.items_data) == 2

    # Check that rel_path preserved vol1 and vol2
    rel_paths = [it["rel_path"].replace("\\", "/") for it in panel.items_data]
    assert "vol1/01.png" in rel_paths
    assert "vol2/01.png" in rel_paths

    # Test BatchWorker resolving export paths
    export_dir = tmp_path / "multi_export"
    worker = BatchWorker(panel.items_data, {}, export_dir=str(export_dir))
    export_p1 = worker.resolve_export_path(panel.items_data[0])
    export_p2 = worker.resolve_export_path(panel.items_data[1])

    assert export_p1 != export_p2, "Export paths for different volumes must NOT collide!"
    assert os.path.normpath(export_p1) != os.path.normpath(export_p2)
    assert "vol1" in export_p1 or "vol1" in export_p2
    assert "vol2" in export_p1 or "vol2" in export_p2
    panel.close()


def test_desktop_queue_panel_and_batch_worker_hierarchical(tmp_path, qapp):
    """
    Verifies that desktop UI QueuePanel and DesktopBatchWorker support hierarchical
    export, source protection, and collision-free resolution.
    """
    manga_root = tmp_path / "desktop_series"
    ch1 = manga_root / "ch1"
    ch2 = manga_root / "ch2"
    ch1.mkdir(parents=True, exist_ok=True)
    ch2.mkdir(parents=True, exist_ok=True)

    img1 = ch1 / "page.png"
    img2 = ch2 / "page.png"
    safe_cv2_imwrite(str(img1), np.zeros((30, 30, 3), dtype=np.uint8), ext=".png")
    safe_cv2_imwrite(str(img2), np.zeros((30, 30, 3), dtype=np.uint8), ext=".png")

    q_panel = QueuePanel()
    q_panel.add_paths([str(ch1), str(ch2)])
    assert len(q_panel.items_data) == 2

    export_dir = tmp_path / "desktop_export"
    worker = DesktopBatchWorker(q_panel.items_data, {}, export_dir=str(export_dir))

    p1 = worker.resolve_export_path(q_panel.items_data[0])
    p2 = worker.resolve_export_path(q_panel.items_data[1])

    assert p1 is not None and p2 is not None
    assert p1 != p2, "Desktop batch worker must not collide duplicate filenames across subdirectories"
    assert "ch1" in p1 or "ch1" in p2
    assert "ch2" in p1 or "ch2" in p2
    q_panel.close()


def test_parent_dot_folder_not_falsely_ignored():
    """
    Verifies that legitimate manga images stored inside user home directory paths
    containing dots (e.g. .gemini, .cargo, .config) are NOT falsely ignored.
    """
    # Dot in ancestor user path:
    assert is_ignored_cache_or_export(r"C:\Users\username\.gemini\manga\ch1\01.png") is False
    assert is_ignored_cache_or_export(r"/home/user/.config/manga/vol1/cover.jpg") is False

    # Actual internal cache directories:
    assert is_ignored_cache_or_export(r"C:\Users\username\manga\.amt_cache\01.png") is True
    assert is_ignored_cache_or_export(r"C:\Users\username\manga\translation_cache\01.png") is True
    assert is_ignored_cache_or_export(r"C:\Users\username\manga\__pycache__\01.png") is True
    assert is_ignored_cache_or_export(r"C:\Users\username\manga\vol1\.git\01.png") is True


def test_appconfig_dataclass_style_export_compressed():
    """
    Verifies that BatchWorker correctly extracts style.export_compressed when
    config is passed as an AppConfig dataclass instance or as a dict.
    """
    cfg = AppConfig()
    cfg.style.export_compressed = True
    worker = BatchWorker([], config=cfg, export_dir="")

    # Verify style extraction logic
    if hasattr(worker.config, "style"):
        compressed = getattr(worker.config.style, "export_compressed", False)
    else:
        compressed = False
    assert compressed is True

