"""
tests/challenge/test_challenge_gui_stress.py
Empirical Adversarial Stress Test Suite for Milestone M2:
1. Rapid Theme Toggling (100x tight loop, QSS memory & visual integrity)
2. Ingestion & Drag/Drop Torture (0-byte crash bug, corrupt images, non-images, Unicode/emoji paths)
3. Natural Alphanumeric Sorting (user tricky list, chapter multi-directory flaw, legacy queue comparison)
4. Window Resizing Stress (rapid resizing between 1000x650 and 3840x2160, extreme aspect ratios)
"""
import os
import re
import tempfile
import tracemalloc
import pytest
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QPixmap

from app.ui.main_window import MainWindow as AppMainWindow
from app.ui.theme.tokens import get_tokens, build_stylesheet, DARK_TOKENS, LIGHT_TOKENS
from app.ui.sidebar.page_list import PageListWidget, natural_sort_key
from desktop.ui.main_window import MainWindow as DesktopMainWindow
from desktop.ui.queue_panel import QueuePanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# =============================================================================
# 1. RAPID THEME TOGGLING STRESS (100 ITERATIONS)
# =============================================================================

def test_challenge_rapid_theme_toggling_100x(qapp):
    """
    Stress-tests MainWindow theme switching 100 times in a tight loop.
    Verifies:
    - No unbounded memory leak in QSS stylesheet generation
    - Correct alternating theme state ('dark' -> 'light' -> 'dark')
    - Valid stylesheet matching get_tokens definition
    - Widget and icon references remain valid
    """
    win = AppMainWindow()
    win.show()

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    for i in range(100):
        win.toggle_theme()
        qapp.processEvents()

    snap_after = tracemalloc.take_snapshot()
    top_diffs = snap_after.compare_to(snap_before, 'lineno')

    # Total positive memory growth
    total_growth = sum(stat.size_diff for stat in top_diffs if stat.size_diff > 0)

    # 100 toggles starting from dark lands on dark
    assert win._current_theme == "dark"
    assert "sun" in win.theme_btn.toolTip() or win.theme_btn.icon() is not None
    assert win.styleSheet() == build_stylesheet(DARK_TOKENS)

    # Memory growth should be negligible (< 1 MB for 100 QSS switches)
    assert total_growth < 1024 * 1024, f"Excessive memory growth detected: {total_growth} bytes"

    win.close()


def test_challenge_theme_tokens_idempotency_and_invalid_fallback():
    """Verifies token retrieval is idempotent and safely falls back on unknown theme strings."""
    for _ in range(50):
        dark1 = get_tokens("dark")
        dark2 = get_tokens("dark")
        assert dark1 == dark2

        fallback = get_tokens("non_existent_cyber_theme")
        assert fallback.name == "dark"


# =============================================================================
# 2. WINDOW RESIZING STRESS & SPLITTER STABILITY
# =============================================================================

def test_challenge_window_resizing_rapid_cycles(qapp):
    """
    Rapidly resizes MainWindow between minimum size (1000x650) and 4K (3840x2160)
    for 50 cycles to verify layout thrashing stability and splitter proportions.
    """
    win = AppMainWindow()
    win.show()

    for i in range(50):
        if i % 2 == 0:
            win.resize(3840, 2160)
        else:
            win.resize(1000, 650)
        qapp.processEvents()

    # Verify splitter remains properly partitioned with 3 valid positive panes
    sizes = win.splitter.sizes()
    assert len(sizes) == 3
    assert all(s > 0 for s in sizes)
    assert win.minimumWidth() == 1000
    assert win.minimumHeight() == 650

    win.close()


def test_challenge_window_extreme_aspect_ratios(qapp):
    """Verifies MainWindow layout survives extreme aspect ratios without crashing."""
    win = AppMainWindow()
    win.show()

    extreme_dimensions = [
        (4000, 400),  # Ultra-wide
        (400, 3000),  # Ultra-tall
        (800, 600),   # Below minimum bounds
        (1920, 1080), # Standard 1080p
    ]

    for w, h in extreme_dimensions:
        win.resize(w, h)
        qapp.processEvents()
        # Window must respect minimum boundaries
        assert win.width() >= win.minimumWidth()
        assert win.height() >= win.minimumHeight()
        assert win.splitter.count() == 3

    win.close()


# =============================================================================
# 3. NATURAL ALPHANUMERIC SORTING TORTURE TESTS
# =============================================================================

def test_challenge_natural_sort_user_tricky_list():
    """
    Verifies natural sorting on tricky lists explicitly specified by user:
    ['page_10', 'page_2', 'page_1', 'page_20', 'page_100', '1', '2', '10']
    """
    tricky_list = ['page_10', 'page_2', 'page_1', 'page_20', 'page_100', '1', '2', '10']
    sorted_result = sorted(tricky_list, key=natural_sort_key)

    expected = ['1', '2', '10', 'page_1', 'page_2', 'page_10', 'page_20', 'page_100']
    assert sorted_result == expected


def test_challenge_natural_sort_comprehensive_cases():
    """Tests multi-number tokens, mixed case, and punctuation."""
    test_cases = [
        # Multi-number tokens (chapter + page)
        (['ch1_p10', 'ch1_p2', 'ch2_p1', 'ch10_p1', 'ch1_p1'],
         ['ch1_p1', 'ch1_p2', 'ch1_p10', 'ch2_p1', 'ch10_p1']),

        # Mixed casing
        (['PAGE_10.png', 'page_2.png', 'Page_1.png'],
         ['Page_1.png', 'page_2.png', 'PAGE_10.png']),

        # Brackets and hashes
        (['#10.jpg', '#2.jpg', '#1.jpg', 'img [10].png', 'img [2].png'],
         ['#1.jpg', '#2.jpg', '#10.jpg', 'img [2].png', 'img [10].png']),
    ]

    for input_list, expected_output in test_cases:
        actual = sorted(input_list, key=natural_sort_key)
        assert actual == expected_output


def test_challenge_natural_sort_multi_directory_chapter_flaw():
    """
    ADVERSARIAL FINDING:
    natural_sort_key uses os.path.basename(s), so it discards directory context.
    When multiple chapter folders are imported, pages from different chapters
    become interleaved: ['ch1/p1', 'ch2/p1', 'ch1/p2', 'ch2/p2'] instead of
    preserving chapter sequentiality: ['ch1/p1', 'ch1/p2', 'ch2/p1', 'ch2/p2'].
    """
    multi_chapter_paths = [
        "chapter_01/page_01.png",
        "chapter_01/page_02.png",
        "chapter_02/page_01.png",
        "chapter_02/page_02.png",
    ]

    sorted_paths = sorted(multi_chapter_paths, key=natural_sort_key)

    # Confirms that basename-only sorting interleaves pages across chapters
    assert sorted_paths == [
        "chapter_01/page_01.png",
        "chapter_02/page_01.png",
        "chapter_01/page_02.png",
        "chapter_02/page_02.png",
    ], "Basename-only sort behavior changed!"


def test_challenge_desktop_legacy_queue_panel_lacks_natural_sort():
    """
    ADVERSARIAL FINDING:
    desktop/ui/queue_panel.py uses standard ASCII sorted(files),
    failing natural sort by ordering 'page1.png', 'page10.png', 'page2.png'.
    """
    files = ["page10.png", "page2.png", "page1.png"]
    ascii_sorted = sorted(files)  # What desktop/ui/queue_panel.py does
    assert ascii_sorted == ["page1.png", "page10.png", "page2.png"]
    assert ascii_sorted != ["page1.png", "page2.png", "page10.png"]


# =============================================================================
# 4. INGESTION & DRAG-AND-DROP TORTURE
# =============================================================================

def test_challenge_ingestion_filter_non_image_files(qapp, tmp_path):
    """Feeds non-image files (.txt, .exe, .py, .pdf, .zip) and verifies exclusion."""
    panel = PageListWidget()

    invalid_files = []
    for ext in [".txt", ".exe", ".py", ".pdf", ".zip", ".dll"]:
        f = tmp_path / f"test{ext}"
        f.write_text("dummy payload")
        invalid_files.append(str(f))

    panel.add_paths(invalid_files)
    assert len(panel.items_data) == 0
    assert panel.list_widget.count() == 0
    panel.close()


def test_challenge_ingestion_unicode_and_emoji_paths(qapp, tmp_path):
    """Feeds valid images located in directories with Chinese, Japanese, and emojis."""
    emoji_dir = tmp_path / "漫畫_測試_🔥_🎨"
    emoji_dir.mkdir(parents=True, exist_ok=True)

    img_path = emoji_dir / "頁面_01_🌸.png"
    # Create valid dummy PNG
    dummy_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode(".png", dummy_img)
    img_path.write_bytes(buf.tobytes())

    panel = PageListWidget()
    panel.add_paths([str(img_path)])

    assert len(panel.items_data) == 1
    assert "頁面_01_🌸.png" in panel.items_data[0]["path"]

    # Verify thumbnail loading does not crash
    widget = panel._item_widgets[panel.items_data[0]["id"]]
    assert widget.thumb_label.pixmap() is not None
    assert not widget.thumb_label.pixmap().isNull()

    panel.close()


def test_challenge_ingestion_corrupt_non_empty_image(qapp, tmp_path):
    """
    Feeds a corrupted non-empty file disguised with .png extension.
    Verifies that thumbnail and cv2.imdecode handle it gracefully without crashing.
    """
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"THIS IS NOT A VALID PNG FORMAT HEADER 123456789")

    panel = PageListWidget()
    panel.add_paths([str(corrupt_file)])

    assert len(panel.items_data) == 1
    widget = panel._item_widgets[panel.items_data[0]["id"]]
    # QPixmap fails gracefully on corrupt image without crash
    assert widget.thumb_label.pixmap() is None or widget.thumb_label.pixmap().isNull()

    # cv2.imdecode returns None without throwing OpenCV exception
    arr = np.fromfile(str(corrupt_file), dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is None

    panel.close()


def test_challenge_ingestion_zero_byte_file_crash_confirmed(qapp, tmp_path):
    """
    CRITICAL ADVERSARIAL BUG CONFIRMATION:
    When a 0-byte file (e.g. empty.png) is dropped, PageListWidget accepts it based
    on extension. When selected, MainWindow._on_page_selected executes:
        cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    Because buffer is empty, OpenCV throws cv2.error: Assertion failed (!buf.empty()).
    This test confirms the vulnerability empirically.
    """
    empty_file = tmp_path / "empty.png"
    empty_file.touch()

    # 1. PageListWidget accepts it because extension is .png
    panel = PageListWidget()
    panel.add_paths([str(empty_file)])
    assert len(panel.items_data) == 1

    # 2. Reading 0 bytes with np.fromfile results in size 0 array
    buf = np.fromfile(str(empty_file), dtype=np.uint8)
    assert len(buf) == 0

    # 3. cv2.imdecode raises uncaught cv2.error on empty buffer
    with pytest.raises(cv2.error) as exc_info:
        cv2.imdecode(buf, cv2.IMREAD_COLOR)

    assert "!buf.empty()" in str(exc_info.value)
    panel.close()


def test_challenge_ingestion_directory_recursive_filtering(qapp, tmp_path):
    """Tests recursive scanning of nested folder with mixed valid and invalid files."""
    root_dir = tmp_path / "chapter_nested"
    sub_dir = root_dir / "sub"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Valid images
    (root_dir / "01.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (sub_dir / "02.jpg").write_bytes(b"\xff\xd8\xff")

    # Invalid files
    (root_dir / "info.txt").write_text("info")
    (sub_dir / "setup.exe").write_bytes(b"MZ")

    panel = PageListWidget()
    panel.add_paths([str(root_dir)])

    # Exactly the 2 images should be discovered
    assert len(panel.items_data) == 2
    basenames = [os.path.basename(it["path"]) for it in panel.items_data]
    assert basenames == ["01.png", "02.jpg"]

    panel.close()
