import os
import sys
import tempfile
import shutil
import json
import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 1. Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. Set headless / offscreen Qt platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication
from desktop.core.config_manager import ConfigManager, DEFAULT_CONFIG

@pytest.fixture(scope="session")
def qapp():
    """Provides a single shared QApplication instance running offscreen."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    yield app

@pytest.fixture
def temp_dir():
    """Provides an isolated temporary directory for test artifacts."""
    d = tempfile.mkdtemp(prefix="manga_test_")
    yield d
    try:
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass

@pytest.fixture
def isolated_config(temp_dir):
    """Provides a clean ConfigManager instance backed by a temporary JSON file."""
    cfg_path = os.path.join(temp_dir, "test_config.json")
    mgr = ConfigManager()
    mgr._data = DEFAULT_CONFIG.copy()
    yield mgr

@pytest.fixture
def sample_manga_image_np():
    """
    Creates a synthetic 800x1200 manga page in BGR format with:
    - Screentone / gray gradient background
    - An elliptical/rectangular white speech bubble
    - Simulated dark Japanese text inside the bubble
    """
    h, w = 1200, 800
    img = np.full((h, w, 3), 220, dtype=np.uint8)

    # Add dark border art
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (50, 50, 50), 3)

    # Add speech bubble: white filled rectangle with border
    # Bubble 1: (xmin=100, ymin=150, xmax=350, ymax=450)
    cv2.rectangle(img, (100, 150), (350, 450), (255, 255, 255), -1)
    cv2.rectangle(img, (100, 150), (350, 450), (10, 10, 10), 2)
    # Add Japanese text simulation
    cv2.putText(img, "KONNICHIWA", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "SEKAI!", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Bubble 2: (xmin=450, ymin=600, xmax=700, ymax=850)
    cv2.rectangle(img, (450, 600), (700, 850), (255, 255, 255), -1)
    cv2.rectangle(img, (450, 600), (700, 850), (10, 10, 10), 2)
    cv2.putText(img, "TASUKETE", (470, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img

@pytest.fixture
def sample_manga_image_file(temp_dir, sample_manga_image_np):
    """Writes the synthetic manga image to disk as PNG and returns path."""
    path = os.path.join(temp_dir, "test_page_01.png")
    _, buf = cv2.imencode(".png", sample_manga_image_np)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path

@pytest.fixture
def sample_translation_blocks():
    """Provides standard mock translation blocks."""
    return [
        {
            "id": "b1001",
            "original_text": "こんにちは、世界！",
            "translated_text": "你好，世界！",
            "xmin": 12.5,
            "ymin": 12.5,
            "xmax": 43.75,
            "ymax": 37.5,
            "bg_color": "#FFFFFF",
            "text_color": "#000000",
            "type": "bubble"
        },
        {
            "id": "b1002",
            "original_text": "助けて！",
            "translated_text": "救命啊！",
            "xmin": 56.25,
            "ymin": 50.0,
            "xmax": 87.5,
            "ymax": 70.83,
            "bg_color": "#FFFFFF",
            "text_color": "#000000",
            "type": "bubble"
        }
    ]

@pytest.fixture
def sample_chapter_dir(temp_dir, sample_manga_image_np):
    """Creates a chapter directory containing multiple pages and non-image files."""
    chap_dir = os.path.join(temp_dir, "Chapter_01")
    os.makedirs(chap_dir, exist_ok=True)

    # 3 manga pages
    for i in range(1, 4):
        p = os.path.join(chap_dir, f"page_{i:02d}.png")
        _, buf = cv2.imencode(".png", sample_manga_image_np)
        with open(p, "wb") as f:
            f.write(buf.tobytes())

    # Non-image files to verify filtering
    with open(os.path.join(chap_dir, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("Chapter 1 translation notes")
    with open(os.path.join(chap_dir, ".DS_Store"), "wb") as f:
        f.write(b"\x00\x00")

    return chap_dir
