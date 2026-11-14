"""
tests/unit/test_bugfixes_audit.py
Regression unit tests covering core, pipeline, OCR, translation, typography, and UI bug fixes.
"""
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image, ImageFont
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from app.core.cache.cache_manager import MangaCacheManager, safe_cv2_imwrite
from app.core.config import AppConfig
from app.core.inpaint.color_analyzer import get_background_color_rgb, refresh_blocks_text_colors
from app.core.inpaint.lama_engine import LaMaInpainter
from app.core.inpaint.opencv_engine import OpenCVInpainter
from app.core.models import StyleConfig, TranslationBlock
from app.core.ocr.easyocr_engine import rotate_crop_upright
from app.core.ocr.qr_filter import QRCodeFilter, QRRegion
from app.core.pipeline.batch_worker import BatchWorker
from app.core.pipeline.exporter import MangaExporter
from app.core.translation import ProviderConfig, TranslationManager
from app.core.translation.custom_provider import CustomOpenAIProvider
from app.core.translation.json_parser import parse_llm_json_response
from app.core.typography.engine import TypographyEngine
from app.core.typography.stroke_renderer import StrokeRenderer
from app.ui.canvas.items.bubble_item import BubbleItem, HANDLE_BOTTOM_RIGHT
from app.ui.inspector.inspector_panel import InspectorPanel
from app.ui.main_window import MainWindow
from app.ui.settings.page_style_dialog import PageStyleDialog
from app.ui.widgets.in_place_editor import ContextualTypographyBar, InPlaceBubbleEditor
from app.ui.widgets.thumbnail_loader import AsyncThumbnailManager
from desktop.core.translation_engine import TranslationEngine


def test_app_config_from_dict_tolerates_null_and_unknown_fields():
    cfg_bad_root = AppConfig.from_dict("not a dict")
    assert isinstance(cfg_bad_root, AppConfig)

    cfg = AppConfig.from_dict({
        "llm": None,
        "ocr": {"engine": "paddle", "unknown_ocr_field": 123},
        "inpaint": {"engine": "opencv", "unknown_inpaint_field": True},
        "unknown_top_level": "ignored",
    })
    assert cfg.ocr.engine == "paddle"
    assert cfg.inpaint.engine == "opencv"
    assert cfg.llm.provider == "openai"


def test_cache_manager_fallback_png_false_pending_merge_and_sibling_clear(tmp_path):
    img = np.full((32, 32, 3), 180, dtype=np.uint8)
    bad_path = str(tmp_path / "out.webp")
    assert safe_cv2_imwrite(bad_path, img, ext=".invalid_ext", fallback_png=False) is False
    assert not os.path.exists(bad_path)

    p1 = str(tmp_path / "page1.png")
    p2 = str(tmp_path / "page2.png")
    assert safe_cv2_imwrite(p1, img, ext=".png")
    assert safe_cv2_imwrite(p2, img, ext=".png")

    cm = MangaCacheManager()
    cm.save_page_cache(p1, blocks=[{"id": "b1", "original_text": "hi"}], rendered_img=img)
    cm.save_page_cache(p2, blocks=[{"id": "b2", "original_text": "yo"}], rendered_img=img)

    # Queue pending async blocks then overwrite only rendered_img -> both must survive
    cm.save_page_cache_async(p1, blocks=[{"id": "b1_new", "original_text": "updated"}])
    cm.save_page_cache(p1, rendered_img=img)
    cm.flush_pending_writes()

    loaded_p1 = cm.load_page_cache(p1, load_images=False)
    b0 = loaded_p1["blocks"][0]
    assert (b0["id"] if isinstance(b0, dict) else b0.id) == "b1_new"

    # Clearing p1 must preserve p2 in same .amt_cache
    cm.clear_caches_for_items([p1])
    assert cm.has_cache(p1)["blocks"] is False
    assert cm.has_cache(p2)["blocks"] is True
    cache_dir = cm.get_cache_paths(p2)["dir"]
    assert os.path.isdir(cache_dir)

    # Clearing last page deletes empty .amt_cache directory
    cm.clear_caches_for_items([p2])
    assert not os.path.exists(cache_dir)


def test_color_analyzer_grayscale_and_neutral_speaker_skip():
    gray = np.full((40, 40), 120, dtype=np.uint8)
    assert get_background_color_rgb(gray) == (120, 120, 120)

    canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
    blocks = [
        {"id": "1", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40, "text_color": "#000000", "translated_text": "(A) 你好"},
        {"id": "2", "xmin": 50, "ymin": 10, "xmax": 80, "ymax": 40, "text_color": "#111111", "translated_text": "(A) 再见"},
    ]
    refresh_blocks_text_colors(canvas, blocks)
    assert blocks[0]["text_color"] == "#000000"


def test_inpaint_engines_accept_dict_blocks():
    img = np.full((64, 64, 3), 240, dtype=np.uint8)
    dict_block = {"id": "d1", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50, "type": "bubble"}

    ocv = OpenCVInpainter()
    res_ocv = ocv.inpaint(img, [dict_block])
    assert res_ocv is not None and res_ocv.shape == img.shape

    lama = LaMaInpainter()
    lama._model_loaded = True
    lama._simple_lama = lambda pil_img, pil_mask: Image.fromarray(img[:, :, ::-1])
    res_lama = lama.inpaint(img, [dict_block])
    assert res_lama is not None and res_lama.shape == img.shape


def test_qr_filter_batch_pixel_mode_and_deskew_sign():
    qr = QRRegion(bbox=(100, 100, 400, 400), bbox_normalized=(10.0, 10.0, 40.0, 40.0))
    boxes = [
        {"xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40, "original_text": "ok"},
        {"xmin": 500, "ymin": 500, "xmax": 750, "ymax": 600, "original_text": "dialogue"},
    ]
    kept = QRCodeFilter().filter_spurious_ocr_boxes(boxes, [qr], w_img=1000, h_img=1000)
    assert len(kept) == 2

    # Tilted crop rotated by -angle deskews a line drawn at +10 deg toward horizontal
    crop = np.zeros((40, 80, 3), dtype=np.uint8)
    # In screen coords (y down), line y=20+(x-40)*tan(+10deg) slopes downward-right (+10deg clockwise)
    cv2.line(crop, (10, 15), (70, 25), (255, 255, 255), 3)
    rot, mat = rotate_crop_upright(crop, 10.0)
    assert rot.shape[0] >= crop.shape[0] and mat is not None and mat.shape == (2, 3)
    ys, xs = np.where(rot[:, :, 0] > 128)
    # Slope after +10 deg counter-clockwise deskew should be near 0 (smaller vertical spread than 10px)
    assert (ys.max() - ys.min()) < 8


def test_provider_endpoint_json_parser_and_local_empty_key():
    assert (
        CustomOpenAIProvider._normalize_endpoint("https://generativelanguage.googleapis.com/v1beta/openai")
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert (
        CustomOpenAIProvider._normalize_endpoint("http://localhost:11434")
        == "http://localhost:11434/v1/chat/completions"
    )

    parsed_trail = parse_llm_json_response('[{"id": "b1", "translated_text": "你好"}]\nTrailing note [1].')
    assert isinstance(parsed_trail, list) and parsed_trail[0]["translated_text"] == "你好"

    parsed_trunc = parse_llm_json_response('[\n  {"id": "b4", "translated_text": "等等"},\n  {"id": "b5", "translated_text": "不\n')
    assert isinstance(parsed_trunc, list) and parsed_trunc[0]["id"] == "b4"

    te = TranslationEngine()
    res_dict = te._parse_json_response('{"translated_text": "单一结果"}')
    assert res_dict == [{"translated_text": "单一结果"}]

    mgr = TranslationManager()
    called = []
    local_prov = SimpleNamespace(
        config=ProviderConfig(provider_name="custom", api_key="", endpoint="http://127.0.0.1:11434/v1"),
        supports_vision=lambda: False,
        translate_text_blocks=lambda blks, **kw: called.append(len(blks)) or blks,
    )
    mgr.get_active_provider = lambda: local_prov
    mgr.translate([TranslationBlock(id="1", original_text="hello")])
    assert called == [1]


def test_batch_worker_raises_on_failed_export(tmp_path):
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    p = str(tmp_path / "p1.png")
    safe_cv2_imwrite(p, img, ext=".png")
    cm = MangaCacheManager()
    blk = TranslationBlock(id="b1", original_text="a", translated_text="b", xmin=10, ymin=10, xmax=50, ymax=50)
    cm.save_page_cache(p, erased_img=img, blocks=[blk], rendered_img=img)

    export_dir = str(tmp_path / "out")
    os.makedirs(export_dir, exist_ok=True)
    worker = BatchWorker([{"id": "1", "path": p}], {"auto_save_cache": False}, export_dir=export_dir)
    failed = []
    worker.sig_item_failed.connect(lambda iid, err: failed.append(err))
    with patch.object(MangaExporter, "export_hierarchical_image", return_value=False):
        worker.run()
    assert len(failed) == 1


def test_stroke_renderer_small_bold_alpha_and_typography_engine():
    font = ImageFont.load_default()
    im_plain = Image.new("RGB", (60, 40), (255, 255, 255))
    im_bold = Image.new("RGB", (60, 40), (255, 255, 255))
    StrokeRenderer.render_text_pil(im_plain, "AB", 10, 10, font, (0, 0, 0, 255), is_bold=False)
    StrokeRenderer.render_text_pil(im_bold, "AB", 10, 10, font, (0, 0, 0, 255), is_bold=True)
    # Force bold delta check at small font_size < 30 via mock font.size < 30
    font_small = MagicMock(wraps=font)
    font_small.size = 18
    im_s_plain = Image.new("RGB", (60, 40), (255, 255, 255))
    im_s_bold = Image.new("RGB", (60, 40), (255, 255, 255))
    StrokeRenderer.render_text_pil(im_s_plain, "AB", 10, 10, font_small, (0, 0, 0, 255), is_bold=False)
    StrokeRenderer.render_text_pil(im_s_bold, "AB", 10, 10, font_small, (0, 0, 0, 255), is_bold=True)
    assert np.sum(np.array(im_s_bold) != np.array(im_s_plain)) > 0

    im_alpha = Image.new("RGB", (60, 40), (200, 200, 200))
    StrokeRenderer.render_text_pil(im_alpha, "A", 10, 10, font_small, (255, 0, 0, 128))
    arr = np.array(im_alpha)
    assert arr[0, 0].tolist() == [200, 200, 200]

    eng = TypographyEngine()
    bg = np.full((240, 240, 3), 255, dtype=np.uint8)
    blocks = [
        TranslationBlock(
            id="rot1",
            original_text="ドキッ",
            translated_text="第一行\n第二行",
            xmin=20,
            ymin=20,
            xmax=70,
            ymax=80,
            angle=42.0,
            direction="vertical",
            line_colors=["#FF0000", "#0000FF"],
        )
    ]
    rendered = eng.render_page(bg, blocks, {"font_size_scale": 1.0})
    assert rendered.shape == bg.shape


def test_ui_components_and_main_window_regressions(qapp, tmp_path, monkeypatch):
    # 1. PageStyleDialog preserves unspecified fields (text_shadow, shadow_color) and accepts dict
    dlg = PageStyleDialog(
        item_data={"path": "p.png"},
        global_style={"text_shadow": True, "shadow_color": "#123456"},
        current_page_style={"text_shadow": True, "shadow_color": "#ABCDEF", "font_bold": False},
    )
    dlg._on_apply()
    assert dlg.applied_style is not None
    assert dlg.applied_style.text_shadow is True
    assert dlg.applied_style.shadow_color == "#ABCDEF"
    dlg.close()

    # 2. AsyncThumbnailManager separates cache keys by target_size and invalidates all sizes
    thumb_path = str(tmp_path / "thumb.png")
    safe_cv2_imwrite(thumb_path, np.full((120, 120, 3), 100, dtype=np.uint8), ext=".png")
    tm = AsyncThumbnailManager()
    tm._cache.clear()
    got = []
    tm.request_thumbnail(thumb_path, (44, 58), lambda pm: got.append((pm.width(), pm.height())), sync=True)
    tm.request_thumbnail(thumb_path, (96, 128), lambda pm: got.append((pm.width(), pm.height())), sync=True)
    assert len(got) == 2 and got[0] != got[1]
    assert (thumb_path, (44, 58)) in tm._cache and (thumb_path, (96, 128)) in tm._cache
    tm.invalidate(thumb_path)
    assert (thumb_path, (44, 58)) not in tm._cache and (thumb_path, (96, 128)) not in tm._cache

    # 3. ContextualTypographyBar syncs font_combo & InPlaceBubbleEditor handles text_edit keys
    bar = ContextualTypographyBar()
    bar.attach_block({"id": "b1", "font_family_override": "漫画体", "style": {}})
    assert bar.font_combo.currentText() == "漫画体"
    bar.attach_block({"id": "b2", "style": {}})
    assert bar.font_combo.currentText() == "默认字体"
    bar.close()

    ed = InPlaceBubbleEditor()
    ed.show()
    ed.attach_block({"id": "b1", "translated_text": "init"}, 1)
    ed.text_edit.setPlainText("updated")
    commits = []
    ed.sig_commit_and_next.connect(lambda b, t: commits.append(t))
    QApplication.sendEvent(ed.text_edit, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert commits == ["updated"]
    closed = []
    ed.sig_closed.connect(lambda: closed.append(True))
    QApplication.sendEvent(ed.text_edit, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert closed == [True]
    ed.close()

    # 4. InspectorPanel recognizes stroke_width_override
    insp = InspectorPanel()
    insp.set_blocks([{"id": "1", "original_text": "hi", "translated_text": "你好", "stroke_width_override": 3.5}])
    insp.select_block_by_id("1")
    assert insp.block_style_override_cb.isChecked() is True
    assert insp.style_controls_box.isEnabled() is True
    insp.close()

    # 5. Rotated BubbleItem resize works without NaN / box collapse
    item = BubbleItem({"id": "r1", "xmin": 20.0, "ymin": 20.0, "xmax": 60.0, "ymax": 40.0, "angle": 90.0}, 1000, 1000)
    item._active_handle = HANDLE_BOTTOM_RIGHT
    item._drag_start_pos = QPointF(600.0, 400.0)
    item._drag_start_rect = item.rect()
    item._drag_start_scene_pos = item.pos()
    fake_ev = SimpleNamespace(scenePos=lambda: QPointF(600.0, 440.0), accept=lambda: None)
    item.mouseMoveEvent(fake_ev)
    assert item.rect().width() > 400.0

    # 6. MainWindow theme persistence, clear cache filename, _re_render_all_pages style leak, session restore
    cfg_file = tmp_path / "desktop_config.json"
    session_file = tmp_path / ".amt_last_session.json"
    p_a = str(tmp_path / "a.png")
    p_b = str(tmp_path / "b.png")
    img_np = np.full((80, 80, 3), 245, dtype=np.uint8)
    safe_cv2_imwrite(p_a, img_np, ext=".png")
    safe_cv2_imwrite(p_b, img_np, ext=".png")
    session_file.write_text(json.dumps({"paths": [p_a, p_b], "active_path": p_b}), encoding="utf-8")

    def _fake_resolve(rel):
        if rel == "desktop_config.json":
            return str(cfg_file)
        if rel == ".amt_last_session.json":
            return str(session_file)
        return str(tmp_path / os.path.basename(rel))

    monkeypatch.setattr("app.core.config.resolve_app_path", _fake_resolve)
    win = MainWindow()
    win.set_theme("light")
    saved_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert saved_cfg["theme"] == "light"

    win.current_image_data = {
        "id": "pa",
        "path": p_a,
        "blocks": [{"id": "1", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50, "translated_text": "hi"}],
        "original_cv": img_np,
    }
    with patch.object(win, "_re_render_current_page"):
        win.page_list.items_data = [win.current_image_data]
        win._re_render_all_pages()
    assert "style" not in win.current_image_data

    # Stale pipeline step must not overwrite unrelated page
    win.active_worker = SimpleNamespace(image_path=p_b)
    prev_blocks = win.current_image_data["blocks"]
    win._on_pipeline_step_done("ocr", [])
    assert win.current_image_data["blocks"] is prev_blocks

    # Clear cache works without NameError on filename
    win.canvas_view.set_data(original_cv=img_np, blocks=[])
    win._on_canvas_clear_cache_requested()
    assert "清除" in win.status_label.text()

    # Session restore selects active_path (p_b, index 1), not index 0
    win.current_image_data = None
    win.page_list.clear_all()
    win.restore_last_session()
    deadline = time.time() + 5.0
    while len(win.page_list.items_data) < 2 and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    assert win.current_image_data is not None
    assert os.path.normpath(win.current_image_data["path"]) == os.path.normpath(p_b)
    win.close()
