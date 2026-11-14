"""
tests/unit/test_onomatopoeia.py
Unit tests verifying that onomatopoeia blocks (e.g. orange-framed tone words and SFX)
are translated, inpainted, and rendered by default without being hidden.
"""
import os
import cv2
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


def test_settings_dialog_onomatopoeia_combo(qapp, tmp_path, monkeypatch):
    """
    Verifies SettingsDialog loads and saves onomatopoeia_mode to config.
    """
    monkeypatch.chdir(tmp_path)
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


def test_translation_manager_skips_and_clears_onomatopoeia_in_ignore_mode():
    """
    Verifies TranslationManager in 'ignore' mode:
    1. Does not send pre-tagged ignored onomatopoeia to provider (and clears stale text).
    2. Clears translated_text if LLM reclassifies a bubble-tagged SFX to 'onomatopoeia'.
    3. Still translates force_erase=True onomatopoeia and normal bubbles.
    4. Skips ignored onomatopoeia even in local demo mode (no API key).
    """
    from unittest.mock import MagicMock
    from app.core.translation.manager import TranslationManager
    from app.core.translation.base import ProviderConfig

    mgr = TranslationManager()
    # Demo mode (empty API key): must NOT generate 【译】 on ignored onomatopoeia
    mgr.set_active_provider("openai", ProviderConfig(provider_name="openai", api_key="", model="gpt-4o-mini"))
    demo_blocks = [
        TranslationBlock(id="d_sfx", original_text="ゴゴゴ", type="onomatopoeia"),
        TranslationBlock(id="d_bub", original_text="こんにちは", type="bubble"),
    ]
    res_demo = mgr.translate(demo_blocks, onomatopoeia_mode="ignore")
    assert res_demo[0].translated_text == ""
    assert res_demo[1].translated_text == "【译】こんにちは"

    # Real provider mode: verifies pre-tagged SFX omitted + LLM-tagged SFX cleared
    mock_prov = MagicMock()
    mock_prov.config = ProviderConfig(provider_name="openai", api_key="sk-test", model="gpt-4o-mini")
    mock_prov.supports_vision.return_value = False
    sent_ids = []

    def fake_translate(blocks, **kwargs):
        sent_ids.extend(b.id for b in blocks)
        for b in blocks:
            if b.id == "b_hidden_sfx":
                b.type = "onomatopoeia"
                b.translated_text = "咚咚"
            else:
                b.translated_text = f"译:{b.original_text}"
        return blocks

    mock_prov.translate_text_blocks.side_effect = fake_translate
    mgr._provider_instances["openai"] = mock_prov
    mgr._active_provider_name = "openai"

    blocks = [
        TranslationBlock(id="b_pre_sfx", original_text="ザッ", translated_text="旧译文", type="onomatopoeia", force_erase=False),
        TranslationBlock(id="b_hidden_sfx", original_text="ドン！", type="bubble"),
        TranslationBlock(id="b_force_sfx", original_text="バン！", type="onomatopoeia", force_erase=True),
        TranslationBlock(id="b_dialogue", original_text="行くぞ", type="bubble"),
    ]
    out = mgr.translate(blocks, onomatopoeia_mode="ignore")
    assert "b_pre_sfx" not in sent_ids
    assert set(sent_ids) == {"b_hidden_sfx", "b_force_sfx", "b_dialogue"}
    assert out[0].translated_text == ""
    assert out[1].type == "onomatopoeia"
    assert out[1].translated_text == ""
    assert out[2].translated_text == "译:バン！"
    assert out[3].translated_text == "译:行くぞ"


def test_align_translations_preserves_onomatopoeia_when_llm_omits_type():
    """Omitted 'type' from LLM output must not reset an onomatopoeia block back to 'bubble'."""
    from app.core.translation.json_parser import align_translations_to_blocks

    blocks = [TranslationBlock(id="s1", original_text="ドキッ", type="onomatopoeia")]
    align_translations_to_blocks([{"id": "s1", "translated_text": "扑通"}], blocks)
    assert blocks[0].type == "onomatopoeia"


def test_pipeline_and_batch_worker_preserve_onomatopoeia_in_ignore_mode(qapp, tmp_path, monkeypatch):
    """
    End-to-end verification across PipelineWorker and BatchWorker with onomatopoeia_mode='ignore':
    Both OCR-detected and LLM-reclassified onomatopoeia are neither translated nor erased,
    preserving exact original pixels while regular speech bubbles are translated and inpainted.
    """
    from unittest.mock import patch, MagicMock
    from app.core.cache.cache_manager import safe_cv2_imwrite
    from app.core.pipeline.pipeline_worker import PipelineWorker
    from app.core.pipeline.batch_worker import BatchWorker

    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Bubble region (10..80, 10..80) -> red text inside
    cv2.putText(img, "HI", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    # Pre-tagged SFX region (110..180, 20..80) -> green stroke
    cv2.line(img, (120, 30), (170, 70), (0, 180, 0), 3)
    # Unboxed SFX initially misclassified as bubble by OCR (110..180, 110..180) -> blue stroke
    cv2.line(img, (120, 120), (170, 170), (200, 0, 0), 3)

    img_path = str(tmp_path / "manga_sfx_page.png")
    safe_cv2_imwrite(img_path, img, ext=".png")

    def make_ocr_blocks():
        return [
            TranslationBlock.from_pixel_box(10, 10, 80, 80, 200, 200, id="b1", original_text="Hi!", type="bubble").to_dict(),
            TranslationBlock.from_pixel_box(110, 20, 180, 80, 200, 200, id="sfx_ocr", original_text="ゴゴ", type="onomatopoeia").to_dict(),
            TranslationBlock.from_pixel_box(110, 110, 180, 180, 200, 200, id="sfx_llm", original_text="ドン！", type="bubble").to_dict(),
        ]

    class FakeProvider:
        config = type("Cfg", (), {"api_key": "sk-fake", "provider_name": "openai"})()
        def supports_vision(self):
            return False
        def translate_text_blocks(self, blocks, **kwargs):
            for b in blocks:
                assert b.id != "sfx_ocr", "Ignored onomatopoeia must not be sent to LLM"
                if b.id == "sfx_llm":
                    b.type = "onomatopoeia"
                    b.translated_text = "咚！"
                else:
                    b.translated_text = "你好！"
            return blocks

    class FakeMgr:
        _chapter_context = type("Ctx", (), {"glossary": {}, "character_notes": {}, "previous_summary": "", "reading_direction": ""})()
        def set_active_provider(self, *args, **kwargs):
            pass
        def get_active_provider(self):
            return FakeProvider()
        def translate(self, blocks, **kwargs):
            from app.core.translation.manager import TranslationManager
            real_mgr = TranslationManager()
            real_mgr.get_active_provider = lambda: FakeProvider()
            return real_mgr.translate(blocks, **kwargs)

    cfg = {
        "inpaint_engine": "opencv",
        "inpaint": {"engine": "opencv"},
        "style": {"onomatopoeia_mode": "ignore"},
        "provider": "openai",
        "api_key": "sk-fake",
        "model": "gpt-4o-mini",
    }

    with patch("app.core.pipeline.pipeline_worker.OCREngine") as mock_ocr_cls, \
         patch("app.core.pipeline.pipeline_worker.TranslationManager.get_instance", return_value=FakeMgr()):
        mock_ocr = MagicMock()
        mock_ocr.detect_and_recognize.return_value = make_ocr_blocks()
        mock_ocr_cls.return_value = mock_ocr

        pw = PipelineWorker(img_path, cfg, mode="full")
        finished = []
        pw.sig_finished.connect(lambda res: finished.append(res))
        pw.run()

        assert len(finished) == 1
        res = finished[0]
        res_blocks = { (b["id"] if isinstance(b, dict) else b.id): b for b in res["blocks"] }
        assert res_blocks["b1"]["translated_text"] == "你好！"
        assert res_blocks["sfx_ocr"]["translated_text"] == ""
        assert res_blocks["sfx_llm"]["type"] == "onomatopoeia"
        assert res_blocks["sfx_llm"]["translated_text"] == ""

        # Original SFX artwork pixels must remain 100% intact in both erased_img and translated_img
        assert np.array_equal(res["erased_img"][20:80, 110:180], img[20:80, 110:180])
        assert np.array_equal(res["translated_img"][20:80, 110:180], img[20:80, 110:180])
        assert np.array_equal(res["erased_img"][110:180, 110:180], img[110:180, 110:180])
        assert np.array_equal(res["translated_img"][110:180, 110:180], img[110:180, 110:180])

    # BatchWorker verification with fresh export dir
    batch_img_path = str(tmp_path / "batch_sfx_page.png")
    safe_cv2_imwrite(batch_img_path, img, ext=".png")
    export_dir = str(tmp_path / "export")
    with patch("app.core.pipeline.batch_worker.OCREngine") as mock_bocr_cls, \
         patch("app.core.pipeline.batch_worker.TranslationManager.get_instance", return_value=FakeMgr()):
        mock_bocr = MagicMock()
        mock_bocr.detect_and_recognize.return_value = make_ocr_blocks()
        mock_bocr_cls.return_value = mock_bocr

        bw = BatchWorker([{"id": "p0", "path": batch_img_path}], cfg, export_dir=export_dir)
        bw.run()

        exported = cv2.imread(os.path.join(export_dir, "batch_sfx_page.png"))
        assert exported is not None
        assert np.array_equal(exported[20:80, 110:180], img[20:80, 110:180])
        assert np.array_equal(exported[110:180, 110:180], img[110:180, 110:180])


def test_typography_render_page_respects_force_erase_in_ignore_mode():
    """force_erase=True renders an onomatopoeia block even when onomatopoeia_mode is 'ignore'."""
    canvas = np.full((240, 240, 3), 255, dtype=np.uint8)
    b_skip = TranslationBlock(id="s1", xmin=5, ymin=5, xmax=45, ymax=45, original_text="ゴ", translated_text="轰", type="onomatopoeia", force_erase=False)
    b_force = TranslationBlock(id="s2", xmin=50, ymin=50, xmax=95, ymax=95, original_text="ド", translated_text="咚", type="onomatopoeia", force_erase=True)
    out = TypographyEngine().render_page(canvas, [b_skip, b_force], StyleConfig(onomatopoeia_mode="ignore"))
    # Skipped area remains pure white; force_erase area has rendered text
    assert np.all(out[10:105, 10:105] == 255)
    assert not np.all(out[120:228, 120:228] == 255)


def test_block_worker_skips_ignored_onomatopoeia(qapp, monkeypatch):
    """Manual box OCR/translate skips translation and erasure when detected text is ignored onomatopoeia."""
    from app.core.pipeline.block_worker import BlockOcrTranslateWorker

    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img, (45, 45), (115, 75), (0, 0, 220), 3)

    class MockOCREngine:
        def __init__(self, *args, **kwargs):
            pass
        def detect_and_recognize(self, crop, **kwargs):
            return [{"xmin": 20.0, "ymin": 20.0, "xmax": 60.0, "ymax": 40.0, "original_text": "ドキッ", "type": "bubble"}]

    monkeypatch.setattr("app.core.pipeline.block_worker.OCREngine", MockOCREngine)

    class FakeMgr:
        def set_active_provider(self, *args, **kwargs):
            pass
        def translate(self, blocks, **kwargs):
            from app.core.translation.manager import TranslationManager
            class FakeProv:
                config = type("Cfg", (), {"api_key": "sk-fake", "provider_name": "openai"})()
                def supports_vision(self): return False
                def translate_text_blocks(self, blks, **kw):
                    for b in blks:
                        b.type = "onomatopoeia"
                        b.translated_text = "扑通"
                    return blks
            real = TranslationManager()
            real.get_active_provider = lambda: FakeProv()
            return real.translate(blocks, **kwargs)

    monkeypatch.setattr("app.core.pipeline.block_worker.TranslationManager.get_instance", lambda *a, **k: FakeMgr())

    tb = {"id": "m1", "xmin": 20.0, "ymin": 20.0, "xmax": 60.0, "ymax": 40.0, "original_text": "", "translated_text": "", "type": "bubble"}
    worker = BlockOcrTranslateWorker(
        image_path="",
        original_cv=img,
        target_block=tb,
        all_blocks=[tb],
        existing_erased=None,
        config={"style": {"onomatopoeia_mode": "ignore"}},
    )
    res_list = []
    worker.sig_completed.connect(lambda r: res_list.append(r))
    worker.run()

    assert len(res_list) == 1
    res = res_list[0]
    assert res["target_block"]["type"] == "onomatopoeia"
    assert res["target_block"]["translated_text"] == ""
    assert np.array_equal(res["erased_img"], img)
    assert np.array_equal(res["translated_img"], img)


def test_ocr_aspect_classification_preserves_textured_sfx():
    """Verify extreme-aspect textured SFX (border_std >= 25, mid luma) classifies as onomatopoeia while uniform white banner stays bubble."""
    from desktop.core.ocr_engine import OCREngine
    noisy_crop = np.full((30, 180, 3), 40, dtype=np.uint8)
    noisy_crop[:, ::2] = 200  # High-contrast manga screentone/hatching (std ~ 80, mid luma ~ 120)
    white_banner = np.full((30, 180, 3), 250, dtype=np.uint8)

    # Full-page wrapper test on synthetic noisy region vs uniform white banner
    img = np.full((300, 300, 3), 140, dtype=np.uint8)
    img[20:50, 20:200] = noisy_crop
    img[100:130, 20:200] = white_banner

    eng = OCREngine(engine_type="ctd", use_gpu=False, lang="ja", is_manual=True, ja_recognizer="paddle")
    monkey_boxes = [
        {"xmin": 20, "ymin": 20, "xmax": 200, "ymax": 50, "text": "ゴゴゴゴ", "conf": 0.95, "lines": []},
        {"xmin": 20, "ymin": 100, "xmax": 200, "ymax": 130, "text": "Chapter 1", "conf": 0.95, "lines": []},
    ]
    eng._ctd_detector = type("MockCTD", (), {"detect": lambda s, im: ([dict(b) for b in monkey_boxes], None)})()
    eng._recognize_region = lambda im, norm_lang="ja", box=None, angle=0.0: (box["text"], box["conf"])

    blocks = eng.detect_and_recognize(img)
    by_text = {b["original_text"]: b for b in blocks}
    assert by_text["ゴゴゴゴ"]["type"] == "onomatopoeia"
    assert by_text["Chapter 1"]["type"] == "bubble"


def test_cached_erased_restores_sfx_on_rerender_and_archive_export(qapp, tmp_path):
    """
    Verifies that even when a cached erased_img previously erased an SFX region,
    switching to onomatopoeia_mode='ignore' restores exact original SFX pixels during
    MainWindow._re_render_single_page and _ChapterArchiveExportWorker archive export.
    """
    import zipfile
    from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imwrite
    from app.ui.main_window import MainWindow, _ChapterArchiveExportWorker

    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img, (115, 115), (175, 175), (20, 20, 220), 4)
    erased = np.full((200, 200, 3), 255, dtype=np.uint8)  # simulated previously erased SFX

    img_path = str(tmp_path / "cached_sfx.png")
    safe_cv2_imwrite(img_path, img, ext=".png")

    sfx_block = TranslationBlock.from_pixel_box(110, 110, 180, 180, 200, 200, id="ig_1", original_text="バン", translated_text="砰", type="onomatopoeia")
    bubble_block = TranslationBlock.from_pixel_box(10, 10, 80, 80, 200, 200, id="b_1", original_text="Hi", translated_text="嗨", type="bubble")
    cache_mgr = get_cache_manager()
    cache_mgr.save_page_cache(img_path, erased_img=erased, blocks=[bubble_block, sfx_block])

    win = MainWindow()
    item = {"id": "p1", "path": img_path, "blocks": [bubble_block, sfx_block], "erased_img": erased}
    win._re_render_single_page(item, StyleConfig(onomatopoeia_mode="ignore"))
    assert item["translated_img"] is not None
    assert np.array_equal(item["translated_img"][110:180, 110:180], img[110:180, 110:180])

    # Clear rendered cache and verify _ChapterArchiveExportWorker restores SFX too (uncompressed PNG mode)
    cache_mgr.invalidate_rendered_cache(img_path)

    zip_path = str(tmp_path / "out.cbz")
    item_zip = {"id": "p1", "path": img_path, "status": "completed", "style": StyleConfig(onomatopoeia_mode="ignore")}
    cfg_cbz = win.config.to_dict()
    cfg_cbz["style"]["export_compressed"] = False
    exp_worker = _ChapterArchiveExportWorker([item_zip], cfg_cbz, zip_path, "cbz")
    exp_worker.run()
    assert os.path.exists(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".png")]
        assert len(names) == 1
        data = np.frombuffer(zf.read(names[0]), dtype=np.uint8)
        exported_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        assert np.array_equal(exported_img[110:180, 110:180], img[110:180, 110:180])

    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_floating_textured_dialogue_keeps_bubble_across_ocr_classification():
    """
    Wide floating subtitles/dialogue over textured 3D/shaded artwork (aspect > 4.0,
    border_std >= 25, mid luma) must stay 'bubble', while genuine textured SFX keeps 'onomatopoeia'.
    """
    from app.core.ocr.base import classify_ocr_block_type
    from desktop.core.ocr_engine import OCREngine

    noisy_strip = np.full((30, 220, 3), 55, dtype=np.uint8)
    noisy_strip[:, ::2] = 195  # Textured dark/light 3D background (std ~ 70, mid luma ~ 125)

    dialogue_samples = [
        '(Sun) "There flowing on the surface of its spinning roller Whether you\'re ticklish or not, It\'s designed to give you the ultimate tickling.\n(Jun) "...Impressive."',
        '(Suk) "Injecting t',
        'orgasm controller.',
        '大丈夫なの…？',
        'ちょっとまってよ！',
        '就连插入控制器带来的疼痛',
        '잠깐만 기다려',
    ]
    for txt in dialogue_samples:
        assert classify_ocr_block_type(noisy_strip, txt, 0, 0, 220, 30) == "bubble", f"Misclassified dialogue: {txt}"

    sfx_samples = [
        'KNOCK!!',
        'GOOOHHH',
        'HAGH! AHHG!!',
        'ゴゴゴゴ',
        'バン!!',
        'どきどき',
        '두근두근',
    ]
    for sfx in sfx_samples:
        assert classify_ocr_block_type(noisy_strip, sfx, 0, 0, 220, 30) == "onomatopoeia", f"Missed SFX: {sfx}"

    # Verify full OCREngine.detect_and_recognize integration preserves floating dialogue
    img = np.full((400, 500, 3), 130, dtype=np.uint8)
    boxes = []
    texts = ['(Suk) "Injecting t', 'orgasm controller.', 'KNOCK!!']
    for idx, t in enumerate(texts):
        y1 = 20 + idx * 60
        img[y1:y1 + 30, 20:240] = noisy_strip
        boxes.append({"xmin": 20, "ymin": y1, "xmax": 240, "ymax": y1 + 30, "text": t, "conf": 0.92, "lines": []})

    eng = OCREngine(engine_type="ctd", use_gpu=False, lang="en", is_manual=True)
    eng._ctd_detector = type("MockCTD", (), {"detect": lambda s, im: ([dict(b) for b in boxes], None)})()
    eng._recognize_region = lambda im, norm_lang="en", box=None, angle=0.0: (box["text"], box["conf"])

    out = {b["original_text"]: b for b in eng.detect_and_recognize(img)}
    assert out['(Suk) "Injecting t']["type"] == "bubble"
    assert out['orgasm controller.']["type"] == "bubble"
    assert out['KNOCK!!']["type"] == "onomatopoeia"


def test_align_and_cache_heal_reject_onomatopoeia_on_dialogue_prose():
    """
    LLM hallucinations tagging spoken dialogue prose as 'onomatopoeia' are normalized back to 'bubble',
    prompt templates explicitly include unboxed dialogue as 'bubble', and stale cached blocks self-heal.
    """
    from app.core.pipeline.utils import heal_mislabeled_onomatopoeia_blocks
    from app.core.translation.json_parser import align_translations_to_blocks
    from app.core.translation.manager import TranslationManager
    from app.core.translation.prompt_templates import PromptTemplates

    sys_prompt = PromptTemplates.build_text_system_prompt("auto", "简体中文")
    assert "without a drawn speech bubble" in sys_prompt
    vis_prompt = PromptTemplates.build_vision_system_prompt("简体中文")
    assert "without a drawn bubble" in vis_prompt

    b_dlg = TranslationBlock(id="d1", original_text="orgasm controller.", type="bubble")
    b_sfx = TranslationBlock(id="s1", original_text="ザワ", type="bubble")
    aligned = align_translations_to_blocks(
        [
            {"id": "d1", "original_text": "orgasm controller.", "translated_text": "高潮控制器。", "type": "onomatopoeia"},
            {"id": "s1", "original_text": "ザワ", "translated_text": "沙沙", "type": "onomatopoeia"},
        ],
        [b_dlg, b_sfx],
    )
    assert aligned[0].type == "bubble"
    assert aligned[0].translated_text == "高潮控制器。"
    assert aligned[1].type == "onomatopoeia"

    class FakeProvider:
        config = type("Cfg", (), {"api_key": "sk-fake", "provider_name": "openai"})()
        def supports_vision(self): return False
        def translate_text_blocks(self, blks, **kw):
            return align_translations_to_blocks(
                [
                    {"id": "d1", "translated_text": "高潮控制器。", "type": "onomatopoeia"},
                    {"id": "s1", "translated_text": "沙沙", "type": "onomatopoeia"},
                ],
                blks,
            )

    mgr = TranslationManager()
    mgr.get_active_provider = lambda: FakeProvider()
    res = mgr.translate(
        [
            TranslationBlock(id="d1", original_text="orgasm controller.", type="bubble"),
            TranslationBlock(id="s1", original_text="ザワ", type="bubble"),
        ],
        onomatopoeia_mode="ignore",
    )
    assert res[0].type == "bubble"
    assert res[0].translated_text == "高潮控制器。"
    assert res[1].type == "onomatopoeia"
    assert res[1].translated_text == ""

    # Stale cache self-healing
    healed = heal_mislabeled_onomatopoeia_blocks([
        {"id": "c1", "original_text": '(Suk) "Injecting t', "translated_text": "", "type": "onomatopoeia"},
        {"id": "c2", "original_text": "KNOCK!!", "translated_text": "", "type": "onomatopoeia"},
    ])
    assert healed[0]["type"] == "bubble"
    assert healed[1]["type"] == "onomatopoeia"


def test_pytest_config_isolation_leaves_live_desktop_config_untouched(qapp):
    """Config writes triggered by MainWindow / ConfigManager inside pytest must never mutate REPO_ROOT/desktop_config.json."""
    import hashlib
    from app.core.config import resolve_app_path
    from app.ui.main_window import MainWindow
    from desktop.core.config_manager import ConfigManager

    live_cfg = resolve_app_path("desktop_config.json")
    before = hashlib.sha256(open(live_cfg, "rb").read()).hexdigest() if os.path.exists(live_cfg) else None

    win = MainWindow()
    win.set_theme("light")
    win.set_theme("dark")
    ConfigManager().set("theme", "light")
    ConfigManager().set("theme", "dark")
    win.close()
    win.deleteLater()
    qapp.processEvents()

    after = hashlib.sha256(open(live_cfg, "rb").read()).hexdigest() if os.path.exists(live_cfg) else None
    assert before == after
