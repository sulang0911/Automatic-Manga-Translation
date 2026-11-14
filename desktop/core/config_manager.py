import os
import json
from typing import Dict, Any
from app.core.config import DEFAULT_SYSTEM_PROMPT, normalize_system_prompt

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "desktop_config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    # Translation LLM Provider Settings
    "provider": "openai",  # "deepseek" | "openai" | "gemini" | "custom"
    "api_key": "1",
    "model": "richardyoung/qwen2.5-7b-instruct-abliterated:Q5_K_M",
    "custom_endpoint": "http://127.0.0.1:11434/v1",
    "target_lang": "简体中文",
    "source_lang": "自动识别",
    "temperature": 0.2,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,

    # OCR Settings
    "ocr_engine": "ctd",  # "ctd" (pure PyTorch) | "paddle" | "paddle_manga" | "easyocr" | "cpu_paddle"
    "use_gpu": True,
    "ocr_lang": "japan",  # "japan" | "ch" | "en" | "korean"
    "ocr_ensemble_detection": True,  # Dual-model detection fusion (CTD + General detector)
    "ocr_ensemble_recognition": True,  # Dual-model recognition ensemble (Manga-OCR + General recognizer)
    "ocr_rec_mode": "smart",  # "smart" | "easyocr" | "paddle_v5" | "manga_ocr"
    "ocr_en_recognizer": "ppocr_v6",  # "ppocr_v6" | "svtrv2" | "ppocr_v5_server" | "easyocr"
    "ocr_zh_recognizer": "ppocr_v5_server",  # "ppocr_v5_server" | "ppocr_v6" | "easyocr"
    "ocr_ja_recognizer": "manga_ocr",  # "manga_ocr" | "ppocr_v5_server" | "easyocr"
    "ocr_paddle_version": "PP-OCRv5",  # "PP-OCRv5" | "PP-OCRv3"
    "ocr_vlm_fallback": False,
    "ocr_vlm_conf_threshold": 0.72,

    # Inpaint Settings
    "inpaint_engine": "auto",  # "auto" | "lama" | "opencv_telea" | "opencv_ns"
    "bubble_dilation": 3,
    "onomatopoeia_dilation": 6,
    "feather_radius": 4,

    # Typography & Style Settings
    "font_family": "霞鹜文楷",
    "font_size_scale": 1.0,
    "auto_fit_font_size": True,
    "font_bold": False,
    "font_italic": False,
    "line_spacing": 1.30,
    "auto_line_spacing": True,
    "text_color": "#000000",
    "text_color_mode": "original",  # "custom" | "original"
    "bg_color": "#FFFFFF",
    "bg_color_mode": "original",  # "original" | "custom" | "none"
    "bg_opacity": 0.95,
    "stroke_mode": "auto",  # "auto" | "manual" | "off"
    "stroke_color": "#FFFFFF",
    "stroke_width": 3.5,
    "auto_stroke_width": True,
    "text_orientation": "auto",  # "auto" | "horizontal" | "vertical"
    "onomatopoeia_mode": "normal",  # "normal" | "transparent" | "ignore"

    # UI Preferences
    "theme": "dark",  # "dark" | "light" | "system"
    "view_mode": "side_by_side",  # "side_by_side" | "split_slider" | "translated" | "original" | "inpainted"
    "recent_export_dir": "",
}

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._data = DEFAULT_CONFIG.copy()
            cls._instance.load()
        return cls._instance

    def load(self):
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self._data.update(saved)
            except Exception as e:
                print(f"[ConfigManager] Failed to load config: {e}")
        self._data["system_prompt"] = normalize_system_prompt(self._data.get("system_prompt"))
        if isinstance(self._data.get("llm"), dict):
            self._data["llm"]["system_prompt"] = normalize_system_prompt(
                self._data["llm"].get("system_prompt")
            )

    def save(self):
        try:
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ConfigManager] Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default if default is not None else DEFAULT_CONFIG.get(key))

    def set(self, key: str, value: Any):
        self._data[key] = value
        self.save()

    def update(self, new_data: Dict[str, Any]):
        self._data.update(new_data)
        self.save()

    @property
    def data(self) -> Dict[str, Any]:
        return self._data
