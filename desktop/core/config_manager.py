import os
import json
from typing import Dict, Any

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "desktop_config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    # Translation LLM Provider Settings
    "provider": "deepseek",  # "deepseek" | "openai" | "gemini" | "custom"
    "api_key": "",
    "model": "deepseek-chat",
    "custom_endpoint": "https://api.deepseek.com/v1",
    "target_lang": "简体中文",
    "source_lang": "日语",
    "temperature": 0.3,
    "system_prompt": (
        "You are an expert manga localization and translation specialist. "
        "Translate the extracted manga dialogue text accurately and naturally into the target language, "
        "preserving colloquial expressions, humor, character personality, and atmospheric tone. "
        "Keep character names and sound effects (onomatopoeia) culturally appropriate."
    ),

    # OCR Settings
    "ocr_engine": "paddle",  # "paddle" | "easyocr" | "cpu_paddle"
    "use_gpu": True,
    "ocr_lang": "japan",  # "japan" | "ch" | "en" | "korean"

    # Inpaint Settings
    "inpaint_engine": "auto",  # "auto" | "lama" | "opencv_telea" | "opencv_ns"
    "bubble_dilation": 3,
    "onomatopoeia_dilation": 6,
    "feather_radius": 4,

    # Typography & Style Settings
    "font_family": "Microsoft YaHei",
    "font_size_scale": 1.0,
    "auto_fit_font_size": True,
    "font_bold": False,
    "font_italic": False,
    "text_color": "#000000",
    "text_color_mode": "custom",  # "custom" | "original"
    "bg_color": "#FFFFFF",
    "bg_color_mode": "original",  # "original" | "custom" | "none"
    "bg_opacity": 0.95,
    "stroke_mode": "auto",  # "auto" | "manual" | "off"
    "stroke_color": "#FFFFFF",
    "stroke_width": 2,
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
