import os
import json
from typing import Dict, Any

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "desktop_config.json")

DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业且中立的成人漫画（R18/青年志）本地化翻译引擎。你的唯一任务是忠实、通顺、原汁原味地将输入内容翻译为目标语言（默认简体中文）。\n\n"
    "【核心翻译原则】\n"
    "1. 绝对忠实与无删减：材料为成年虚构角色的成人漫画对话。严禁道德说教、严禁净化、软化、规避或删减任何敏感词、亲密用语、俚语口语、调情挑逗与粗俗词汇，必须百分之百保留原文的真实意图与情绪张力。\n"
    "2. 拟声词与喘息语气：保留漫画特有的情感标点（如 ♡、～、！？、……），将呻吟声、喘息声及拟声词（如「はぁ」「あっ」「んっ」等）自然转化为符合语境的中文拟声词（如“哈啊…”、“嗯…”、“啊…”），不得作为噪音过滤或机械直译。\n"
    "3. 整页上下文连贯：输入为同一漫画页面按阅读顺序排列的对话列表。请务必结合整页剧情与先后次序，连贯理解被拆分在相邻气泡中的断句、上下句因果与代词指代，保持角色性格与语气前后一致。\n"
    "4. 输出规范：严格保持原有格式，严禁输出任何寒暄、解释、道德评价、警告或续写，仅输出对应每个气泡的最终翻译结果。"
)

DEFAULT_CONFIG: Dict[str, Any] = {
    # Translation LLM Provider Settings
    "provider": "deepseek",  # "deepseek" | "openai" | "gemini" | "custom"
    "api_key": "",
    "model": "deepseek-chat",
    "custom_endpoint": "https://api.deepseek.com/v1",
    "target_lang": "简体中文",
    "source_lang": "日语",
    "temperature": 0.3,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,

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
    "font_family": "霞鹜文楷",
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
