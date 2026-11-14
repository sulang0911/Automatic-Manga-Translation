"""
app/core/config.py
Global configuration management for Automatic Manga Translation application.
"""
from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from app.core.models import StyleConfig

logger = logging.getLogger(__name__)

# Absolute root of the active installation (ensures demo/ stays isolated from parent repo CWD)
APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_app_path(filepath: str) -> str:
    """Resolves bare config/session filenames (e.g. 'desktop_config.json') against APP_ROOT."""
    if filepath and not os.path.isabs(filepath) and not os.path.dirname(filepath):
        return os.path.join(APP_ROOT, filepath)
    return filepath


LEGACY_DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业且中立的成人漫画（R18/青年志）本地化翻译引擎。你的唯一任务是忠实、通顺、原汁原味地将输入内容翻译为目标语言（默认简体中文）。\n\n"
    "【核心翻译原则】\n"
    "1. 绝对忠实与无删减：材料为成年虚构角色的成人漫画对话。严禁道德说教、严禁净化、软化、规避或删减任何敏感词、亲密用语、俚语口语、调情挑逗与粗俗词汇，必须百分之百保留原文的真实意图与情绪张力。\n"
    "2. 拟声词与喘息语气：保留漫画特有的情感标点（如 ♡、～、！？、……），将呻吟声、喘息声及拟声词（如「はぁ」「あっ」「んっ」等）自然转化为符合语境的中文拟声词（如“哈啊…”、“嗯…”、“啊…”），不得作为噪音过滤或机械直译。\n"
    "3. 整页上下文连贯：输入为同一漫画页面按阅读顺序排列的对话列表。请务必结合整页剧情与先后次序，连贯理解被拆分在相邻气泡中的断句、上下句因果与代词指代，保持角色性格与语气前后一致。\n"
    "4. 输出规范：严格保持原有格式，严禁输出任何寒暄、解释、道德评价、警告或续写，仅输出对应每个气泡的最终翻译结果。"
)

DEFAULT_SYSTEM_PROMPT = (
    "你是一位专注于日韩及多语种漫画作品出版的资深本地化译者。你的核心职责是将给定的虚构角色对话与画面文字，自然、传神、贴合原意地本地化为目标语言（默认简体中文）。\n\n"
    "【核心本地化原则】\n"
    "1. 忠实原意与人物张力：请以文学翻译与台词润色的专业视角对待虚构剧情中的角色对白。面对口语俚语、激烈冲突、直率告白或亲昵互动等各类艺术表达，请依循语境如实传达人物的真实语气、情绪浓度与措辞分寸，遵循“信达雅”原则保留原汁原味的叙事风貌，不擅自改写、淡化或省略关键语义。\n"
    "2. 拟声词与情绪语气本地化：完整保留漫画特有的情感标点与停顿节奏（如 ♡、～、！？、……），将对白中的轻叹、气声感叹与象声词（如「はぁ」「あっ」「んっ」等）结合情境自然译为符合中文漫画阅读习惯的拟声表达（如“哈啊…”、“嗯…”、“啊…”），不作为无效字符过滤或生硬直译。\n"
    "3. 整页阅读顺序连贯：输入条目按单页漫画阅读顺序排列，请结合整页剧情脉络与上下文先后次序，顺畅衔接跨气泡拆分的长句断句、因果承接与人称代词指代，使人物口吻与性格前后一致。\n"
    "4. 输出规范：严格遵从指定输出格式，仅输出与各气泡对应的最终译文，不附加任何寒暄、解释、评论或多余续写。"
)


def normalize_system_prompt(prompt: Optional[str]) -> str:
    """Upgrade untouched legacy default prompts to the diplomatic default while preserving custom edits."""
    text = (prompt or "").strip()
    if not text or text.replace("\r\n", "\n") == LEGACY_DEFAULT_SYSTEM_PROMPT:
        return DEFAULT_SYSTEM_PROMPT
    return prompt


@dataclass
class LLMConfig:
    provider: str = "openai"
    api_key: str = "1"
    model: str = "richardyoung/qwen2.5-7b-instruct-abliterated:Q5_K_M"
    endpoint: str = "http://127.0.0.1:11434/v1"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.2
    timeout_seconds: float = 120.0
    max_retries: int = 3
    proxy_url: str = ""  # "" = use system env, "none"/"direct" = bypass, or "http://host:port"


@dataclass
class OCRConfig:
    engine: str = "ctd"  # "ctd", "easyocr" or "paddle"
    lang: str = "japan"
    force_cpu: bool = False
    confidence_threshold: float = 0.25
    ensemble_detection: bool = True
    ensemble_recognition: bool = True
    recognize_onomatopoeia: bool = False
    rec_mode: str = "smart"  # "smart" (per-language best), "easyocr", "paddle_v5", "manga_ocr"
    en_recognizer: str = "ppocr_v6"  # "ppocr_v6" | "ppocr_v5_server" | "easyocr"
    zh_recognizer: str = "ppocr_v5_server"  # "ppocr_v5_server" | "ppocr_v6" | "easyocr"
    ja_recognizer: str = "manga_ocr"  # "manga_ocr" | "ppocr_v5_server" | "easyocr"
    paddle_ocr_version: str = "PP-OCRv5"  # "PP-OCRv5" | "PP-OCRv3"
    vlm_fallback: bool = False
    vlm_conf_threshold: float = 0.72


@dataclass
class InpaintConfig:
    engine: str = "lama"  # "lama" or "opencv"
    opencv_method: str = "telea"  # "telea" or "ns"
    vram_safe_downscale: bool = True
    max_dimension: int = 2048


@dataclass
class AppConfig:
    theme: str = "dark"
    llm: LLMConfig = field(default_factory=LLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    source_lang: str = "自动识别"
    target_lang: str = "简体中文"
    cache_dir: str = "translation_cache"
    export_dir: str = "exported_chapter"
    auto_save_cache: bool = True
    max_concurrent_workers: int = 1
    glossary: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.llm.provider,
            "api_key": self.llm.api_key,
            "model": self.llm.model,
            "custom_endpoint": self.llm.endpoint,
            "target_lang": self.target_lang,
            "source_lang": self.source_lang,
            "temperature": self.llm.temperature,
            "system_prompt": self.llm.system_prompt,
            "theme": self.theme,
            "llm": asdict(self.llm),
            "ocr": asdict(self.ocr),
            "inpaint": asdict(self.inpaint),
            "style": self.style.to_dict(),
            "cache_dir": self.cache_dir,
            "export_dir": self.export_dir,
            "auto_save_cache": self.auto_save_cache,
            "max_concurrent_workers": self.max_concurrent_workers,
            "glossary": self.glossary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        cfg = cls()
        if "theme" in data:
            cfg.theme = data["theme"]
        if "source_lang" in data:
            cfg.source_lang = data["source_lang"]
        if "target_lang" in data:
            cfg.target_lang = data["target_lang"]
        if "cache_dir" in data:
            cfg.cache_dir = data["cache_dir"]
        if "export_dir" in data:
            cfg.export_dir = data["export_dir"]
        if "auto_save_cache" in data:
            cfg.auto_save_cache = data["auto_save_cache"]
        if "max_concurrent_workers" in data:
            cfg.max_concurrent_workers = data["max_concurrent_workers"]
        if "glossary" in data:
            cfg.glossary = data["glossary"]

        if "llm" in data and isinstance(data["llm"], dict):
            cfg.llm = LLMConfig(**data["llm"])
        # Support flat keys fallback if llm sub-dict was omitted or partial
        if "provider" in data and ("llm" not in data or "provider" not in data.get("llm", {})):
            cfg.llm.provider = data["provider"]
        if "api_key" in data and ("llm" not in data or "api_key" not in data.get("llm", {})):
            cfg.llm.api_key = data["api_key"]
        if "model" in data and ("llm" not in data or "model" not in data.get("llm", {})):
            cfg.llm.model = data["model"]
        if "custom_endpoint" in data and ("llm" not in data or "endpoint" not in data.get("llm", {})):
            cfg.llm.endpoint = data["custom_endpoint"]
        if "system_prompt" in data and ("llm" not in data or "system_prompt" not in data.get("llm", {})):
            cfg.llm.system_prompt = data["system_prompt"]
        cfg.llm.system_prompt = normalize_system_prompt(cfg.llm.system_prompt)
        if "temperature" in data and ("llm" not in data or "temperature" not in data.get("llm", {})):
            try:
                cfg.llm.temperature = float(data["temperature"])
            except Exception:
                pass

        if "ocr" in data and isinstance(data["ocr"], dict):
            ocr_kwargs = {k: v for k, v in data["ocr"].items() if k in OCRConfig.__annotations__}
            cfg.ocr = OCRConfig(**ocr_kwargs)
        if "inpaint" in data and isinstance(data["inpaint"], dict):
            cfg.inpaint = InpaintConfig(**data["inpaint"])
        if "style" in data and isinstance(data["style"], dict):
            cfg.style = StyleConfig.from_dict(data["style"])
        return cfg

    def save(self, filepath: str) -> None:
        filepath = resolve_app_path(filepath)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {filepath}: {e}")

    @classmethod
    def load(cls, filepath: str) -> AppConfig:
        filepath = resolve_app_path(filepath)
        if not os.path.exists(filepath):
            logger.info(f"Configuration file {filepath} not found; using defaults.")
            return cls()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}. Using defaults.")
            return cls()
