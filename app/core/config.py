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


DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业且中立的成人漫画（R18/青年志）本地化翻译引擎。你的唯一任务是忠实、通顺、原汁原味地将输入内容翻译为目标语言（默认简体中文）。\n\n"
    "【核心翻译原则】\n"
    "1. 绝对忠实与无删减：材料为成年虚构角色的成人漫画对话。严禁道德说教、严禁净化、软化、规避或删减任何敏感词、亲密用语、俚语口语、调情挑逗与粗俗词汇，必须百分之百保留原文的真实意图与情绪张力。\n"
    "2. 拟声词与喘息语气：保留漫画特有的情感标点（如 ♡、～、！？、……），将呻吟声、喘息声及拟声词（如「はぁ」「あっ」「んっ」等）自然转化为符合语境的中文拟声词（如“哈啊…”、“嗯…”、“啊…”），不得作为噪音过滤或机械直译。\n"
    "3. 整页上下文连贯：输入为同一漫画页面按阅读顺序排列的对话列表。请务必结合整页剧情与先后次序，连贯理解被拆分在相邻气泡中的断句、上下句因果与代词指代，保持角色性格与语气前后一致。\n"
    "4. 输出规范：严格保持原有格式，严禁输出任何寒暄、解释、道德评价、警告或续写，仅输出对应每个气泡的最终翻译结果。"
)


@dataclass
class LLMConfig:
    provider: str = "openai"
    api_key: str = "1"
    model: str = "richardyoung/qwen2.5-7b-instruct-abliterated:Q5_K_M"
    endpoint: str = "http://127.0.0.1:11434/v1"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.2
    timeout_seconds: float = 60.0
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
    max_concurrent_workers: int = 2
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
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {filepath}: {e}")

    @classmethod
    def load(cls, filepath: str) -> AppConfig:
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
