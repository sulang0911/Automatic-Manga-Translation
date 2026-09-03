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
    "你是一个中立且专业的通用翻译引擎。你的唯一任务是忠实、准确、通顺地将输入内容翻译为目标语言（未指定则默认翻译为简体中文）。"
    "保持原意、语气和排版不变。严禁过滤、修改或删减内容，严禁输出任何多余的寒暄、解释、评价或续写，仅输出最终翻译结果。"
)


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    api_key: str = ""
    model: str = "deepseek-chat"
    endpoint: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.2
    timeout_seconds: float = 60.0
    max_retries: int = 3


@dataclass
class OCRConfig:
    engine: str = "easyocr"  # "easyocr" or "paddle"
    lang: str = "japan"
    force_cpu: bool = False
    confidence_threshold: float = 0.25


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
    auto_save_cache: bool = True
    max_concurrent_workers: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme": self.theme,
            "llm": asdict(self.llm),
            "ocr": asdict(self.ocr),
            "inpaint": asdict(self.inpaint),
            "style": self.style.to_dict(),
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "cache_dir": self.cache_dir,
            "auto_save_cache": self.auto_save_cache,
            "max_concurrent_workers": self.max_concurrent_workers,
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
        if "auto_save_cache" in data:
            cfg.auto_save_cache = data["auto_save_cache"]
        if "max_concurrent_workers" in data:
            cfg.max_concurrent_workers = data["max_concurrent_workers"]

        if "llm" in data and isinstance(data["llm"], dict):
            cfg.llm = LLMConfig(**data["llm"])
        if "ocr" in data and isinstance(data["ocr"], dict):
            cfg.ocr = OCRConfig(**data["ocr"])
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
