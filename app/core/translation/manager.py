"""
app/core/translation/manager.py
Translation Manager: Provider Registry, Switcher, Credential Store, and Diagnostic Ping.
"""
from typing import Dict, Type, Optional, List, Any, Callable, Union
import numpy as np

from app.core.models import TranslationBlock
from app.core.translation.base import (
    BaseTranslationProvider, ProviderConfig,
    TranslationContext, DiagnosticResult, TranslationError
)
from app.core.translation.deepseek_provider import DeepSeekProvider
from app.core.translation.openai_provider import OpenAIProvider
from app.core.translation.gemini_provider import GeminiProvider
from app.core.translation.custom_provider import CustomOpenAIProvider


class TranslationManager:
    _instance: Optional['TranslationManager'] = None
    _providers: Dict[str, Type[BaseTranslationProvider]] = {}

    def __init__(self):
        self._provider_instances: Dict[str, BaseTranslationProvider] = {}
        self._credentials: Dict[str, ProviderConfig] = {}
        self._active_provider_name: str = "deepseek"
        self._chapter_context: TranslationContext = TranslationContext()
        self._ensure_default_providers()

    def _ensure_default_providers(self):
        if not self._providers:
            self._providers["deepseek"] = DeepSeekProvider
            self._providers["openai"] = OpenAIProvider
            self._providers["gemini"] = GeminiProvider
            self._providers["custom"] = CustomOpenAIProvider

    @classmethod
    def get_instance(cls) -> 'TranslationManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseTranslationProvider]):
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def get_registered_providers(cls) -> List[str]:
        if not cls._providers:
            cls._providers["deepseek"] = DeepSeekProvider
            cls._providers["openai"] = OpenAIProvider
            cls._providers["gemini"] = GeminiProvider
            cls._providers["custom"] = CustomOpenAIProvider
        return list(cls._providers.keys())

    def set_active_provider(self, name: str, config: Optional[ProviderConfig] = None):
        self._ensure_default_providers()
        name_key = name.lower()
        if name_key not in self._providers:
            raise ValueError(f"Provider '{name}' is not registered. Available: {self.get_registered_providers()}")

        self._active_provider_name = name_key
        if config is not None:
            self._credentials[name_key] = config
            self._provider_instances.pop(name_key, None)

    def get_active_provider(self) -> BaseTranslationProvider:
        self._ensure_default_providers()
        name = self._active_provider_name
        if name not in self._provider_instances:
            provider_cls = self._providers[name]
            config = self._credentials.get(name, ProviderConfig(provider_name=name))
            self._provider_instances[name] = provider_cls(config)
        return self._provider_instances[name]

    def update_credentials(self, provider_name: str, **kwargs):
        name_key = provider_name.lower()
        current_cfg = self._credentials.get(name_key, ProviderConfig(provider_name=name_key))
        for k, v in kwargs.items():
            if hasattr(current_cfg, k):
                setattr(current_cfg, k, v)
        self._credentials[name_key] = current_cfg
        self._provider_instances.pop(name_key, None)

    def get_credentials(self, provider_name: str) -> Optional[ProviderConfig]:
        return self._credentials.get(provider_name.lower())

    def test_connection(
        self,
        provider_name: Optional[str] = None,
        config: Optional[ProviderConfig] = None
    ) -> DiagnosticResult:
        """Executes diagnostic ping to test provider connectivity, model status, and latency."""
        self._ensure_default_providers()
        name = (provider_name or self._active_provider_name).lower()
        if name not in self._providers:
            return DiagnosticResult(
                success=False, provider=name, model="", latency_ms=0,
                message=f"未注册的服务提供商: {name}", suggested_action="请选择支持的服务提供商"
            )

        provider_cls = self._providers[name]
        cfg = config or self._credentials.get(name, ProviderConfig(provider_name=name))
        provider_inst = provider_cls(cfg)
        return provider_inst.test_connection()

    def translate(
        self,
        blocks: List[Union[TranslationBlock, Dict[str, Any]]],
        image: Optional[np.ndarray] = None,
        mode: str = "text",
        source_lang: str = "自动识别",
        target_lang: str = "简体中文",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[Any]:
        """Unified translation entry point with format conversion and fallback handling."""
        provider = self.get_active_provider()

        # Handle empty blocks
        if not blocks and mode != "vision":
            return []

        # Convert input items to TranslationBlock instances for uniform internal handling
        is_dict_input = isinstance(blocks[0], dict) if blocks else False
        tb_blocks: List[TranslationBlock] = []
        for b in blocks:
            if isinstance(b, TranslationBlock):
                tb_blocks.append(b)
            elif isinstance(b, dict):
                tb_blocks.append(TranslationBlock.from_dict(b))

        # Determine reading order mode based on context or source_lang
        reading_mode = "manga_rtl"
        if self._chapter_context and self._chapter_context.reading_direction:
            rd = self._chapter_context.reading_direction.lower()
            if rd in ("ltr", "western_ltr"):
                reading_mode = "western_ltr"
            elif rd in ("ttb", "webtoon_ttb", "vertical"):
                reading_mode = "webtoon_ttb"
            elif rd in ("rtl", "manga_rtl"):
                reading_mode = "manga_rtl"
        elif source_lang:
            from app.core.translation.prompt_templates import normalize_source_lang
            norm_lang = normalize_source_lang(source_lang)
            if norm_lang == "en":
                reading_mode = "western_ltr"
            elif norm_lang == "ko":
                reading_mode = "webtoon_ttb"

        # Ensure sequential reading order to guarantee dialogue continuity for LLM
        has_reading_order = any(b.reading_order > 0 for b in tb_blocks)
        if has_reading_order:
            x_mult = 1 if reading_mode == "western_ltr" else -1
            tb_blocks.sort(key=lambda b: (b.reading_order if b.reading_order > 0 else 9999, x_mult * b.xmin, b.ymin))
        elif len(tb_blocks) > 1 and mode != "vision":
            try:
                from app.core.ocr.reading_order import sort_reading_order
                tb_blocks = sort_reading_order(tb_blocks, mode=reading_mode)
            except Exception:
                pass

        # Separate blocks with actual text from empty/whitespace blocks
        active_blocks = [b for b in tb_blocks if b.original_text and b.original_text.strip()]
        for b in tb_blocks:
            if not (b.original_text and b.original_text.strip()):
                b.translated_text = ""

        if not active_blocks and mode != "vision":
            return [b.to_dict() for b in tb_blocks] if is_dict_input else tb_blocks

        # Local demonstration mode when API key is empty
        if not provider.config.api_key and provider.config.provider_name != "custom":
            if progress_callback:
                progress_callback(50, "未检测到 API Key，正在运行本地演示翻译模式...")
            for b in active_blocks:
                if not b.translated_text:
                    b.translated_text = f"【译】{b.original_text}"
            return [b.to_dict() for b in tb_blocks] if is_dict_input else tb_blocks

        # Vision mode fallback if requested and supported
        if mode == "vision" and provider.supports_vision() and image is not None:
            import cv2
            _, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            translated_tb = provider.translate_vision(
                buf.tobytes(), target_lang=target_lang, source_lang=source_lang,
                progress_callback=progress_callback
            )
            return [b.to_dict() for b in translated_tb] if is_dict_input else translated_tb

        # Standard text translation mode (with automatic chunking for large page sets)
        CHUNK_SIZE = 25
        if len(active_blocks) > CHUNK_SIZE:
            total_active = len(active_blocks)
            for i in range(0, total_active, CHUNK_SIZE):
                chunk = active_blocks[i:i + CHUNK_SIZE]
                chunk_start_pct = int(20 + 75 * (i / total_active))
                chunk_end_pct = int(20 + 75 * (min(i + CHUNK_SIZE, total_active) / total_active))

                def make_chunk_cb(c_start: int, c_end: int, c_from: int, c_to: int):
                    def cb(pct: int, msg: str):
                        if progress_callback:
                            scaled = int(c_start + (c_end - c_start) * (pct / 100.0))
                            progress_callback(
                                min(98, max(1, scaled)),
                                f"[{c_from}-{c_to}/{total_active}] {msg}"
                            )
                    return cb

                chunk_cb = make_chunk_cb(chunk_start_pct, chunk_end_pct, i + 1, min(i + CHUNK_SIZE, total_active))

                # Build context for this chunk carrying narrative continuity from preceding chunks
                chunk_context = TranslationContext(
                    glossary=self._chapter_context.glossary,
                    character_notes=self._chapter_context.character_notes,
                    previous_summary=self._chapter_context.previous_summary,
                    reading_direction=self._chapter_context.reading_direction
                )
                if i > 0:
                    prev_dialogues = "\n".join(
                        f"- [{b.id}] 原文: {b.original_text} => 译文: {b.translated_text}"
                        for b in active_blocks[max(0, i - 3):i]
                        if b.translated_text
                    )
                    if prev_dialogues:
                        prefix = (chunk_context.previous_summary + "\n") if chunk_context.previous_summary else ""
                        chunk_context.previous_summary = f"{prefix}Previous dialogue context:\n{prev_dialogues}"

                provider.translate_text_blocks(
                    chunk,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    context=chunk_context,
                    progress_callback=chunk_cb
                )
            if progress_callback:
                progress_callback(100, f"所有 {len(active_blocks)} 个文本气泡翻译完成")
        else:
            provider.translate_text_blocks(
                active_blocks,
                source_lang=source_lang,
                target_lang=target_lang,
                context=self._chapter_context,
                progress_callback=progress_callback
            )

        return [b.to_dict() for b in tb_blocks] if is_dict_input else tb_blocks

    # Chapter Glossary & Terminology Continuity
    def set_glossary(self, glossary: Dict[str, str]):
        self._chapter_context.glossary = glossary

    def add_glossary_term(self, term: str, translation: str):
        self._chapter_context.glossary[term] = translation

    def clear_context(self):
        self._chapter_context = TranslationContext()
