"""
app/core/translation/base.py
Abstract base class, exceptions, and data contracts for LLM translation providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from app.core.models import TranslationBlock, BlockType


@dataclass
class TranslationContext:
    """Carries chapter-level terminology, character register, and narrative continuity."""
    glossary: Dict[str, str] = field(default_factory=dict)
    character_notes: Dict[str, str] = field(default_factory=dict)
    previous_summary: str = ""
    reading_direction: str = "rtl"  # "rtl" for manga, "ltr" for webtoons


@dataclass
class ProviderConfig:
    provider_name: str
    api_key: str = ""
    model: str = ""
    endpoint: str = ""
    temperature: float = 0.2
    timeout_seconds: float = 60.0
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    backoff_factor: float = 2.0
    custom_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)
    proxy_url: str = ""  # "" = use system env, "none"/"direct" = bypass proxy, or explicit URL e.g. "http://127.0.0.1:10808"


@dataclass
class DiagnosticResult:
    success: bool
    provider: str
    model: str
    latency_ms: float
    message: str
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    suggested_action: Optional[str] = None


# Custom Exception Hierarchy
class TranslationError(Exception):
    """Base exception for all translation-related failures."""
    def __init__(
        self,
        message: str,
        provider: str = "",
        status_code: Optional[int] = None,
        retryable: bool = False,
        suggested_action: str = ""
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.suggested_action = suggested_action


class RateLimitError(TranslationError):
    def __init__(self, message: str, provider: str = "", retry_after: Optional[float] = None):
        super().__init__(
            message,
            provider=provider,
            status_code=429,
            retryable=True,
            suggested_action="API 请求频率超限，正在自动退避重试，请稍候..."
        )
        self.retry_after = retry_after


class QuotaExhaustedError(TranslationError):
    def __init__(self, message: str, provider: str = ""):
        super().__init__(
            message,
            provider=provider,
            status_code=429,
            retryable=False,
            suggested_action="API 账户配额或余额已耗尽，请在服务商控制台充值或更换 API Key。"
        )


class AuthenticationError(TranslationError):
    def __init__(self, message: str, provider: str = ""):
        super().__init__(
            message,
            provider=provider,
            status_code=401,
            retryable=False,
            suggested_action="API Key 无效或已被注销，请在设置中重新检查密钥。"
        )


class ModelNotFoundError(TranslationError):
    def __init__(self, message: str, provider: str = "", model: str = ""):
        super().__init__(
            message,
            provider=provider,
            status_code=404,
            retryable=False,
            suggested_action=f"模型 '{model}' 不存在或无访问权限，请检查模型名称。"
        )


class BaseTranslationProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    def resolve_proxies(self) -> Optional[Dict[str, str]]:
        """
        Converts config.proxy_url to a proxies dict for requests.
          - proxy_url == ""            → None  (inherit system HTTP_PROXY/HTTPS_PROXY env vars)
          - proxy_url == "none"|"direct" → {}  (bypass all proxies, connect directly)
          - proxy_url == "http://..."  → {"http": url, "https": url}
        """
        url = self.config.proxy_url.strip()
        if not url:
            return None  # use system env
        if url.lower() in ("none", "direct", "no_proxy", "noproxy"):
            return {}    # bypass proxy entirely
        return {"http": url, "https": url}

    @abstractmethod
    def translate_text_blocks(
        self,
        blocks: List[TranslationBlock],
        source_lang: str,
        target_lang: str,
        context: Optional[TranslationContext] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[TranslationBlock]:
        """Translates an array of detected OCR text blocks."""
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """Returns True if this provider can perform multimodal vision translation."""
        pass

    def translate_vision(
        self,
        image_bytes: bytes,
        target_lang: str,
        source_lang: str = "auto",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[TranslationBlock]:
        """Multimodal image translation fallback (optional for vision models)."""
        raise NotImplementedError(f"Provider {self.config.provider_name} does not support vision translation.")

    @abstractmethod
    def test_connection(self) -> DiagnosticResult:
        """Sends a minimal diagnostic query to verify credentials, endpoint, and latency."""
        pass
