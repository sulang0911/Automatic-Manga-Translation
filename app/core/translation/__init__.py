"""
app/core/translation - Multi-provider LLM translation manager and localization pipeline.
"""
from .base import (
    BaseTranslationProvider,
    ProviderConfig,
    TranslationContext,
    DiagnosticResult,
    TranslationError,
    RateLimitError,
    QuotaExhaustedError,
    AuthenticationError,
    ModelNotFoundError,
)
from .prompt_templates import PromptTemplates
from .json_parser import parse_llm_json_response, extract_translation_map
from .retry_handler import execute_http_request_with_retry
from .deepseek_provider import DeepSeekProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .custom_provider import CustomOpenAIProvider
from .manager import TranslationManager

__all__ = [
    "BaseTranslationProvider",
    "ProviderConfig",
    "TranslationContext",
    "DiagnosticResult",
    "TranslationError",
    "RateLimitError",
    "QuotaExhaustedError",
    "AuthenticationError",
    "ModelNotFoundError",
    "PromptTemplates",
    "parse_llm_json_response",
    "extract_translation_map",
    "execute_http_request_with_retry",
    "DeepSeekProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "CustomOpenAIProvider",
    "TranslationManager",
]
