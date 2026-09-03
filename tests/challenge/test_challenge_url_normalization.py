"""
tests/challenge/test_challenge_url_normalization.py
Empirical Adversarial Challenge Suite 3: Provider URL Normalization & Endpoint Robustness.
Demonstrates robust normalization and confirms critical endpoint construction bugs.
"""
import pytest
from unittest.mock import MagicMock
import requests

from app.core.translation.base import ProviderConfig
from app.core.translation.custom_provider import CustomOpenAIProvider
from app.core.translation.openai_provider import OpenAIProvider
from app.core.translation.deepseek_provider import DeepSeekProvider
from app.core.translation.gemini_provider import GeminiProvider
from app.core.models import TranslationBlock


# =========================================================================
# SECTION 1: ROBUST BEHAVIORS (CONFIRMED RESILIENT)
# =========================================================================

def test_pass_custom_provider_default_endpoint():
    """Confirms empty endpoint defaults to local Ollama /v1/chat/completions."""
    cfg = ProviderConfig(provider_name="custom", endpoint="")
    p = CustomOpenAIProvider(cfg)
    assert p.endpoint == "http://localhost:11434/v1/chat/completions"


def test_pass_custom_provider_trailing_slash():
    """Confirms trailing slash on base path is stripped before appending /chat/completions."""
    cfg = ProviderConfig(provider_name="custom", endpoint="https://api.siliconflow.cn/v1/")
    p = CustomOpenAIProvider(cfg)
    assert p.endpoint == "https://api.siliconflow.cn/v1/chat/completions"


def test_pass_custom_provider_full_endpoint_not_duplicated():
    """Confirms CustomOpenAIProvider does not duplicate /chat/completions if already present."""
    cfg = ProviderConfig(provider_name="custom", endpoint="https://api.siliconflow.cn/v1/chat/completions")
    p = CustomOpenAIProvider(cfg)
    assert p.endpoint == "https://api.siliconflow.cn/v1/chat/completions"


def test_pass_openai_default_and_trailing_slash():
    """Confirms OpenAI default endpoint and trailing slash removal on base url."""
    p_def = OpenAIProvider(ProviderConfig(provider_name="openai"))
    assert p_def.endpoint == "https://api.openai.com/v1"

    p_slash = OpenAIProvider(ProviderConfig(provider_name="openai", endpoint="https://my-proxy.com/v1/"))
    assert p_slash.endpoint == "https://my-proxy.com/v1"


def test_pass_gemini_default_endpoint():
    """Confirms Gemini default endpoint."""
    p = GeminiProvider(ProviderConfig(provider_name="gemini"))
    assert p.endpoint == "https://generativelanguage.googleapis.com"


# =========================================================================
# SECTION 2: CONFIRMED VULNERABILITIES & FAILURE MODES
# =========================================================================

def test_fixed_openai_provider_no_duplicated_chat_completions(monkeypatch):
    """
    PREVIOUSLY BUG 5 (now FIXED): OpenAIProvider now strips trailing '/chat/completions'
    from user-supplied endpoints before appending its own, preventing URL duplication.
    """
    captured_urls = []
    def mock_req(method, url, **kwargs):
        captured_urls.append(url)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '[{"id": "b1", "translated_text": "ok", "type": "bubble"}]'}}]
        }
        return mock_resp

    monkeypatch.setattr(requests, "request", mock_req)

    cfg = ProviderConfig(
        provider_name="openai",
        endpoint="https://my-proxy.com/v1/chat/completions",
        api_key="sk-test"
    )
    p = OpenAIProvider(cfg)
    blocks = [TranslationBlock(id="b1", original_text="test")]
    p.translate_text_blocks(blocks, "日语", "简体中文")

    assert len(captured_urls) == 1
    # FIXED by M2 hardening: endpoint normalizer strips trailing /chat/completions
    assert captured_urls[0] == "https://my-proxy.com/v1/chat/completions"


def test_fixed_deepseek_provider_no_duplicated_chat_completions(monkeypatch):
    """
    PREVIOUSLY BUG 6 (now FIXED): DeepSeekProvider now strips trailing '/chat/completions'
    from user-supplied endpoints before appending its own.
    """
    captured_urls = []
    def mock_req(method, url, **kwargs):
        captured_urls.append(url)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '[{"id": "b1", "translated_text": "ok", "type": "bubble"}]'}}]
        }
        return mock_resp

    monkeypatch.setattr(requests, "request", mock_req)

    cfg = ProviderConfig(
        provider_name="deepseek",
        endpoint="https://api.deepseek.com/v1/chat/completions",
        api_key="sk-test"
    )
    p = DeepSeekProvider(cfg)
    blocks = [TranslationBlock(id="b1", original_text="test")]
    p.translate_text_blocks(blocks, "日语", "简体中文")

    assert len(captured_urls) == 1
    # FIXED: no duplicate /chat/completions
    assert captured_urls[0] == "https://api.deepseek.com/v1/chat/completions"


def test_fixed_custom_provider_adds_v1_prefix():
    """
    PREVIOUSLY BUG 7 (now FIXED): CustomOpenAIProvider._normalize_endpoint now
    ensures /v1 prefix is attached when not present.
    """
    ep = CustomOpenAIProvider._normalize_endpoint("http://localhost:11434")
    # FIXED: /v1/ prefix is now properly added
    assert "/v1/" in ep
    assert ep == "http://localhost:11434/v1/chat/completions"


def test_fixed_gemini_no_duplicated_v1beta(monkeypatch):
    """
    PREVIOUSLY BUG 8 (now FIXED): GeminiProvider now strips trailing '/v1beta'
    from user-supplied endpoints before appending its own path.
    """
    captured_urls = []
    def mock_req(method, url, **kwargs):
        captured_urls.append(url)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '[{"id": "b1", "translated_text": "ok", "type": "bubble"}]'}]}}]
        }
        return mock_resp

    monkeypatch.setattr(requests, "request", mock_req)

    cfg = ProviderConfig(
        provider_name="gemini",
        endpoint="https://my-gemini-proxy.com/v1beta",
        api_key="test-key"
    )
    p = GeminiProvider(cfg)
    blocks = [TranslationBlock(id="b1", original_text="test")]
    p.translate_text_blocks(blocks, "日语", "简体中文")

    assert len(captured_urls) == 1
    # FIXED: no duplicate /v1beta
    assert not captured_urls[0].startswith("https://my-gemini-proxy.com/v1beta/v1beta/")
    assert "/v1beta/models/" in captured_urls[0]

