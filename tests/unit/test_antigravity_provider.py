from types import SimpleNamespace
import pytest

from app.core.models import TranslationBlock
from app.core.translation.antigravity_provider import AntigravityGeminiProvider
from app.core.translation.base import (
    AuthenticationError,
    ProviderConfig,
    TranslationError,
)
from desktop.core.translation_engine import TranslationEngine


def _make_provider(**overrides) -> AntigravityGeminiProvider:
    cfg_kwargs = {
        "provider_name": "antigravity",
        "api_key": "sk-test-key",
        "model": "gemini-3.8-flash",
        "endpoint": "https://sub2api.emberanvil.dev/",
        "timeout_seconds": 60.0,
    }
    cfg_kwargs.update(overrides)
    return AntigravityGeminiProvider(ProviderConfig(**cfg_kwargs))


def test_build_headers_sets_curl_ua_and_auth():
    prov = _make_provider()
    headers = prov._build_headers()
    assert headers["User-Agent"] == "curl/8.4.0"
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["x-goog-api-key"] == "sk-test-key"
    assert headers["Content-Type"] == "application/json"


def test_build_candidate_routes_omits_query_api_key():
    prov = _make_provider()
    routes = prov._build_candidate_routes()
    assert routes
    assert routes[0][1] == (
        "https://sub2api.emberanvil.dev/antigravity/v1beta/models/"
        "gemini-3.8-flash:generateContent"
    )
    assert all("?key=" not in url for _, url in routes)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 120.0),
        (0.0, 120.0),
        (25.0, 120.0),
        (60.0, 120.0),
        (120.0, 120.0),
        (180.0, 180.0),
    ],
)
def test_effective_timeout_upgrades_legacy_values(configured, expected):
    cfg = ProviderConfig(provider_name="antigravity", api_key="sk-test", model="gemini-3.8-flash")
    cfg.timeout_seconds = configured
    prov = AntigravityGeminiProvider(cfg)
    assert prov._effective_timeout(120.0) == expected


def test_test_connection_uses_extended_timeout_and_output_tokens(monkeypatch):
    monkeypatch.setattr(AntigravityGeminiProvider, "_working_route_cache", {})
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return SimpleNamespace(
            status_code=200,
            text='{"candidates": []}',
            json=lambda: {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "internal thought", "thought": True},
                                {"text": "连通成功"},
                            ]
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("app.core.translation.antigravity_provider.requests.post", fake_post)
    prov = _make_provider(timeout_seconds=15.0)
    res = prov.test_connection("ping")

    assert res.success is True
    assert res.message == "连通成功"
    assert captured["timeout"] >= 120.0
    assert captured["headers"]["User-Agent"] == "curl/8.4.0"
    assert captured["json"]["generationConfig"]["maxOutputTokens"] >= 512


def test_translation_engine_antigravity_test_connection_passes_min_timeout(monkeypatch):
    monkeypatch.setattr(AntigravityGeminiProvider, "_working_route_cache", {})
    seen = {}

    def fake_retry(**kwargs):
        seen["timeout"] = kwargs["timeout"]
        seen["headers"] = kwargs["headers"]
        return SimpleNamespace(
            json=lambda: {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '[{"id": "test_1", "translated_text": "你好，这是连通测试。"}]'
                                }
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "app.core.translation.antigravity_provider.execute_http_request_with_retry",
        fake_retry,
    )
    eng = TranslationEngine(
        provider="antigravity",
        api_key="sk-test-key",
        model="gemini-3.8-flash",
        custom_endpoint="https://sub2api.emberanvil.dev/",
    )
    out = eng.test_connection()
    assert out == "你好，这是连通测试。"
    assert seen["timeout"] >= 120.0
    assert seen["headers"]["User-Agent"] == "curl/8.4.0"


def test_resilient_blocks_does_not_swallow_auth_or_timeout_errors(monkeypatch):
    prov = _make_provider()
    blocks = [
        TranslationBlock(id="b1", original_text="Hello"),
        TranslationBlock(id="b2", original_text="World"),
    ]

    for err in (
        AuthenticationError("bad key", provider="Antigravity Gemini"),
        TranslationError("timeout", provider="Antigravity Gemini", status_code=None),
    ):
        calls = []

        def fake_exec(**kwargs):
            calls.append(kwargs)
            raise err

        monkeypatch.setattr(prov, "_execute_with_fallback", fake_exec)
        with pytest.raises(type(err)):
            prov._translate_blocks_resilient(blocks, "English", "简体中文")
        assert len(calls) == 1


def test_resilient_blocks_falls_back_to_line_mode_on_http_400_or_refusal(monkeypatch):
    prov = _make_provider()
    blocks = [
        TranslationBlock(id="b1", original_text="Hello"),
        TranslationBlock(id="b2", original_text="World"),
    ]
    calls = []

    def fake_exec_http400(**kwargs):
        calls.append(kwargs.get("force_plain", False))
        if len(calls) == 1:
            raise TranslationError("safety block", provider="Antigravity Gemini", status_code=400)
        return "[b1] 你好\n[b2] 世界"

    monkeypatch.setattr(prov, "_execute_with_fallback", fake_exec_http400)
    res = prov._translate_blocks_resilient(blocks, "English", "简体中文")
    assert calls == [False, True]
    assert [b.translated_text for b in res] == ["你好", "世界"]

    calls_refusal = []

    def fake_exec_refusal(**kwargs):
        calls_refusal.append(kwargs.get("force_plain", False))
        if len(calls_refusal) == 1:
            return "Sorry, I cannot assist with that."
        return "[b1] 你好\n[b2] 世界"

    monkeypatch.setattr(prov, "_execute_with_fallback", fake_exec_refusal)
    res2 = prov._translate_blocks_resilient(blocks, "English", "简体中文")
    assert calls_refusal == [False, True]
    assert [b.translated_text for b in res2] == ["你好", "世界"]
