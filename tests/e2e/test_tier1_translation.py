import os
import json
import pytest
import requests
from desktop.core.translation_engine import TranslationEngine

# ============================================================================
# F-TRN-01: Multi-Provider LLM Manager
# ============================================================================

def test_ftrn_01_deepseek_endpoint_and_headers():
    engine = TranslationEngine(provider="deepseek", api_key="sk-test-deepseek")
    url, headers = engine._get_api_url_and_headers()
    assert "api.deepseek.com" in url
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer sk-test-deepseek"

def test_ftrn_01_openai_endpoint_and_headers():
    engine = TranslationEngine(provider="openai", api_key="sk-test-openai", custom_endpoint="")
    url, headers = engine._get_api_url_and_headers()
    assert "api.openai.com" in url
    assert headers["Authorization"] == "Bearer sk-test-openai"

def test_ftrn_01_gemini_endpoint_and_headers():
    engine = TranslationEngine(provider="gemini", api_key="sk-test-gemini", custom_endpoint="")
    url, headers = engine._get_api_url_and_headers()
    assert "generativelanguage.googleapis.com" in url
    assert headers["Authorization"] == "Bearer sk-test-gemini"

def test_ftrn_01_custom_endpoint_normalization():
    # Test without trailing slash and without /chat/completions
    engine = TranslationEngine(provider="custom", api_key="sk-custom", custom_endpoint="https://my-llm-host.com/v1")
    url, headers = engine._get_api_url_and_headers()
    assert url == "https://my-llm-host.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-custom"

    # Test already ending with /chat/completions
    engine2 = TranslationEngine(provider="custom", api_key="sk-custom", custom_endpoint="https://my-llm-host.com/v1/chat/completions")
    url2, _ = engine2._get_api_url_and_headers()
    assert url2 == "https://my-llm-host.com/v1/chat/completions"

def test_ftrn_01_empty_api_key_demo_mode():
    engine = TranslationEngine(provider="deepseek", api_key="")
    blocks = [{"id": "b1", "original_text": "こんにちは", "translated_text": ""}]
    res = engine.translate_blocks(blocks)
    assert len(res) == 1
    assert "【译】こんにちは" in res[0]["translated_text"]

# ============================================================================
# F-TRN-02: Structured Text JSON Translation
# ============================================================================

def test_ftrn_02_parse_clean_json_array():
    engine = TranslationEngine()
    raw = '[{"id": "b1", "translated_text": "你好"}]'
    parsed = engine._parse_json_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "b1"
    assert parsed[0]["translated_text"] == "你好"

def test_ftrn_02_parse_markdown_fence_wrapped_json():
    engine = TranslationEngine()
    raw = '```json\n[\n  {"id": "b2", "translated_text": "再见"}\n]\n```'
    parsed = engine._parse_json_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "b2"
    assert parsed[0]["translated_text"] == "再见"

def test_ftrn_02_parse_dict_with_nested_key():
    engine = TranslationEngine()
    raw = '{"dialogues": [{"id": "b3", "translated_text": "快跑！"}]}'
    parsed = engine._parse_json_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "b3"
    assert parsed[0]["translated_text"] == "快跑！"

def test_ftrn_02_regex_fallback_on_corrupt_json():
    engine = TranslationEngine()
    raw = 'Here is your translation:\n"id": "b4", "translated_text": "小心！"\nsome other text'
    parsed = engine._parse_json_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "b4"
    assert parsed[0]["translated_text"] == "小心！"

def test_ftrn_02_block_id_integrity_mapping(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    # Mock post to return json
    class MockResp:
        status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps([
                            {"id": "box_001", "translated_text": "第一句话"},
                            {"id": "box_002", "translated_text": "第二句话"}
                        ])
                    }
                }]
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResp())
    blocks = [
        {"id": "box_001", "original_text": "original 1", "translated_text": ""},
        {"id": "box_002", "original_text": "original 2", "translated_text": ""}
    ]
    res = engine.translate_blocks(blocks)
    assert res[0]["translated_text"] == "第一句话"
    assert res[1]["translated_text"] == "第二句话"

# ============================================================================
# F-TRN-03: Multimodal Vision Translation
# ============================================================================

def test_ftrn_03_vision_model_naming():
    engine = TranslationEngine(provider="openai", model="gpt-4o", api_key="sk-test")
    assert "gpt-4" in engine.model

def test_ftrn_03_gemini_vision_model_naming():
    engine = TranslationEngine(provider="gemini", model="gemini-2.0-flash", api_key="sk-test")
    assert "gemini" in engine.model

def test_ftrn_03_empty_blocks_safe():
    engine = TranslationEngine(api_key="sk-test")
    assert engine.translate_blocks([]) == []

def test_ftrn_03_progress_reporting_during_translation(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    class MockResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"id": "1", "translated_text": "译文"}]'}}]}
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResp())

    calls = []
    blocks = [{"id": "1", "original_text": "原文", "translated_text": ""}]
    engine.translate_blocks(blocks, progress_callback=lambda pct, msg: calls.append(pct))
    assert len(calls) >= 2
    assert calls[-1] == 100

def test_ftrn_03_missing_id_in_response_preserves_original(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    class MockResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"id": "other_id", "translated_text": "其他"}]'}}]}
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResp())

    blocks = [{"id": "my_id", "original_text": "原本文本", "translated_text": ""}]
    res = engine.translate_blocks(blocks)
    assert res[0]["translated_text"] == "原本文本"

# ============================================================================
# F-TRN-04: Manga-Tuned Prompt Engineering
# ============================================================================

def test_ftrn_04_default_system_prompt():
    engine = TranslationEngine()
    assert "manga" in engine.system_prompt.lower()
    assert "dialogue" in engine.system_prompt.lower()

def test_ftrn_04_custom_system_prompt():
    custom = "Translate comic bubbles with funny humor."
    engine = TranslationEngine(system_prompt=custom)
    assert engine.system_prompt == custom

def test_ftrn_04_target_language_configuration():
    engine = TranslationEngine(target_lang="English", source_lang="Japanese")
    assert engine.target_lang == "English"
    assert engine.source_lang == "Japanese"

def test_ftrn_04_temperature_setting():
    engine = TranslationEngine(temperature=0.7)
    assert engine.temperature == 0.7

def test_ftrn_04_user_message_includes_target_lang(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test", target_lang="繁體中文")
    captured_body = {}
    class MockResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"id": "1", "translated_text": "測試"}]'}}]}
    def mock_post(url, headers, json, timeout):
        captured_body.update(json)
        return MockResp()

    monkeypatch.setattr(requests, "post", mock_post)
    engine.translate_blocks([{"id": "1", "original_text": "テスト"}])
    user_content = captured_body["messages"][1]["content"]
    assert "Target Language: 繁體中文" in user_content

# ============================================================================
# F-TRN-05: Context & Glossary Preservation
# ============================================================================

def test_ftrn_05_multi_turn_dialogue_payload(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    captured_body = {}
    class MockResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"id": "b1", "translated_text": "你好"}, {"id": "b2", "translated_text": "世界"}]'}}]}
    monkeypatch.setattr(requests, "post", lambda url, headers, json, timeout: (captured_body.update(json), MockResp())[1])

    blocks = [
        {"id": "b1", "original_text": "Hello"},
        {"id": "b2", "original_text": "World"}
    ]
    engine.translate_blocks(blocks)
    user_content = captured_body["messages"][1]["content"]
    assert "b1" in user_content
    assert "b2" in user_content

def test_ftrn_05_ordered_sequence_maintained(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    class MockResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"id": "1", "translated_text": "A"}, {"id": "2", "translated_text": "B"}]'}}]}
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResp())

    blocks = [{"id": "1", "original_text": "1"}, {"id": "2", "original_text": "2"}]
    res = engine.translate_blocks(blocks)
    assert res[0]["id"] == "1"
    assert res[1]["id"] == "2"

def test_ftrn_05_partial_translations_preserved():
    engine = TranslationEngine(api_key="")  # demo mode
    blocks = [
        {"id": "1", "original_text": "未翻译", "translated_text": ""},
        {"id": "2", "original_text": "已翻译", "translated_text": "已有译文"}
    ]
    res = engine.translate_blocks(blocks)
    assert res[1]["translated_text"] == "已有译文"

def test_ftrn_05_error_attaches_error_tag(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    def mock_fail(*args, **kwargs):
        raise requests.ConnectionError("Network disconnected")
    monkeypatch.setattr(requests, "post", mock_fail)

    blocks = [{"id": "1", "original_text": "Text", "translated_text": ""}]
    with pytest.raises(requests.ConnectionError):
        engine.translate_blocks(blocks)
    assert "[翻译错误" in blocks[0]["translated_text"]

def test_ftrn_05_json_response_format_in_body(monkeypatch):
    engine = TranslationEngine(provider="deepseek", model="deepseek-chat", api_key="sk-test")
    captured = {}
    class MockResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"id": "1", "translated_text": "OK"}]'}}]}
    monkeypatch.setattr(requests, "post", lambda url, headers, json, timeout: (captured.update(json), MockResp())[1])

    engine.translate_blocks([{"id": "1", "original_text": "test"}])
    assert "response_format" in captured
    assert captured["response_format"] == {"type": "json_object"}

# ============================================================================
# F-TRN-06: API Connection Diagnostic Tester
# ============================================================================

def test_ftrn_06_http_401_unauthorized(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-invalid")
    class Mock401:
        status_code = 401
        text = '{"error": {"message": "Invalid API key"}}'
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock401())

    blocks = [{"id": "1", "original_text": "text"}]
    with pytest.raises(RuntimeError) as exc_info:
        engine.translate_blocks(blocks)
    assert "401" in str(exc_info.value)

def test_ftrn_06_http_429_rate_limit(monkeypatch):
    engine = TranslationEngine(provider="openai", api_key="sk-quota")
    class Mock429:
        status_code = 429
        text = '{"error": {"message": "Rate limit reached or insufficient quota"}}'
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock429())

    blocks = [{"id": "1", "original_text": "text"}]
    with pytest.raises(RuntimeError) as exc_info:
        engine.translate_blocks(blocks)
    assert "429" in str(exc_info.value)

def test_ftrn_06_http_500_server_error(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    class Mock500:
        status_code = 500
        text = "Internal Server Error"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock500())

    blocks = [{"id": "1", "original_text": "text"}]
    with pytest.raises(RuntimeError) as exc_info:
        engine.translate_blocks(blocks)
    assert "500" in str(exc_info.value)

def test_ftrn_06_request_timeout(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    def mock_timeout(*args, **kwargs):
        raise requests.Timeout("Connection timed out after 60s")
    monkeypatch.setattr(requests, "post", mock_timeout)

    blocks = [{"id": "1", "original_text": "text"}]
    with pytest.raises(requests.Timeout):
        engine.translate_blocks(blocks)

def test_ftrn_06_successful_translation_roundtrip(monkeypatch):
    engine = TranslationEngine(provider="deepseek", api_key="sk-test")
    class Mock200:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '[{"id": "b1", "translated_text": "成功"}]'}}]}
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Mock200())

    blocks = [{"id": "b1", "original_text": "success", "translated_text": ""}]
    res = engine.translate_blocks(blocks)
    assert res[0]["translated_text"] == "成功"
