"""
tests/unit/test_translation.py
Unit tests for multi-provider translation, 7-stage JSON parser, prompts, and retry logic.
"""
import json
import pytest
import requests
from unittest.mock import MagicMock

from app.core.models import TranslationBlock
from app.core.translation.base import (
    ProviderConfig, TranslationContext,
    RateLimitError, QuotaExhaustedError, AuthenticationError, ModelNotFoundError
)
from app.core.translation.prompt_templates import PromptTemplates
from app.core.translation.json_parser import (
    parse_llm_json_response,
    extract_translation_map,
    _repair_unclosed_json
)
from app.core.translation.manager import TranslationManager
from app.core.translation.retry_handler import execute_http_request_with_retry


def test_json_parser_markdown_fences():
    raw = """```json
[
  {"id": "b1", "translated_text": "你好", "type": "bubble"}
]
```"""
    res = parse_llm_json_response(raw)
    assert isinstance(res, list)
    assert res[0]["id"] == "b1"
    assert res[0]["translated_text"] == "你好"


def test_json_parser_deepseek_reasoning_tags():
    raw = """<think>
The speaker is using polite keigo, so we translate with respect.
</think>
[
  {"id": "b2", "translated_text": "谢谢你", "type": "bubble"}
]"""
    res = parse_llm_json_response(raw)
    assert isinstance(res, list)
    assert res[0]["id"] == "b2"
    assert res[0]["translated_text"] == "谢谢你"


def test_json_parser_trailing_commas():
    raw = """[
  {"id": "b3", "translated_text": "再见", "type": "bubble",},
]"""
    res = parse_llm_json_response(raw)
    assert isinstance(res, list)
    assert res[0]["id"] == "b3"


def test_json_parser_truncated_unclosed_json():
    raw = """[
  {"id": "b4", "translated_text": "等等我！", "type": "bubble"},
  {"id": "b5", "translated_text": "不
"""
    repaired = _repair_unclosed_json(raw)
    assert repaired.endswith("]") or repaired.endswith("}")
    # Full parser should salvage or repair
    res = parse_llm_json_response(raw)
    assert len(res) >= 1
    assert res[0]["id"] == "b4"


def test_json_parser_regex_salvage():
    corrupted = """Here is the result:
    id: b6, translated_text: 哇啊啊！, type: onomatopoeia
    some extra rambling
    id: b7, translated_text: 真的吗？, type: bubble
    """
    salvaged = parse_llm_json_response(corrupted)
    assert len(salvaged) == 2
    assert salvaged[0]["id"] == "b6"
    assert salvaged[0]["type"] == "onomatopoeia"
    assert salvaged[1]["id"] == "b7"


def test_extract_translation_map():
    data = [
        {"id": "b1", "translated_text": "Hello", "type": "bubble"},
        {"id": "b2", "translated_text": "Boom!", "type": "onomatopoeia"}
    ]
    t_map = extract_translation_map(data)
    assert "b1" in t_map
    assert t_map["b1"]["translated_text"] == "Hello"
    assert t_map["b2"]["type"] == "onomatopoeia"


def test_prompt_templates_manga_continuity_and_glossary():
    ctx = TranslationContext(
        glossary={"ナルト": "鸣人", "サスケ": "佐助"},
        previous_summary="鸣人与佐助在终末之谷展开激战。"
    )
    sys_prompt = PromptTemplates.build_text_system_prompt("日语", "简体中文", ctx)

    assert "Right-to-Left" in sys_prompt
    assert "-san" in sys_prompt
    assert "CHAPTER GLOSSARY" in sys_prompt
    assert "ナルト => 鸣人" in sys_prompt
    assert "终末之谷" in sys_prompt

    blocks = [TranslationBlock(id="blk_01", original_text="ナルト！")]
    user_msg = PromptTemplates.build_text_user_message(blocks, "日语", "简体中文")
    assert "blk_01" in user_msg
    assert "ナルト！" in user_msg


def test_prompt_templates_auto_detect_and_adult_manga():
    # Test auto language detection prompt framing
    sys_prompt_auto = PromptTemplates.build_text_system_prompt("自动识别", "简体中文")
    assert "automatically detecting the source language" in sys_prompt_auto
    assert "Language Auto-Detection" in sys_prompt_auto
    assert "NO CENSORSHIP" in sys_prompt_auto
    assert "Sensual Onomatopoeia" in sys_prompt_auto

    user_msg_auto = PromptTemplates.build_text_user_message(
        [TranslationBlock(id="b1", original_text="あん…っ…はぁ")],
        "自动识别",
        "简体中文"
    )
    assert "Auto Detect" in user_msg_auto
    assert "b1" in user_msg_auto
    assert "あん…っ…はぁ" in user_msg_auto

    # Test explicit language specification
    sys_prompt_ko = PromptTemplates.build_text_system_prompt("韩语", "简体中文")
    assert "translating from 韩语" in sys_prompt_ko
    assert "Translate faithfully from 韩语 into 简体中文" in sys_prompt_ko



def test_translation_manager_local_demo_mode():
    mgr = TranslationManager.get_instance()
    # Empty API key triggers local preview fallback
    mgr.set_active_provider("deepseek", ProviderConfig(provider_name="deepseek", api_key=""))

    blocks = [
        TranslationBlock(id="1", original_text="おはよう"),
        TranslationBlock(id="2", original_text="さようなら")
    ]
    translated = mgr.translate(blocks, mode="text")

    assert len(translated) == 2
    assert translated[0].translated_text == "【译】おはよう"
    assert translated[1].translated_text == "【译】さようなら"


def test_translation_manager_glossary():
    mgr = TranslationManager.get_instance()
    mgr.clear_context()
    mgr.set_glossary({"Senpai": "学长"})
    mgr.add_glossary_term("Sensei", "老师")

    assert mgr._chapter_context.glossary["Senpai"] == "学长"
    assert mgr._chapter_context.glossary["Sensei"] == "老师"


def test_retry_handler_fatal_errors(monkeypatch):
    # Mock requests.request to return 401 Unauthorized
    mock_resp_401 = MagicMock()
    mock_resp_401.status_code = 401
    mock_resp_401.text = "Invalid API Key"

    monkeypatch.setattr(requests, "request", lambda *a, **kw: mock_resp_401)

    with pytest.raises(AuthenticationError):
        execute_http_request_with_retry("POST", "https://mock.api/v1", headers={})

    # Mock 429 with Insufficient Quota
    mock_resp_quota = MagicMock()
    mock_resp_quota.status_code = 429
    mock_resp_quota.text = "Error: insufficient_quota for this account"

    monkeypatch.setattr(requests, "request", lambda *a, **kw: mock_resp_quota)

    with pytest.raises(QuotaExhaustedError):
        execute_http_request_with_retry("POST", "https://mock.api/v1", headers={})

    # Mock 404 Model Not Found
    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404
    mock_resp_404.text = "The model 'unknown-model' does not exist"

    monkeypatch.setattr(requests, "request", lambda *a, **kw: mock_resp_404)

    with pytest.raises(ModelNotFoundError):
        execute_http_request_with_retry("POST", "https://mock.api/v1", headers={})
