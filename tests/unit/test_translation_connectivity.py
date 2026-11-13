"""
tests/unit/test_translation_connectivity.py
Unit tests verifying custom model system prompt configuration and translation connectivity test feature.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from desktop.core.config_manager import DEFAULT_CONFIG, DEFAULT_SYSTEM_PROMPT, ConfigManager
from desktop.core.translation_engine import TranslationEngine
from app.core.config import (
    AppConfig,
    DEFAULT_SYSTEM_PROMPT as APP_DEFAULT_SYSTEM_PROMPT,
    LEGACY_DEFAULT_SYSTEM_PROMPT,
)


EXPECTED_DEFAULT_PROMPT = (
    "你是一位专注于日韩及多语种漫画作品出版的资深本地化译者。你的核心职责是将给定的虚构角色对话与画面文字，自然、传神、贴合原意地本地化为目标语言（默认简体中文）。\n\n"
    "【核心本地化原则】\n"
    "1. 忠实原意与人物张力：请以文学翻译与台词润色的专业视角对待虚构剧情中的角色对白。面对口语俚语、激烈冲突、直率告白或亲昵互动等各类艺术表达，请依循语境如实传达人物的真实语气、情绪浓度与措辞分寸，遵循“信达雅”原则保留原汁原味的叙事风貌，不擅自改写、淡化或省略关键语义。\n"
    "2. 拟声词与情绪语气本地化：完整保留漫画特有的情感标点与停顿节奏（如 ♡、～、！？、……），将对白中的轻叹、气声感叹与象声词（如「はぁ」「あっ」「んっ」等）结合情境自然译为符合中文漫画阅读习惯的拟声表达（如“哈啊…”、“嗯…”、“啊…”），不作为无效字符过滤或生硬直译。\n"
    "3. 整页阅读顺序连贯：输入条目按单页漫画阅读顺序排列，请结合整页剧情脉络与上下文先后次序，顺畅衔接跨气泡拆分的长句断句、因果承接与人称代词指代，使人物口吻与性格前后一致。\n"
    "4. 输出规范：严格遵从指定输出格式，仅输出与各气泡对应的最终译文，不附加任何寒暄、解释、评论或多余续写。"
)


def test_default_system_prompt_matches_exact_requirement():
    """Verify that both desktop and app configuration use the diplomatic default system prompt."""
    assert DEFAULT_SYSTEM_PROMPT == EXPECTED_DEFAULT_PROMPT
    assert APP_DEFAULT_SYSTEM_PROMPT == EXPECTED_DEFAULT_PROMPT
    assert DEFAULT_CONFIG["system_prompt"] == EXPECTED_DEFAULT_PROMPT

    app_cfg = AppConfig()
    assert app_cfg.llm.system_prompt == EXPECTED_DEFAULT_PROMPT

    # Ensure high-refusal trigger terms are omitted while preserving core localization cues
    for forbidden in ("成人", "R18", "无删减", "严禁道德说教", "敏感词", "粗俗", "呻吟", "喘息"):
        assert forbidden not in EXPECTED_DEFAULT_PROMPT
    for required in ("漫画", "对话", "♡", "はぁ", "信达雅"):
        assert required in EXPECTED_DEFAULT_PROMPT


def test_legacy_default_prompt_auto_migrates_while_preserving_custom():
    """Saved untouched legacy prompts upgrade automatically; user-authored custom prompts stay intact."""
    migrated = AppConfig.from_dict({
        "system_prompt": LEGACY_DEFAULT_SYSTEM_PROMPT,
        "llm": {"system_prompt": LEGACY_DEFAULT_SYSTEM_PROMPT},
    })
    assert migrated.llm.system_prompt == EXPECTED_DEFAULT_PROMPT

    custom_prompt = "Custom manga translator prompt"
    kept = AppConfig.from_dict({
        "system_prompt": custom_prompt,
        "llm": {"system_prompt": custom_prompt},
    })
    assert kept.llm.system_prompt == custom_prompt
    assert TranslationEngine(system_prompt=LEGACY_DEFAULT_SYSTEM_PROMPT).system_prompt == EXPECTED_DEFAULT_PROMPT


def test_translation_engine_uses_custom_or_default_prompt():
    """Verify that TranslationEngine respects custom prompt or falls back to default."""
    engine_default = TranslationEngine()
    assert engine_default.system_prompt == EXPECTED_DEFAULT_PROMPT

    custom_prompt = "Custom manga translator prompt"
    engine_custom = TranslationEngine(system_prompt=custom_prompt)
    assert engine_custom.system_prompt == custom_prompt


def test_test_connection_missing_api_key_raises_value_error():
    """Verify that test_connection raises ValueError if api_key is missing for non-custom provider."""
    engine = TranslationEngine(provider="deepseek", api_key="")
    with pytest.raises(ValueError, match="未配置 API Key"):
        engine.test_connection("Hello")


@patch("requests.post")
def test_test_connection_success(mock_post):
    """Verify that test_connection correctly extracts and returns translated text from response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "你好！这是一次翻译连通性测试。"
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    engine = TranslationEngine(provider="deepseek", api_key="sk-test-key", model="deepseek-chat")
    result = engine.test_connection("Hello! This is a translation connectivity test.")

    assert result == "你好！这是一次翻译连通性测试。"
    assert mock_post.called
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["model"] == "deepseek-chat"
    assert call_kwargs["json"]["messages"][0]["content"] == EXPECTED_DEFAULT_PROMPT
    assert "Hello! This is a translation connectivity test." in call_kwargs["json"]["messages"][1]["content"]


@patch("requests.post")
def test_test_connection_http_error(mock_post):
    """Verify that test_connection raises RuntimeError with error message when HTTP fails."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {
        "error": {
            "message": "Incorrect API key provided"
        }
    }
    mock_response.text = '{"error": {"message": "Incorrect API key provided"}}'
    mock_post.return_value = mock_response

    engine = TranslationEngine(provider="openai", api_key="sk-bad-key", model="gpt-4o-mini")
    with pytest.raises(RuntimeError, match="HTTP 401: Incorrect API key provided"):
        engine.test_connection("Hello")


@patch("requests.post")
def test_test_connection_network_exception(mock_post):
    """Verify that test_connection handles network timeout or connection error cleanly."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Failed to establish a new connection")

    engine = TranslationEngine(provider="custom", custom_endpoint="http://127.0.0.1:9999/v1", model="test-model")
    with pytest.raises(RuntimeError, match="网络连接失败或超时"):
        engine.test_connection("Hello")
