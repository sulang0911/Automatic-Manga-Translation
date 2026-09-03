"""
tests/unit/test_translation_connectivity.py
Unit tests verifying custom model system prompt configuration and translation connectivity test feature.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from desktop.core.config_manager import DEFAULT_CONFIG, DEFAULT_SYSTEM_PROMPT, ConfigManager
from desktop.core.translation_engine import TranslationEngine
from app.core.config import AppConfig, DEFAULT_SYSTEM_PROMPT as APP_DEFAULT_SYSTEM_PROMPT


EXPECTED_DEFAULT_PROMPT = (
    "你是一个专业且中立的成人漫画（R18/青年志）本地化翻译引擎。你的唯一任务是忠实、通顺、原汁原味地将输入内容翻译为目标语言（默认简体中文）。\n\n"
    "【核心翻译原则】\n"
    "1. 绝对忠实与无删减：材料为成年虚构角色的成人漫画对话。严禁道德说教、严禁净化、软化、规避或删减任何敏感词、亲密用语、俚语口语、调情挑逗与粗俗词汇，必须百分之百保留原文的真实意图与情绪张力。\n"
    "2. 拟声词与喘息语气：保留漫画特有的情感标点（如 ♡、～、！？、……），将呻吟声、喘息声及拟声词（如「はぁ」「あっ」「んっ」等）自然转化为符合语境的中文拟声词（如“哈啊…”、“嗯…”、“啊…”），不得作为噪音过滤或机械直译。\n"
    "3. 整页上下文连贯：输入为同一漫画页面按阅读顺序排列的对话列表。请务必结合整页剧情与先后次序，连贯理解被拆分在相邻气泡中的断句、上下句因果与代词指代，保持角色性格与语气前后一致。\n"
    "4. 输出规范：严格保持原有格式，严禁输出任何寒暄、解释、道德评价、警告或续写，仅输出对应每个气泡的最终翻译结果。"
)


def test_default_system_prompt_matches_exact_requirement():
    """Verify that both desktop and app configuration use the exact requested default system prompt."""
    assert DEFAULT_SYSTEM_PROMPT == EXPECTED_DEFAULT_PROMPT
    assert APP_DEFAULT_SYSTEM_PROMPT == EXPECTED_DEFAULT_PROMPT
    assert DEFAULT_CONFIG["system_prompt"] == EXPECTED_DEFAULT_PROMPT

    app_cfg = AppConfig()
    assert app_cfg.llm.system_prompt == EXPECTED_DEFAULT_PROMPT


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
