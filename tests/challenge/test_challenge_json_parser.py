"""
tests/challenge/test_challenge_json_parser.py
Empirical Adversarial Challenge Suite 1: 7-Stage Robust JSON Parser Torture Tests.
Empirically separates robust behaviors from confirmed parser vulnerabilities.
"""
import pytest
import json
from app.core.translation.json_parser import (
    parse_llm_json_response,
    _repair_unclosed_json,
    _salvage_translation_blocks,
    extract_translation_map
)


# =========================================================================
# SECTION 1: ROBUST BEHAVIORS (CONFIRMED RESILIENT)
# =========================================================================

def test_pass_thinking_tags_with_distractor_json():
    """Confirms <think> tags containing distractor JSON are stripped before parsing."""
    raw = (
        "<think>\n"
        "Let's see: maybe we should return [{\"id\": \"fake\", \"translated_text\": \"wrong\"}]\n"
        "</think>\n"
        "[\n"
        '  {"id": "t_think_real", "translated_text": "真正翻译内容", "type": "bubble"}\n'
        "]\n"
    )
    result = parse_llm_json_response(raw)
    trans_map = extract_translation_map(result)
    assert "t_think_real" in trans_map
    assert trans_map["t_think_real"]["translated_text"] == "真正翻译内容"
    assert "fake" not in trans_map


def test_pass_case_insensitive_multiline_thinking_tags():
    """Confirms case-insensitive <THINK> tags and multiple reasoning blocks are handled."""
    raw = (
        "<THINK>\nStep 1: analyze context.\n</THINK>\n"
        "<think>\nStep 2: refine wording.\n</think>\n"
        "```json\n"
        '[\n  {"id": "t_think_multi", "translated_text": "多段思考标签", "type": "onomatopoeia"}\n]\n'
        "```\n"
    )
    result = parse_llm_json_response(raw)
    trans_map = extract_translation_map(result)
    assert "t_think_multi" in trans_map
    assert trans_map["t_think_multi"]["type"] == "onomatopoeia"


def test_pass_unclosed_markdown_fence():
    """Confirms opening ```json without closing ``` fence is salvaged."""
    unclosed = (
        "Sure, here is the translated JSON:\n"
        "```json\n"
        '[\n  {"id": "t_unclosed_1", "translated_text": "未闭合代码块", "type": "bubble"}\n]'
    )
    result = parse_llm_json_response(unclosed)
    trans_map = extract_translation_map(result)
    assert "t_unclosed_1" in trans_map
    assert trans_map["t_unclosed_1"]["translated_text"] == "未闭合代码块"


def test_pass_truncated_string_closing():
    """Confirms token cutoff in the middle of a string value auto-closes quotes & brackets."""
    truncated = (
        "[\n"
        '  {"id": "t_trunc_1", "translated_text": "完整气泡", "type": "bubble"},\n'
        '  {"id": "t_trunc_2", "translated_text": "突然中断的文'
    )
    repaired = _repair_unclosed_json(truncated)
    assert repaired.endswith("]") or repaired.endswith("}")
    result = parse_llm_json_response(truncated)
    trans_map = extract_translation_map(result)
    assert "t_trunc_1" in trans_map


def test_pass_truncated_after_comma():
    """Confirms truncation right after a comma is salvaged."""
    truncated = '[\n  {"id": "t_trunc_comma", "translated_text": "逗号截断测试", "type": "bubble"},\n'
    result = parse_llm_json_response(truncated)
    trans_map = extract_translation_map(result)
    assert "t_trunc_comma" in trans_map


def test_pass_trailing_commas_in_objects_and_arrays():
    """Confirms redundant trailing commas in objects and arrays are stripped."""
    bad_json = (
        '[\n'
        '  {"id": "t_trail_1", "translated_text": "尾随逗号 1", "type": "bubble",},\n'
        '  {"id": "t_trail_2", "translated_text": "尾随逗号 2", "type": "onomatopoeia",,},\n'
        ']'
    )
    result = parse_llm_json_response(bad_json)
    trans_map = extract_translation_map(result)
    assert "t_trail_1" in trans_map
    assert "t_trail_2" in trans_map


def test_pass_nested_envelope_variations():
    """Confirms envelope variations (translations, blocks, data, etc.) are unpacked."""
    for key in ["translations", "blocks", "data", "results"]:
        envelope = {key: [{"id": f"t_{key}", "translated_text": f"ok {key}", "type": "bubble"}]}
        result = parse_llm_json_response(json.dumps(envelope))
        trans_map = extract_translation_map(result)
        assert f"t_{key}" in trans_map


def test_pass_empty_and_corrupt_inputs_raise():
    """Confirms empty input or pure conversational text raises ValueError."""
    with pytest.raises(ValueError):
        parse_llm_json_response("")
    with pytest.raises(ValueError):
        parse_llm_json_response("Hello, I cannot translate this.")


# =========================================================================
# SECTION 2: PREVIOUSLY CONFIRMED VULNERABILITIES (NOW FIXED BY M2 HARDENING)
# =========================================================================

def test_fixed_nested_markdown_code_fences():
    r"""
    PREVIOUSLY BUG 1 (now FIXED by M2 hardening): Nested markdown fences.
    Pass 2 now iterates through all code fences searching for valid JSON containers.
    """
    nested = (
        "Here is the translation result:\n"
        "```markdown\n"
        "Some commentary before fence\n"
        "```json\n"
        "[\n"
        '  {"id": "t_nest_1", "translated_text": "嵌套代码块测试", "type": "bubble"}\n'
        "]\n"
        "```\n"
        "```\n"
    )
    # After M2 hardening, nested fences are parsed correctly
    result = parse_llm_json_response(nested)
    trans_map = extract_translation_map(result)
    assert "t_nest_1" in trans_map
    assert trans_map["t_nest_1"]["translated_text"] == "嵌套代码块测试"


def test_fixed_multiple_code_fences_non_json_first():
    """
    PREVIOUSLY BUG 2 (now FIXED by M2 hardening): Multiple code fences with non-JSON first.
    Pass 2 now prioritizes ```json fences and skips non-JSON fences.
    """
    multi_fence = (
        "Original prompt analysis:\n"
        "```text\n"
        "Japanese manga text analyzed.\n"
        "```\n"
        "Translation payload:\n"
        "```json\n"
        "[\n"
        '  {"id": "t_multi_1", "translated_text": "多代码块解析", "type": "bubble"}\n'
        "]\n"
        "```\n"
    )
    # After M2 hardening, the correct ```json fence is found
    result = parse_llm_json_response(multi_fence)
    trans_map = extract_translation_map(result)
    assert "t_multi_1" in trans_map
    assert trans_map["t_multi_1"]["translated_text"] == "多代码块解析"


def test_fixed_single_block_dict_by_extract_translation_map():
    """
    PREVIOUSLY BUG 3 (now FIXED by M2 hardening): Single block dict is now handled correctly.
    extract_translation_map detects single-block dicts with 'id' key and maps them properly.
    """
    single_block_json = '{"id": "b1", "translated_text": "单独一个气泡", "type": "bubble"}'
    parsed = parse_llm_json_response(single_block_json)
    assert isinstance(parsed, dict)

    trans_map = extract_translation_map(parsed)
    # FIXED: "b1" is correctly mapped as the block ID
    assert "b1" in trans_map
    assert trans_map["b1"]["translated_text"] == "单独一个气泡"


def test_fixed_salvage_preserves_commas_in_quoted_dialogue():
    """
    M2 hardening improved salvage regex to support quoted strings, preserving
    commas within quoted dialogue. Unquoted freeform text still truncates at commas
    (expected salvage behavior for unstructured text).
    """
    # Quoted text: commas are preserved
    quoted_raw = (
        'id: b_quoted\n'
        'translated_text: "Hello, world! This is a long sentence, right?"\n'
        'type: bubble\n'
    )
    salvaged = _salvage_translation_blocks(quoted_raw)
    assert len(salvaged) == 1
    assert "Hello, world!" in salvaged[0]["translated_text"]
    assert "right?" in salvaged[0]["translated_text"]

    # Unquoted text: truncates at first comma (expected for salvage mode)
    unquoted_raw = (
        "id: b_unquoted\n"
        "translated_text: Hello, world!\n"
        "type: bubble\n"
    )
    salvaged2 = _salvage_translation_blocks(unquoted_raw)
    assert len(salvaged2) == 1
    assert salvaged2[0]["translated_text"] == "Hello"


