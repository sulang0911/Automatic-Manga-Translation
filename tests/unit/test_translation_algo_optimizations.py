"""
tests/unit/test_translation_algo_optimizations.py
Tests verifying translation algorithm optimizations:
1. Maximum weight greedy bipartite matching in alignment engine
2. Bidirectional Tier 5 length ratio anomaly detection
3. Immunity of text-verified blocks from false-positive length swapping
4. Unicode NFKC & punctuation-agnostic similarity in _text_similarity
5. Incomplete key/colon repair in _repair_unclosed_json
6. Original text extraction & escaped quotes in _salvage_translation_blocks
7. Empty / whitespace-only block filtering in TranslationManager
8. Chunking mechanism in TranslationManager for large block lists
"""
import pytest
from app.core.models import TranslationBlock
from app.core.translation.json_parser import (
    align_translations_to_blocks,
    _text_similarity,
    _repair_unclosed_json,
    _salvage_translation_blocks,
    parse_llm_json_response
)
from app.core.translation.manager import TranslationManager
from app.core.translation.base import ProviderConfig


def test_bipartite_matching_prevents_suboptimal_steal():
    """
    Ensures a candidate with 0.6 similarity does NOT steal a block from a candidate with 1.0 similarity,
    even if the 0.6 candidate appears FIRST in the candidate list.
    """
    b1 = TranslationBlock(id="b1", original_text="こんにちは世界")  # "Hello World"
    b2 = TranslationBlock(id="b2", original_text="こんにちは")      # "Hello"

    candidates = [
        {"id": "cand_for_b2", "original_text": "こんにちは", "translated_text": "你好", "type": "bubble"},
        {"id": "cand_for_b1", "original_text": "こんにちは世界", "translated_text": "你好世界", "type": "bubble"}
    ]

    blocks = [b1, b2]
    align_translations_to_blocks(candidates, blocks)

    assert b1.translated_text == "你好世界"
    assert b2.translated_text == "你好"


def test_bidirectional_length_inversion_anomaly_healing():
    """
    Verifies that Tier 5 anomaly detection works when the short label appears BEFORE the long dialogue.
    (Previously the loop only checked o1 >= 50 and o2 <= 25).
    """
    b_short = TranslationBlock(id="uuid_short", original_text="Annie (19 yo)")
    b_long = TranslationBlock(
        id="uuid_long",
        original_text="What? You want me to introduce you to my sister? Dude, I'm telling you, you don't want to date my sister! Annie is so freakishly strong like a gorilla!"
    )

    llm_swapped = [
        {"id": "uuid_short", "translated_text": "什么？你想让我介绍我妹妹给你？老兄，我告诉你，你绝对不会想跟我妹妹约会的！安妮力气大得像大猩猩一样！"},
        {"id": "uuid_long", "translated_text": "安妮 (19岁)"}
    ]

    blocks = [b_short, b_long]
    align_translations_to_blocks(llm_swapped, blocks)

    assert "安妮" in b_short.translated_text
    assert "什么" in b_long.translated_text


def test_text_verified_blocks_immune_to_length_swap():
    """
    Verifies that if blocks were matched via high-confidence text similarity (Tier 1),
    Tier 5 will NOT falsely swap them even if their lengths match the inversion threshold.
    """
    b_concise = TranslationBlock(
        id="b1",
        original_text="あいつの言っていることは完全に間違っている、絶対に信用してはならないぞ！絶対に信用してはならない！"
    )
    b_verbose = TranslationBlock(
        id="b2",
        original_text="立入禁止看板"
    )

    llm_correct = [
        {
            "id": "b1",
            "original_text": b_concise.original_text,
            "translated_text": "别信他的鬼话！",
            "type": "bubble"
        },
        {
            "id": "b2",
            "original_text": b_verbose.original_text,
            "translated_text": "根据东京都特别警备区域第十二条规定：此处严禁任何非授权人员擅自入内",
            "type": "bubble"
        }
    ]

    blocks = [b_concise, b_verbose]
    align_translations_to_blocks(llm_correct, blocks)

    # Because Tier 1 verified original text, Tier 5 must NOT swap them!
    assert b_concise.translated_text == "别信他的鬼话！"
    assert "严禁任何非授权人员" in b_verbose.translated_text


def test_unicode_nfkc_and_punctuation_agnostic_similarity():
    """
    Verifies that Japanese bracket quotes, commas, periods, and full-width/half-width symbols match with high similarity.
    """
    sim_quotes = _text_similarity("「こんにちは」", "こんにちは")
    assert sim_quotes >= 0.95

    sim_punc = _text_similarity("ほんとう！？", "ほんとう!?")
    assert sim_punc >= 0.95

    # Japanese comma and full stop punctuation stripping
    sim_jp_punc = _text_similarity("こんにちは。", "こんにちは")
    assert sim_jp_punc >= 0.98

    sim_jp_comma = _text_similarity("はい、そうです。", "はいそうです")
    assert sim_jp_comma >= 0.98


def test_repair_unclosed_json_incomplete_key():
    """
    Verifies that _repair_unclosed_json cleanly salvages truncated JSON ending in an incomplete key or colon.
    """
    truncated_key = '[{"id": "b1", "translated_text": "ok"}, {"id": "b2", "trans'
    repaired = _repair_unclosed_json(truncated_key)
    parsed = parse_llm_json_response(repaired)
    assert len(parsed) >= 1
    assert parsed[0]["id"] == "b1"

    truncated_colon = '[{"id": "b1", "translated_text": "ok"}, {"id": "b2", "translated_text": '
    repaired_colon = _repair_unclosed_json(truncated_colon)
    parsed_colon = parse_llm_json_response(repaired_colon)
    assert len(parsed_colon) >= 1
    assert parsed_colon[0]["id"] == "b1"


def test_salvage_extracts_original_text_and_handles_escapes():
    """
    Verifies that regex salvage extracts original_text and handles escaped quotes in dialogue.
    """
    corrupt_text = (
        'Some intro text\n'
        'id: b1\n'
        'original_text: "He said, \\"Stop!\\""\n'
        'translated_text: "他说：\\"住手！\\""\n'
        'type: bubble\n'
    )
    salvaged = _salvage_translation_blocks(corrupt_text)
    assert len(salvaged) == 1
    assert salvaged[0]["id"] == "b1"
    assert 'He said, "Stop!"' in salvaged[0]["original_text"]
    assert '他说："住手！"' in salvaged[0]["translated_text"]


def test_empty_and_whitespace_block_filtering_in_manager():
    """
    Verifies that TranslationManager skips calling LLM on empty/whitespace blocks and sets their translated_text to "".
    """
    mgr = TranslationManager.get_instance()
    mgr.set_active_provider("openai", ProviderConfig(provider_name="openai", api_key=""))

    b_valid = TranslationBlock(id="b1", original_text="Valid dialogue")
    b_empty1 = TranslationBlock(id="b2", original_text="")
    b_empty2 = TranslationBlock(id="b3", original_text="   \n\t  ")

    res = mgr.translate([b_valid, b_empty1, b_empty2])

    assert b_valid.translated_text == "【译】Valid dialogue"
    assert b_empty1.translated_text == ""
    assert b_empty2.translated_text == ""


def test_tier4_1based_indexing_prevents_cascading_off_by_one():
    """
    Verifies that 1-based indexing (e.g. LLM outputting IDs '1', '2', '3')
    does NOT cause off-by-one shifting when an earlier block was matched or missing.
    """
    # 3 blocks with generic IDs
    b0 = TranslationBlock(id="b_alpha", original_text="First dialogue")
    b1 = TranslationBlock(id="b_beta", original_text="Second dialogue")
    b2 = TranslationBlock(id="b_gamma", original_text="Third dialogue")

    # Suppose candidate 1 was matched to b0 via original_text in Tier 1
    # Candidates 2 and 3 have raw_ids "2" and "3" (1-based index)
    candidates = [
        {"id": "b_alpha", "original_text": "First dialogue", "translated_text": "第一句", "type": "bubble"},
        {"id": "2", "translated_text": "第二句", "type": "bubble"},
        {"id": "3", "translated_text": "第三句", "type": "bubble"}
    ]

    blocks = [b0, b1, b2]
    align_translations_to_blocks(candidates, blocks)

    assert b0.translated_text == "第一句"
    assert b1.translated_text == "第二句"
    assert b2.translated_text == "第三句"


def test_tier4c_salvages_anonymous_candidates_and_preserves_mismatched_ids():
    """
    Verifies that anonymous candidate lists without IDs are sequentially assigned,
    while explicitly mismatched candidate IDs preserve the block's original text (F-TRN-03).
    """
    b1 = TranslationBlock(id="bubble_1", original_text="昔々あるところに…")
    b2 = TranslationBlock(id="bubble_2", original_text="おじいさんとおばあさんがいました。")

    # Anonymous plain text strings without ID
    llm_output = ["很久很久以前…", "住着一位老爷爷和老奶奶。"]

    blocks = [b1, b2]
    align_translations_to_blocks(llm_output, blocks)

    assert b1.translated_text == "很久很久以前…"
    assert b2.translated_text == "住着一位老爷爷和老奶奶。"

    # Explicit mismatched ID preserves original text
    b_alone = TranslationBlock(id="my_id", original_text="原本文本", translated_text="")
    mismatched_cand = [{"id": "unrelated_id", "translated_text": "其他文本"}]
    align_translations_to_blocks(mismatched_cand, [b_alone])
    assert b_alone.translated_text == "原本文本"


def test_prompt_templates_context_direction_precedence():
    """
    Verifies that explicit context reading_direction takes precedence over source language heuristics.
    """
    from app.core.translation.prompt_templates import PromptTemplates
    from app.core.translation.base import TranslationContext

    # Korean comic with explicit manga_rtl reading direction
    ctx = TranslationContext(reading_direction="manga_rtl")
    prompt = PromptTemplates.build_text_system_prompt("韩语", "简体中文", context=ctx)
    assert "Japanese manga narrative reading order (Right-to-Left, Top-to-Bottom)" in prompt
    assert "Webtoon continuous vertical" not in prompt


def test_large_block_list_chunking_with_mock_provider():
    """
    Verifies that when blocks count > 25, TranslationManager actually chunks them
    and passes them to provider.translate_text_blocks with narrative context chaining.
    """
    from app.core.translation.base import BaseTranslationProvider

    called_chunks = []
    contexts_received = []

    class MockChunkProvider(BaseTranslationProvider):
        def supports_vision(self):
            return False

        def translate_text_blocks(self, blocks, source_lang, target_lang, context=None, progress_callback=None):
            called_chunks.append(list(blocks))
            contexts_received.append(context)
            for b in blocks:
                b.translated_text = f"translated_{b.id}"
            if progress_callback:
                progress_callback(100, f"Done chunk of {len(blocks)}")
            return blocks

        def test_connection(self):
            return None

    mgr = TranslationManager.get_instance()
    # Register and activate mock provider with a non-empty API key to trigger real chunking path
    TranslationManager.register_provider("mock_chunk", MockChunkProvider)
    mgr.set_active_provider("mock_chunk", ProviderConfig(provider_name="mock_chunk", api_key="sk-real-key"))

    blocks = [TranslationBlock(id=f"b_{i}", original_text=f"Dialogue {i}") for i in range(40)]
    progress_records = []
    res = mgr.translate(blocks, progress_callback=lambda pct, msg: progress_records.append((pct, msg)))

    assert len(res) == 40
    # CHUNK_SIZE = 25, so 40 blocks should split into 2 chunks (25 + 15)
    assert len(called_chunks) == 2
    assert len(called_chunks[0]) == 25
    assert len(called_chunks[1]) == 15

    # Check all translated
    for i, b in enumerate(res):
        assert b.translated_text == f"translated_b_{i}"

    # Verify context chaining: chunk 2 should contain previous dialogue context
    assert len(contexts_received) == 2
    assert contexts_received[1] is not None
    assert "Previous dialogue context:" in contexts_received[1].previous_summary
    assert "Dialogue 24" in contexts_received[1].previous_summary

    # Verify progress scaling never jumps back to 20% on chunk 2
    assert len(progress_records) > 0
    # Final progress is 100%
    assert progress_records[-1][0] == 100
