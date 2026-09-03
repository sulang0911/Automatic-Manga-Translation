"""
tests/unit/test_bubble_alignment.py
Unit tests for multi-tier bubble translation alignment, semantic healing, and reading order.
"""
import pytest
from app.core.models import TranslationBlock, ReadingOrderMode
from app.core.translation.json_parser import align_translations_to_blocks, _text_similarity
from app.core.ocr.reading_order import sort_reading_order


def test_text_similarity():
    # Exact match
    assert _text_similarity("Hello World", "Hello World") == 1.0
    # Case and whitespace insensitive
    assert _text_similarity("Hello World", "hello   world") == 1.0
    # Substring / partial sentence
    assert _text_similarity(
        "What? You want me to introduce you to my sister?",
        "What? You want me to introduce you to my sister? Dude, I'm telling you..."
    ) >= 0.8


def test_semantic_healing_of_swapped_ids():
    """
    Simulates the exact user bug:
    LLM output the ID 'aba5e381' for the short label text and 'c7f1904a' for the large dialogue.
    Tier 1 semantic matching should automatically bind each translation to the correct bubble!
    """
    b_dialogue = TranslationBlock(
        id="aba5e381",
        original_text="What? You want me to introduce you to my sister? Dude, I'm telling you, you don't want to date my sister! Annie is so freakishly strong like a gorilla! She's gonna beat you up if you ever get into a fight!",
        reading_order=1
    )
    b_label = TranslationBlock(
        id="c7f1904a",
        original_text="Annie (19 yo) Avery's sister",
        reading_order=2
    )

    # LLM wrongly swapped IDs in its JSON response
    llm_swapped_response = [
        {
            "id": "aba5e381",  # Wrong ID! Belongs to dialogue, but LLM put it on label
            "original_text": "Annie (19 yo) Avery's sister",
            "translated_text": "安妮 (19岁) 艾very的妹妹",
            "type": "bubble"
        },
        {
            "id": "c7f1904a",  # Wrong ID! Belongs to label, but LLM put it on dialogue
            "original_text": "What? You want me to introduce you to my sister? Dude, I'm telling you, you don't want to date my sister! Annie is so freakishly strong like a gorilla!",
            "translated_text": "什么？你想让我介绍我妹妹给你？老兄，我告诉你，你绝对不会想跟我妹妹约会的！安妮力气大得像大猩猩一样！",
            "type": "bubble"
        }
    ]

    blocks = [b_dialogue, b_label]
    align_translations_to_blocks(llm_swapped_response, blocks)

    # b_dialogue should have the dialogue translation
    assert "什么" in b_dialogue.translated_text
    assert "安妮" not in b_dialogue.translated_text[:10]

    # b_label should have the label translation
    assert "安妮" in b_label.translated_text
    assert "什么" not in b_label.translated_text


def test_prefix_hash_id_match():
    """
    Verifies that 4-character ID tag ('#aba5') matches 8-character ID ('aba5e381').
    """
    b1 = TranslationBlock(id="aba5e381", original_text="Hello world")
    llm_response = [{"id": "#aba5", "translated_text": "你好世界"}]

    align_translations_to_blocks(llm_response, [b1])
    assert b1.translated_text == "你好世界"


def test_integer_reading_order_match():
    """
    Verifies that sequential integer IDs ('1', '2') match blocks by reading order.
    """
    b1 = TranslationBlock(id="uuid_a", original_text="First", reading_order=1)
    b2 = TranslationBlock(id="uuid_b", original_text="Second", reading_order=2)

    llm_response = [
        {"id": 1, "translated_text": "第一句"},
        {"id": 2, "translated_text": "第二句"}
    ]

    align_translations_to_blocks(llm_response, [b1, b2])
    assert b1.translated_text == "第一句"
    assert b2.translated_text == "第二句"


def test_length_inversion_anomaly_self_healing():
    """
    Verifies that when no original_text is returned, but the LLM swapped a 200-char dialogue
    with a 10-char label, the length ratio anomaly detector catches and auto-heals the swap.
    """
    b_long = TranslationBlock(
        id="uuid_long",
        original_text="What? You want me to introduce you to my sister? Dude, I'm telling you, you don't want to date my sister! Annie is so freakishly strong like a gorilla! She's gonna beat you up if you ever get into a fight!"
    )
    b_short = TranslationBlock(
        id="uuid_short",
        original_text="Annie (19 yo)"
    )

    # LLM output short translation for long block and long translation for short block
    llm_swapped = [
        {"id": "uuid_long", "translated_text": "安妮 (19岁)"},
        {"id": "uuid_short", "translated_text": "什么？你想让我介绍我妹妹给你？老兄，我告诉你，你绝对不会想跟我妹妹约会的！安妮力气大得像大猩猩一样！"}
    ]

    align_translations_to_blocks(llm_swapped, [b_long, b_short])
    assert "什么" in b_long.translated_text
    assert "安妮" in b_short.translated_text


def test_western_ltr_reading_order():
    """
    Verifies Western comic reading order: Left bubble comes before Right bubble on same tier.
    """
    b_left = TranslationBlock(id="left", xmin=10.0, ymin=20.0, xmax=35.0, ymax=40.0)
    b_right = TranslationBlock(id="right", xmin=60.0, ymin=22.0, xmax=85.0, ymax=42.0)

    sorted_ltr = sort_reading_order([b_right, b_left], mode=ReadingOrderMode.WESTERN_LTR.value)
    assert sorted_ltr[0].id == "left"
    assert sorted_ltr[1].id == "right"
    assert sorted_ltr[0].reading_order == 1
    assert sorted_ltr[1].reading_order == 2