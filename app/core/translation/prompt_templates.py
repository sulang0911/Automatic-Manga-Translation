"""
app/core/translation/prompt_templates.py
Manga localization system prompts preserving dialogue continuity, honorifics, tone, and SFX classification.
"""
import json
from typing import List, Optional
from app.core.models import TranslationBlock
from app.core.translation.base import TranslationContext


def normalize_source_lang(lang: Optional[str]) -> str:
    """
    Normalizes a user-specified or config source language to a canonical code:
    'auto', 'ja', 'en', 'ko', 'chs', 'cht'.
    """
    if not lang:
        return "auto"
    l = str(lang).strip().lower()
    if l in ("auto", "自动识别", "自动", "unknown", "none", ""):
        return "auto"
    if any(w in l for w in ["japan", "ja", "日"]):
        return "ja"
    if any(w in l for w in ["en", "eng", "english", "latin", "英"]):
        return "en"
    if any(w in l for w in ["korean", "ko", "hangul", "韩"]):
        return "ko"
    if any(w in l for w in ["cht", "zh-tw", "chinese_cht", "ch_tra", "繁"]):
        return "cht"
    if any(w in l for w in ["chinese", "chs", "ch", "zh", "zh-cn", "ch_sim", "中"]):
        return "chs"
    return l


def is_auto_source_lang(lang: Optional[str]) -> bool:
    """Returns True if the source language is set to automatic identification."""
    return normalize_source_lang(lang) == "auto"


def source_lang_to_ocr_lang(lang: Optional[str]) -> str:
    """
    Maps a canonical or raw source language to the internal OCR engine language parameter.
    'en' -> 'en'
    'ko' -> 'korean'
    'chs' -> 'ch'
    'cht' -> 'chinese_cht'
    'ja' -> 'japan'
    'auto' -> 'auto'
    """
    norm = normalize_source_lang(lang)
    if norm == "en":
        return "en"
    elif norm == "ko":
        return "korean"
    elif norm == "chs":
        return "ch"
    elif norm == "cht":
        return "chinese_cht"
    elif norm == "ja":
        return "japan"
    return "auto"


class PromptTemplates:
    @staticmethod
    def is_auto_source_lang(lang: str) -> bool:
        """Determines if source language is configured for automatic detection."""
        return is_auto_source_lang(lang)

    @staticmethod
    def build_text_system_prompt(
        source_lang: str,
        target_lang: str,
        context: Optional[TranslationContext] = None,
        reading_order_mode: str = "manga_rtl"
    ) -> str:
        norm = normalize_source_lang(source_lang)
        is_auto = (norm == "auto")
        source_desc = "automatically detecting the source language (Japanese, Korean, English, etc.)" if is_auto else f"translating from {source_lang}"

        explicit_mode = False
        if context and context.reading_direction:
            dir_clean = context.reading_direction.lower()
            if dir_clean in ("ltr", "western_ltr"):
                reading_order_mode = "western_ltr"
                explicit_mode = True
            elif dir_clean in ("ttb", "webtoon_ttb", "vertical"):
                reading_order_mode = "webtoon_ttb"
                explicit_mode = True
            elif dir_clean in ("rtl", "manga_rtl"):
                reading_order_mode = "manga_rtl"
                explicit_mode = True

        if not explicit_mode:
            if norm == "en":
                reading_order_mode = "western_ltr"
            elif norm == "ko":
                reading_order_mode = "webtoon_ttb"

        if reading_order_mode == "western_ltr":
            reading_order_desc = "Western comic narrative reading order (Left-to-Right, Top-to-Bottom)"
        elif reading_order_mode == "webtoon_ttb":
            reading_order_desc = "Webtoon continuous vertical reading order (Top-to-Bottom)"
        elif norm in ("chs", "cht"):
            reading_order_desc = "Chinese manhua narrative reading order (Top-to-Bottom or Left-to-Right)"
        else:
            reading_order_desc = "Japanese manga narrative reading order (Right-to-Left, Top-to-Bottom)"

        if is_auto:
            rule_1 = (
                "1. Language Auto-Detection:\n"
                f"   - Automatically detect whether each text block is Japanese, Korean, English, or Chinese based on characters/kana/hangul/alphabet and translate faithfully into {target_lang}.\n"
            )
        else:
            rule_1 = (
                f"1. Source Language Fidelity:\n"
                f"   - The source language is strictly specified by the user as {source_lang}. Translate faithfully from {source_lang} into {target_lang}.\n"
                f"   - Do NOT assume the original text is Japanese if the source language is {source_lang}; strictly follow the actual {source_lang} input.\n"
            )

        if norm == "ko":
            rule_3 = (
                "3. Speech Register & Honorifics:\n"
                "   - Preserve Korean speech hierarchy (banmal vs jondaetmal/haeyoche/hasipsioche).\n"
                f"   - Contextually adapt Korean honorific terms (sunbae -> 前辈/学长, hyung/oppa/unnie/noona -> 哥/姐/亲切称谓, nim -> 先生/大人) naturally into {target_lang}.\n"
            )
        elif norm == "en":
            rule_3 = (
                "3. Tone & Colloquial Comic Expressions:\n"
                f"   - Preserve English comic dialogue style, slang, contractions, sarcasm/witty nuance, and character personality accurately into {target_lang}.\n"
            )
        elif norm in ("chs", "cht"):
            rule_3 = (
                "3. Tone & Voice Continuity:\n"
                f"   - Preserve Chinese comic dialogue tone, slang, emotional intensity, and rhetorical expressions faithfully into {target_lang}.\n"
            )
        else:
            rule_3 = (
                "3. Honorifics & Speech Register:\n"
                "   - Preserve character personality: tsundere, polite (keigo), delinquent (ore/tameguchi), archaic/samurai, gyaru, or child speech.\n"
                "   - Handle Japanese honorifics contextually:\n"
                "     * When translating to Chinese: '-san' -> 先生/小姐/同学/自然称呼; '-chan' -> 小X/酱; '-kun' -> 君/同学; '-senpai' -> 前辈/学长; '-sensei' -> 老师; '-sama' -> 大人/阁下.\n"
                "     * When translating to English: Retain common manga honorifics (e.g. Senpai, Sensei, -san) where culturally fitting.\n"
            )

        base_prompt = (
            f"You are a master manga and webtoon localization specialist {source_desc} to {target_lang}.\n"
            "Your objective is to produce dialogue that feels punchy, expressive, and completely natural to native readers of the target language.\n\n"
            "LOCALIZATION PRINCIPLES:\n"
            "- Faithful translation: Preserve authentic dialogue, character voice, tone, and colloquial comic expressions accurately without omission.\n"
            "- Colloquial fluency: Adapt idioms, informal slang, and conversational flow into natural, high-quality dialogue suited for comic publication.\n\n"
            "CRITICAL LOCALIZATION RULES:\n"
            + rule_1
            + "2. Dialogue Continuity & Reading Flow:\n"
            f"   - The dialogue blocks are provided in {reading_order_desc}.\n"
            "   - When an utterance is split across two or more consecutive speech bubbles (e.g. '俺は絶対に...' followed by '...諦めない！'), "
            "maintain sentence flow, tone, and grammatical continuity seamlessly.\n"
            + rule_3
            + "4. Expressive Onomatopoeia & Comic Vocalizations:\n"
            "   - Preserve dramatic punctuation (！？, ……, 〜, ッ) and emotional intensity without robotic literal translation.\n"
            "   - Interjections, sighs, and manga sound effects (SFX) must be translated into idiomatic comic sound effects matching context rather than omitted or transcribed as meaningless noise.\n"
            "5. Classification Guidelines for 'type':\n"
            "   - 'bubble': Regular dialogue, narration captions, thoughts, or any text spoken INSIDE a speech or thought bubble.\n"
            "   - 'onomatopoeia': Hand-drawn sound effects (SFX, e.g. ドキドキ, バン, ドン), physical action noises, screams, or side notes OUTSIDE bubbles.\n"
            "6. Output Format & Strict One-to-One Alignment:\n"
            "   - Return ONLY a valid JSON array of objects. No introductory commentary, no trailing remarks.\n"
            "   - Each object MUST correspond strictly to one input block and contain:\n"
            "     * 'id': string matching the exact input block ID (do NOT renumber or swap IDs).\n"
            "     * 'original_text': the original source text of this block (used as verification anchor).\n"
            "     * 'translated_text': localized translation string.\n"
            "     * 'type': 'bubble' or 'onomatopoeia'.\n"
            "   - CRITICAL: Never swap the translations or IDs of different speech bubbles.\n"
        )

        if context and context.glossary:
            glossary_lines = [f"- {src} => {dst}" for src, dst in context.glossary.items()]
            base_prompt += "\nCHAPTER GLOSSARY (Strictly enforce these terms):\n" + "\n".join(glossary_lines) + "\n"

        if context and context.previous_summary:
            base_prompt += f"\nPREVIOUS PAGE CONTEXT:\n{context.previous_summary}\n"

        return base_prompt

    @staticmethod
    def build_text_user_message(
        blocks: List[TranslationBlock],
        source_lang: str,
        target_lang: str
    ) -> str:
        payload = [
            {
                "index": idx + 1,
                "id": b.id,
                "text": b.original_text,
                "xmin": round(b.xmin, 1),
                "ymin": round(b.ymin, 1),
                "xmax": round(b.xmax, 1),
                "ymax": round(b.ymax, 1)
            }
            for idx, b in enumerate(blocks)
        ]
        is_auto = PromptTemplates.is_auto_source_lang(source_lang)
        src_label = "Auto Detect (Identify Japanese, Korean, English, etc. per block)" if is_auto else source_lang
        return (
            f"Source Language: {src_label}\n"
            f"Target Language: {target_lang}\n\n"
            f"Manga Text Blocks (Sequential Reading Order):\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Translate all blocks faithfully in sequential reading order. "
            "Ensure every translation is strictly bound to its corresponding block ID and original text. "
            "Return strictly a JSON array:\n"
            '[\n  {"id": "...", "original_text": "...", "translated_text": "...", "type": "bubble"}\n]'
        )

    @staticmethod
    def build_vision_system_prompt(target_lang: str, source_lang: Optional[str] = None) -> str:
        norm = normalize_source_lang(source_lang)
        lang_str = "Japanese" if norm in ("ja", "auto") else (source_lang or "comic")
        return (
            "You are a high-precision manga OCR and translation engine.\n"
            f"Task: Detect all text in the manga image, translate from {lang_str} into {target_lang}, and output bounding coordinates.\n\n"
            "Rules:\n"
            f"1. Detect vertical and horizontal {lang_str} text blocks accurately.\n"
            "2. Coordinates (ymin, xmin, ymax, xmax) must be normalized percentages (0.0 to 100.0).\n"
            "3. Identify text_color and bg_color as hex codes (e.g. #000000, #FFFFFF).\n"
            "4. Classify each block as 'bubble' (inside speech bubble) or 'onomatopoeia' (sound effects outside bubbles).\n"
            "5. Return strictly a JSON object conforming to: {\"blocks\": [{\"original_text\": \"...\", \"translated_text\": \"...\", \"ymin\": 10.5, \"xmin\": 20.0, \"ymax\": 30.0, \"xmax\": 40.0, \"text_color\": \"#000000\", \"bg_color\": \"#FFFFFF\", \"type\": \"bubble\"}]}"
        )
