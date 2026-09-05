"""
app/core/pipeline.py
Core pipeline execution utilities, language routing, domain slang mapping, and translation post-processing.
Provides unified English language prioritization, syntax artifact cleanup, and comic slang disambiguation.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Domain slang and abbreviations dictionary for manga / comic translation
DOMAIN_SLANG_MAP: Dict[str, str] = {
    "GF": "girlfriend",
    "gf": "girlfriend",
    "BF": "boyfriend",
    "bf": "boyfriend",
    "BF's": "boyfriend's",
    "bf's": "boyfriend's",
    "BFs": "boyfriends",
    "bfs": "boyfriends",
    "no-fap": "no-fap",
    "nofap": "no-fap",
    "No-fap": "no-fap",
    "No-Fap": "no-fap",
}

DOMAIN_SLANG_TRANSLATIONS: Dict[str, str] = {
    "gf": "女朋友",
    "girlfriend": "女朋友",
    "bf": "男朋友",
    "boyfriend": "男朋友",
    "bf's": "男朋友的",
    "no-fap": "禁欲",
    "nofap": "禁欲",
}


def normalize_domain_slang(text: str) -> str:
    """
    Normalizes domain slang and common manga abbreviations (GF, BF, no-fap)
    so downstream translation engines can process them with maximum semantic fidelity.
    """
    if not text:
        return ""
    
    # 1. GF -> girlfriend (when used as standalone noun)
    normalized = re.sub(r'\bGF\b', 'girlfriend', text)
    # 2. BF's -> boyfriend's
    normalized = re.sub(r"\bBF['’]s\b", "boyfriend's", normalized)
    # 3. BF -> boyfriend
    normalized = re.sub(r'\bBF\b', 'boyfriend', normalized)
    # 4. no-fap / nofap variations
    normalized = re.sub(r'\b(no[-_ ]?fap)\b', 'no-fap', normalized, flags=re.IGNORECASE)
    
    return normalized


def clean_ocr_syntax(text: str) -> str:
    """
    Sanitizes OCR-detected English text:
    - Removes spurious mid-sentence colons inserted before fragmented words (e.g. '2 weeks: dress' -> '2 weeks dress')
    - Removes trailing colons at the end of speech bubbles (e.g. 'sisters:' -> 'sisters')
    - Eliminates orphan brackets or misdetected symbols (e.g. 'What naught》' -> 'What naughty')
    """
    if not text:
        return ""
    
    t = text.strip()
    
    # Fix spurious colon/semicolon before word: e.g. "2 weeks: dress" -> "2 weeks dress", "again: days" -> "again days"
    t = re.sub(r'(\b\w+)\s*[:;]\s*([a-zA-Z]+)\b', r'\1 \2', t)
    
    # Fix trailing colon or semicolon at end of bubble: e.g. "sisters:" -> "sisters"
    t = re.sub(r'[:;]\s*$', '', t)
    
    # Remove spurious CJK brackets/quotes falsely recognized from English punctuation
    t = t.replace('》', '!').replace('《', '').replace('「', '"').replace('」', '"')
    
    # Normalize double spaces
    t = re.sub(r'\s{2,}', ' ', t).strip()
    
    return t


def clean_translation_syntax(translated_text: str, original_text: str = "") -> str:
    """
    Cleans up translation post-processing artifacts and syntax errors:
    - Strips ungrammatical trailing colons or broken words (e.g. "：衣服", "：天", "：裙子")
      caused by OCR line fragment displacement.
    - Strips orphan trailing punctuation (e.g. "这件新裙子也是你姐姐的：")
    - Fixes slang translations when needed.
    """
    if not translated_text:
        return ""
    
    t = translated_text.strip()
    
    # 1. Remove trailing colon with orphan fragment word: e.g. "……在家具上射精：衣服" -> "……在家具上射精"
    t = re.sub(r'[：:]\s*[\u4e00-\u9fa5a-zA-Z]{1,4}\s*$', '', t)
    
    # 2. Remove trailing colons, semicolons, or broken hyphens at the very end
    t = re.sub(r'[：:;；\-_]\s*$', '', t)
    
    # 3. Fix slang if translated as untranslated abbreviation
    if "BF" in t:
        t = re.sub(r'\bBF\b', '男朋友', t)
    if "GF" in t:
        t = re.sub(r'\bGF\b', '女朋友', t)
        
    return t.strip()


def prioritize_english_routing(
    source_lang: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    sample_text: Optional[str] = None
) -> bool:
    """
    Determines whether the execution pipeline should prioritize English language OCR routing (EasyOCR)
    and strictly gate out Japanese-only engines (Manga-OCR).
    
    Returns True if:
    1. source_lang is explicitly English ('en', 'english', etc.)
    2. source_lang is 'auto' / '自动识别' and sample text or image shows English predominance.
    """
    if source_lang:
        sl = str(source_lang).lower().strip()
        if any(w in sl for w in ["en", "eng", "english", "latin"]):
            return True
        if sl in ["auto", "unknown", "自动识别", "自动"]:
            # Auto-detection required
            pass
        elif any(w in sl for w in ["japan", "ja", "日", "chinese", "zh", "中", "korean", "ko"]):
            # Explicit non-English language requested by user
            return False

    if sample_text:
        latin_words = re.findall(r'[a-zA-Z]{2,}', sample_text)
        cjk_chars = [c for c in sample_text if ('\u4e00' <= c <= '\u9fff') or ('\u3040' <= c <= '\u309f') or ('\u30a0' <= c <= '\u30ff')]
        if len(latin_words) >= 3 and len("".join(latin_words)) > len(cjk_chars) * 2:
            return True
        if len(latin_words) >= 2 and len(cjk_chars) == 0:
            return True

    return False


def post_process_translation_blocks(blocks: List[Any]) -> List[Any]:
    """
    Post-processes translation results across a list of TranslationBlock objects or dicts:
    - Cleans up trailing colons or broken words (e.g. "：衣服", "：天")
    - Fixes ungrammatical trailing punctuation
    """
    for b in blocks:
        if isinstance(b, dict):
            trans = b.get("translated_text", "")
            orig = b.get("original_text", "")
            if trans:
                b["translated_text"] = clean_translation_syntax(trans, orig)
        elif hasattr(b, "translated_text"):
            trans = getattr(b, "translated_text", "")
            orig = getattr(b, "original_text", "")
            if trans:
                setattr(b, "translated_text", clean_translation_syntax(trans, orig))
    return blocks
