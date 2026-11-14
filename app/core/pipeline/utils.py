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

# Matches a speaker prefix at the start of a line or mid-line after sentence/quote punctuation:
# e.g. (Suk), (Jun), [Yuna], (锡), （锡）, (俊), （俊）, [由奈], 【由奈】
SPEAKER_LINE_RE = re.compile(
    r'^\s*[\(\[\{（【〔][A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af_.\- ]{1,14}[\)\]\}）】〕]'
)
SPEAKER_SPLIT_RE = re.compile(
    r'(?:^|(?<=[。！？!?.”"」』）】〕\s]))[ \t]*(?=[\(\[\{（【〔][^)\]\}）】〕\n]{1,14}[\)\]\}）】〕]\s*[“"「『（\(]?)'
)


def format_speaker_dialogue(text: str, original_text: str = "") -> Tuple[str, int]:
    """
    Ensures multi-speaker dialogue lines (e.g. '(Suk) "..." (Jun) "..."' or '(锡)「...」(俊)「...」')
    are cleanly separated onto individual lines with '\\n' rather than merged into one paragraph.
    Returns (formatted_text, speaker_turn_count).
    """
    if not text:
        return text, 0

    norm = SPEAKER_SPLIT_RE.sub("\n", text.strip())
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]
    speaker_cnt = sum(1 for ln in lines if SPEAKER_LINE_RE.match(ln))
    if speaker_cnt >= 2:
        return "\n".join(lines), speaker_cnt

    if original_text:
        orig_norm = SPEAKER_SPLIT_RE.sub("\n", original_text.strip())
        orig_lines = [ln.strip() for ln in orig_norm.split("\n") if ln.strip()]
        orig_spk = sum(1 for ln in orig_lines if SPEAKER_LINE_RE.match(ln))
        t_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if orig_spk >= 2 and len(t_lines) >= 2:
            return "\n".join(t_lines), len(t_lines)

    return text, speaker_cnt


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
    
    normalized = re.sub(r'\bGF\b', 'girlfriend', text)
    normalized = re.sub(r"\bBF['’]s\b", "boyfriend's", normalized)
    normalized = re.sub(r'\bBF\b', 'boyfriend', normalized)
    normalized = re.sub(r'\b(no[-_ ]?fap)\b', 'no-fap', normalized, flags=re.IGNORECASE)
    
    return normalized


def clean_ocr_syntax(text: str) -> str:
    """
    Sanitizes OCR-detected English text while preserving speaker-turn line breaks ('\\n').
    """
    if not text:
        return ""
    
    t = text.strip()
    t = re.sub(r'(\b\w+)\s*[:;]\s*([a-zA-Z]+)\b', r'\1 \2', t)
    t = re.sub(r'[:;]\s*$', '', t)
    t = t.replace('》', '!').replace('《', '').replace('「', '"').replace('」', '"')
    
    # Normalize horizontal double spaces while preserving '\n' line breaks
    raw_lines = [re.sub(r'[ \t]{2,}', ' ', ln).strip() for ln in t.splitlines() if ln.strip()]
    t = "\n".join(raw_lines)
    t, _ = format_speaker_dialogue(t)
    return t


def clean_translation_syntax(translated_text: str, original_text: str = "") -> str:
    """
    Cleans up translation post-processing artifacts and preserves line-by-line speaker turns.
    """
    if not translated_text:
        return ""
    
    t = translated_text.strip()
    t = re.sub(r'[：:]\s*[\u4e00-\u9fa5a-zA-Z]{1,4}\s*$', '', t)
    t = re.sub(r'[：:;；\-_]\s*$', '', t)
    
    if "BF" in t:
        t = re.sub(r'\bBF\b', '男朋友', t)
    if "GF" in t:
        t = re.sub(r'\bGF\b', '女朋友', t)

    t, _ = format_speaker_dialogue(t, original_text)
    return t.strip()


from app.core.translation.prompt_templates import (
    normalize_source_lang,
    is_auto_source_lang,
    source_lang_to_ocr_lang,
)


def prioritize_english_routing(
    source_lang: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    sample_text: Optional[str] = None
) -> bool:
    if source_lang:
        norm = normalize_source_lang(source_lang)
        if norm == "en":
            return True
        if norm != "auto":
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


def build_ocr_engine_kwargs(
    config: Optional[Dict[str, Any]],
    lang: str,
    is_manual: bool,
    source_lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds unified OCREngine initialization keyword arguments from app config dict."""
    cfg = config if isinstance(config, dict) else {}
    ocr_cfg = cfg.get("ocr", {}) if isinstance(cfg.get("ocr"), dict) else {}
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm"), dict) else {}

    vlm_fallback = bool(ocr_cfg.get("vlm_fallback", cfg.get("ocr_vlm_fallback", False)))
    vlm_llm_obj = None
    if vlm_fallback:
        from types import SimpleNamespace
        vlm_llm_obj = SimpleNamespace(
            provider=llm_cfg.get("provider", cfg.get("provider", "openai")),
            api_key=llm_cfg.get("api_key", cfg.get("api_key", "")),
            model=llm_cfg.get("model", cfg.get("model", "")),
            endpoint=llm_cfg.get("endpoint", cfg.get("custom_endpoint", "")),
            proxy_url=llm_cfg.get("proxy_url", cfg.get("proxy_url", "")),
        )

    return {
        "engine_type": ocr_cfg.get("engine", cfg.get("ocr_engine", "ctd")),
        "use_gpu": not ocr_cfg.get("force_cpu", False) if "force_cpu" in ocr_cfg else cfg.get("use_gpu", True),
        "lang": lang,
        "enable_ensemble_detection": ocr_cfg.get("ensemble_detection", cfg.get("ocr_ensemble_detection", False)),
        "enable_ensemble_recognition": ocr_cfg.get("ensemble_recognition", cfg.get("ocr_ensemble_recognition", False)),
        "is_manual": is_manual,
        "source_lang": source_lang,
        "rec_mode": ocr_cfg.get("rec_mode", cfg.get("ocr_rec_mode", "smart")),
        "en_recognizer": ocr_cfg.get("en_recognizer", cfg.get("ocr_en_recognizer", "ppocr_v6")),
        "zh_recognizer": ocr_cfg.get("zh_recognizer", cfg.get("ocr_zh_recognizer", "ppocr_v5_server")),
        "ja_recognizer": ocr_cfg.get("ja_recognizer", cfg.get("ocr_ja_recognizer", "manga_ocr")),
        "paddle_ocr_version": ocr_cfg.get("paddle_ocr_version", cfg.get("ocr_paddle_version", "PP-OCRv5")),
        "vlm_fallback": vlm_fallback,
        "vlm_conf_threshold": float(ocr_cfg.get("vlm_conf_threshold", cfg.get("ocr_vlm_conf_threshold", 0.72))),
        "vlm_llm_config": vlm_llm_obj,
    }
