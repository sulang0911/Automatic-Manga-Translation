"""
app/core/translation/json_parser.py
Robust 7-stage JSON Parser for LLM manga translation outputs.
Handles thinking tags, markdown wrappers, trailing commas, truncated JSON, and syntax salvage.
"""
import json
import re
import difflib
from typing import Any, List, Dict, Optional, Set
from app.core.models import TranslationBlock


def parse_llm_json_response(raw_text: str) -> Any:
    """
    Extracts and normalizes structured JSON payload from model response.
    Pass 1: Strip <think>...</think> reasoning tags (DeepSeek-R1 / Qwen).
    Pass 2: Extract content from markdown code fences (```json ... ```).
    Pass 3: Standard json.loads fast path.
    Pass 4: Bracket/brace container boundary extraction.
    Pass 5: Trailing comma stripping before } or ].
    Pass 6: Truncated JSON balancing / unclosed bracket closure.
    Pass 7: Regex salvage scanner for (id, translated_text, type) triplets.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("收到空响应文本，无法解析翻译结果。")

    # Pass 1: Strip thinking/reasoning tags
    text = re.sub(r'<think>[\s\S]*?</think>', '', raw_text, flags=re.IGNORECASE).strip()

    # Pass 2: Extract code block from markdown fences
    json_match = re.search(r'```(?:json)\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE)
    if json_match:
        text = json_match.group(1).strip()
    else:
        blocks = re.findall(r'```(?:[a-zA-Z0-9_\-]+)?\s*([\s\S]*?)\s*```', text)
        found = None
        for b in blocks:
            b_s = b.strip()
            if ('[' in b_s and ']' in b_s) or ('{' in b_s and '}' in b_s):
                found = b_s
                break
        if found is not None:
            text = found
        elif blocks:
            text = blocks[-1].strip()
        elif text.startswith('```'):
            text = re.sub(r'^```[a-zA-Z]*\n?', '', text).strip()
            text = re.sub(r'\n?```$', '', text).strip()

    # Pass 3: Direct standard parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pass 4: Locate outer container
    first_bracket = text.find('[')
    first_brace = text.find('{')

    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        start_idx = first_bracket
        last_bracket = text.rfind(']')
        candidate = text[start_idx:last_bracket + 1] if last_bracket > start_idx else text[start_idx:]
    elif first_brace != -1:
        start_idx = first_brace
        last_brace = text.rfind('}')
        candidate = text[start_idx:last_brace + 1] if last_brace > start_idx else text[start_idx:]
    else:
        candidate = text

    # Pass 5: Remove trailing commas before closing braces/brackets
    cleaned_candidate = re.sub(r',\s*([}\]])', r'\1', candidate)

    try:
        return json.loads(cleaned_candidate)
    except json.JSONDecodeError:
        pass

    # Pass 6: Auto-close truncated JSON
    repaired = _repair_unclosed_json(cleaned_candidate)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Pass 7: Regex salvage fallback
    salvaged = _salvage_translation_blocks(text)
    if salvaged:
        return salvaged

    raise ValueError(f"无法从大模型响应中解析出结构化 JSON: {raw_text[:200]}...")


def _repair_unclosed_json(text: str) -> str:
    """Balances and closes unclosed braces and brackets for truncated LLM responses."""
    stack = []
    in_string = False
    escape = False

    for char in text:
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char in '{[':
            stack.append(char)
        elif char == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif char == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    repaired = text.rstrip()
    if in_string:
        repaired += '"'

    repaired = re.sub(r',\s*$', '', repaired)

    for opener in reversed(stack):
        if opener == '{':
            repaired += '}'
        elif opener == '[':
            repaired += ']'
    return repaired


def _salvage_translation_blocks(text: str) -> List[Dict[str, Any]]:
    """Extracts id and translated_text pairs via regex when JSON syntax is severely corrupted."""
    items = []
    # Split text into chunks at each 'id' / 'block_id' declaration
    segments = re.split(r'(?=["\']?(?:id|block_id)["\']?\s*[:=])', text, flags=re.IGNORECASE)
    for seg in segments:
        id_match = re.search(
            r'["\']?(?:id|block_id)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]+)["\']?',
            seg, flags=re.IGNORECASE
        )
        text_match = re.search(
            r'["\']?(?:translated_text|text|translation)["\']?\s*[:=]\s*(?:"([^"]*)"|\'([^\']*)\'|([^"\'\n\r,}]+))',
            seg, flags=re.IGNORECASE
        )
        type_match = re.search(
            r'["\']?(?:type)["\']?\s*[:=]\s*["\']?(bubble|onomatopoeia|other)["\']?',
            seg, flags=re.IGNORECASE
        )

        if id_match and text_match:
            bid = id_match.group(1).strip()
            btext = (text_match.group(1) or text_match.group(2) or text_match.group(3) or "").strip()
            btype = type_match.group(1).strip().lower() if type_match else "bubble"
            items.append({
                "id": bid,
                "translated_text": btext,
                "type": btype
            })
    return items


def extract_translation_map(parsed_data: Any) -> Dict[str, Dict[str, Any]]:
    """Normalizes parsed JSON output into a lookup dictionary {id: {"translated_text": str, "type": str}}."""
    result = {}
    if isinstance(parsed_data, list):
        for idx, item in enumerate(parsed_data):
            if isinstance(item, dict):
                item_id = str(item.get("id", item.get("block_id", idx + 1))).strip()
                result[item_id] = {
                    "translated_text": str(item.get("translated_text", item.get("translation", ""))).strip(),
                    "type": item.get("type", "bubble"),
                    "original_text": str(item.get("original_text", item.get("text", ""))).strip()
                }
            elif isinstance(item, str):
                result[str(idx + 1)] = {
                    "translated_text": item.strip(),
                    "type": "bubble",
                    "original_text": ""
                }
    elif isinstance(parsed_data, dict):
        # Direct single-block dict handling: {"id": "...", "translated_text": "..."}
        if ("id" in parsed_data or "block_id" in parsed_data) and ("translated_text" in parsed_data or "translation" in parsed_data):
            bid = str(parsed_data.get("id", parsed_data.get("block_id", "1"))).strip()
            return {
                bid: {
                    "translated_text": str(parsed_data.get("translated_text", parsed_data.get("translation", ""))).strip(),
                    "type": parsed_data.get("type", "bubble"),
                    "original_text": str(parsed_data.get("original_text", parsed_data.get("text", ""))).strip()
                }
            }
        for key in ["translations", "blocks", "data", "results", "result", "items", "dialogues", "output", "response", "content"]:
            if key in parsed_data and isinstance(parsed_data[key], (list, dict)):
                return extract_translation_map(parsed_data[key])
        # Fallback for any key that wraps a list of block dictionaries
        for val in parsed_data.values():
            if isinstance(val, list) and val and isinstance(val[0], (dict, str)):
                return extract_translation_map(val)
        for k, v in parsed_data.items():
            if isinstance(v, dict) and ("translated_text" in v or "translation" in v):
                result[str(k)] = {
                    "translated_text": str(v.get("translated_text", v.get("translation", ""))).strip(),
                    "type": v.get("type", "bubble"),
                    "original_text": str(v.get("original_text", v.get("text", ""))).strip()
                }
            elif isinstance(v, str):
                result[str(k)] = {"translated_text": v.strip(), "type": "bubble", "original_text": ""}
    return result


def _text_similarity(s1: str, s2: str) -> float:
    """Calculates normalized text similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    s1_c = "".join(s1.lower().split())
    s2_c = "".join(s2.lower().split())
    if s1_c == s2_c:
        return 1.0
    if s1_c in s2_c or s2_c in s1_c:
        min_l = min(len(s1_c), len(s2_c))
        max_l = max(len(s1_c), len(s2_c))
        if min_l >= 3 and (min_l / max_l) >= 0.4:
            return 0.90
    return difflib.SequenceMatcher(None, s1_c, s2_c).ratio()


def align_translations_to_blocks(parsed_data: Any, blocks: List[TranslationBlock]) -> List[TranslationBlock]:
    """
    Multi-Tier Resilient Alignment Engine (五级鲁棒对齐引擎).
    Guarantees translated texts are assigned to the correct speech bubbles,
    even when LLM swaps IDs, outputs integer indices, truncates hashes, or drifts attention.
    """
    if not blocks:
        return []

    # 1. Normalize parsed_data into a list of candidate translation items
    candidates: List[Dict[str, Any]] = []

    if isinstance(parsed_data, list):
        for idx, item in enumerate(parsed_data):
            if isinstance(item, dict):
                candidates.append({
                    "raw_id": str(item.get("id", item.get("block_id", ""))).strip(),
                    "orig_text": str(item.get("original_text", item.get("text", item.get("source_text", "")))).strip(),
                    "trans_text": str(item.get("translated_text", item.get("translation", ""))).strip(),
                    "type": item.get("type", "bubble"),
                    "index": idx
                })
            elif isinstance(item, str):
                candidates.append({
                    "raw_id": "",
                    "orig_text": "",
                    "trans_text": item.strip(),
                    "type": "bubble",
                    "index": idx
                })
    elif isinstance(parsed_data, dict):
        for key in ["translations", "blocks", "data", "results", "result", "items", "dialogues", "output", "response", "content"]:
            if key in parsed_data and isinstance(parsed_data[key], list):
                return align_translations_to_blocks(parsed_data[key], blocks)
        for val in parsed_data.values():
            if isinstance(val, list) and val and isinstance(val[0], dict) and ("id" in val[0] or "translated_text" in val[0]):
                return align_translations_to_blocks(val, blocks)
        # Direct dict mapping: {id: text} or {id: {translated_text: ...}}
        for idx, (k, v) in enumerate(parsed_data.items()):
            if isinstance(v, dict):
                candidates.append({
                    "raw_id": str(k).strip(),
                    "orig_text": str(v.get("original_text", v.get("text", ""))).strip(),
                    "trans_text": str(v.get("translated_text", v.get("translation", ""))).strip(),
                    "type": v.get("type", "bubble"),
                    "index": idx
                })
            elif isinstance(v, str):
                candidates.append({
                    "raw_id": str(k).strip(),
                    "orig_text": "",
                    "trans_text": v.strip(),
                    "type": "bubble",
                    "index": idx
                })

    matched_blocks: Set[int] = set()
    matched_candidates: Set[int] = set()
    assignments: Dict[int, Dict[str, Any]] = {}  # block_idx -> candidate

    # Tier 1: Original text semantic similarity matching
    # If the LLM returned original_text, match by content first to heal any swapped IDs!
    for c_idx, cand in enumerate(candidates):
        if not cand["orig_text"] or not cand["trans_text"]:
            continue
        best_b_idx = None
        best_score = 0.0
        for b_idx, block in enumerate(blocks):
            if b_idx in matched_blocks:
                continue
            sim = _text_similarity(cand["orig_text"], block.original_text)
            if sim > best_score:
                best_score = sim
                best_b_idx = b_idx
        if best_b_idx is not None and best_score >= 0.55:
            assignments[best_b_idx] = cand
            matched_blocks.add(best_b_idx)
            matched_candidates.add(c_idx)

    # Tier 2: Exact ID match
    for c_idx, cand in enumerate(candidates):
        if c_idx in matched_candidates or not cand["raw_id"] or not cand["trans_text"]:
            continue
        for b_idx, block in enumerate(blocks):
            if b_idx in matched_blocks:
                continue
            if cand["raw_id"].lower() == block.id.lower():
                assignments[b_idx] = cand
                matched_blocks.add(b_idx)
                matched_candidates.add(c_idx)
                break

    # Tier 3: Prefix / Suffix / Hash match (e.g. 'aba5' matches 'aba5e381' or '#aba5')
    for c_idx, cand in enumerate(candidates):
        if c_idx in matched_candidates or not cand["raw_id"] or not cand["trans_text"]:
            continue
        c_clean = cand["raw_id"].lstrip("#").lower()
        if len(c_clean) < 3:
            continue
        for b_idx, block in enumerate(blocks):
            if b_idx in matched_blocks:
                continue
            b_clean = block.id.lstrip("#").lower()
            if b_clean.startswith(c_clean) or c_clean.startswith(b_clean):
                assignments[b_idx] = cand
                matched_blocks.add(b_idx)
                matched_candidates.add(c_idx)
                break

    # Tier 4: Reading Order or Sequential Index match (e.g. '1', '2', 'block_1')
    for c_idx, cand in enumerate(candidates):
        if c_idx in matched_candidates or not cand["trans_text"]:
            continue
        c_clean = cand["raw_id"].lower().replace("block_", "").replace("bubble_", "").replace("b", "")
        if c_clean.isdigit():
            k = int(c_clean)
            for b_idx, block in enumerate(blocks):
                if b_idx in matched_blocks:
                    continue
                if block.reading_order == k or (b_idx + 1) == k or b_idx == k:
                    assignments[b_idx] = cand
                    matched_blocks.add(b_idx)
                    matched_candidates.add(c_idx)
                    break

    # Tier 4b: Sequential alignment for remaining unmatched
    remaining_b_indices = [i for i in range(len(blocks)) if i not in matched_blocks]
    remaining_c_indices = [i for i in range(len(candidates)) if i not in matched_candidates and candidates[i]["trans_text"]]
    for b_idx, c_idx in zip(remaining_b_indices, remaining_c_indices):
        assignments[b_idx] = candidates[c_idx]
        matched_blocks.add(b_idx)
        matched_candidates.add(c_idx)

    # Apply assignments
    for b_idx, block in enumerate(blocks):
        if b_idx in assignments:
            cand = assignments[b_idx]
            block.translated_text = cand["trans_text"]
            if cand.get("type"):
                block.type = cand["type"]
        elif not block.translated_text:
            block.translated_text = block.original_text

    # Tier 5: Length Ratio Inversion Anomaly Detection & Self-Healing
    # Detect cases where long dialogue and short label got swapped
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            b1, b2 = blocks[i], blocks[j]
            o1, o2 = len(b1.original_text), len(b2.original_text)
            t1, t2 = len(b1.translated_text), len(b2.translated_text)
            # If b1 is much longer than b2 in original, but b2 is much longer than b1 in translation:
            if o1 >= 50 and o2 <= 25 and t1 <= 18 and t2 >= 35:
                # Anomaly detected: severe length inversion
                b1.translated_text, b2.translated_text = b2.translated_text, b1.translated_text
                b1.type, b2.type = b2.type, b1.type

    return blocks
