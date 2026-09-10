"""
app/core/translation/json_parser.py
Robust 7-stage JSON Parser for LLM manga translation outputs.
Handles thinking tags, markdown wrappers, trailing commas, truncated JSON, and syntax salvage.
"""
import json
import re
import difflib
import unicodedata
from typing import Any, List, Dict, Optional, Set, Tuple
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
    text = re.sub(r'<(?:think|thought|reasoning)>[\s\S]*?</(?:think|thought|reasoning)>', '', raw_text, flags=re.IGNORECASE).strip()

    # Pass 2: Extract code block from markdown fences
    json_match = re.search(r'```(?:json)\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE)
    if json_match:
        text = json_match.group(1).strip()
    else:
        # Check for unclosed code fence continuing to EOF
        unclosed = re.search(r'```(?:json)?\s*([\s\S]+)$', text, flags=re.IGNORECASE)
        if unclosed and ('[' in unclosed.group(1) or '{' in unclosed.group(1)):
            text = unclosed.group(1).strip()
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

    # Strip incomplete trailing key or key-value pair, e.g. `, "key":` or `, "key"` or `,`
    repaired = re.sub(r',\s*"[^"]*"\s*(?::\s*)?$', '', repaired)
    repaired = re.sub(r'{\s*"[^"]*"\s*(?::\s*)?$', '{', repaired)
    repaired = re.sub(r',\s*$', '', repaired)

    for opener in reversed(stack):
        if opener == '{':
            repaired += '}'
        elif opener == '[':
            repaired += ']'
    return repaired


def _salvage_translation_blocks(text: str) -> List[Dict[str, Any]]:
    """Extracts id, original_text, and translated_text pairs via regex when JSON syntax is severely corrupted."""
    items = []
    # Split text into chunks at each 'id' / 'block_id' declaration
    segments = re.split(r'(?=["\']?(?:id|block_id)["\']?\s*[:=])', text, flags=re.IGNORECASE)
    for seg in segments:
        id_match = re.search(
            r'(?<![a-zA-Z0-9_])["\']?(?:id|block_id)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]+)["\']?',
            seg, flags=re.IGNORECASE
        )
        orig_match = re.search(
            r'(?<![a-zA-Z0-9_])["\']?(?:original_text|source_text|orig_text)["\']?\s*[:=]\s*(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'|([^"\'\n\r,}]+))',
            seg, flags=re.IGNORECASE
        )
        text_match = re.search(
            r'(?<![a-zA-Z0-9_])["\']?(?:translated_text|translation|target_text|text)["\']?\s*[:=]\s*(?:"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'|([^"\'\n\r,}]+))',
            seg, flags=re.IGNORECASE
        )
        type_match = re.search(
            r'(?<![a-zA-Z0-9_])["\']?(?:type)["\']?\s*[:=]\s*["\']?(bubble|onomatopoeia|other)["\']?',
            seg, flags=re.IGNORECASE
        )

        if id_match and text_match:
            bid = id_match.group(1).strip()
            btext = (text_match.group(1) or text_match.group(2) or text_match.group(3) or "").strip()
            btext = btext.replace('\\"', '"').replace("\\'", "'")
            otext = ""
            if orig_match:
                otext = (orig_match.group(1) or orig_match.group(2) or orig_match.group(3) or "").strip()
                otext = otext.replace('\\"', '"').replace("\\'", "'")
            btype = type_match.group(1).strip().lower() if type_match else "bubble"
            items.append({
                "id": bid,
                "original_text": otext,
                "translated_text": btext,
                "type": btype
            })

    # Plain text list fallback (e.g. "1. [id] translated text" or "1. translated text")
    if not items:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines:
            m = re.match(r'^(?:[-*]|\d+[\.\)])\s*(?:\[([a-zA-Z0-9_\-]+)\]|([a-zA-Z0-9_\-]+)\s*[:：])?\s*(.+)$', line)
            if m:
                lid = (m.group(1) or m.group(2) or str(len(items) + 1)).strip()
                ltxt = m.group(3).strip()
                if ltxt and not ltxt.startswith("```"):
                    items.append({
                        "id": lid,
                        "original_text": "",
                        "translated_text": ltxt,
                        "type": "bubble"
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
    s1_norm = unicodedata.normalize('NFKC', s1)
    s2_norm = unicodedata.normalize('NFKC', s2)
    s1_c = "".join(s1_norm.lower().split())
    s2_c = "".join(s2_norm.lower().split())
    if s1_c == s2_c:
        return 1.0

    # Strip common punctuation for punctuation-agnostic match
    punc_clean = str.maketrans("", "", ".,!?;:\"'「」『』()（）[]【】…〜~-—_ 、。")
    s1_p = s1_c.translate(punc_clean)
    s2_p = s2_c.translate(punc_clean)
    if s1_p and s2_p and s1_p == s2_p:
        return 0.98

    min_l = min(len(s1_c), len(s2_c))
    max_l = max(len(s1_c), len(s2_c))

    # Substring check with coverage-weighted scoring
    if s1_c in s2_c or s2_c in s1_c:
        if min_l >= 3 and (min_l / max_l) >= 0.4:
            return round(0.75 + 0.25 * (min_l / max_l), 2)

    # Early exit for drastic length discrepancies where ratio cannot reach 0.55
    if max_l > 0 and (min_l / max_l) < 0.25 and min_l < 4:
        return 0.0

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
    text_verified_blocks: Set[int] = set()
    assignments: Dict[int, Dict[str, Any]] = {}  # block_idx -> candidate

    # Tier 1: Original text semantic similarity matching with Global Maximum Weight Greedy Matching
    # Collect all candidate-block pairs and prioritize by highest similarity descending
    sim_pairs: List[Tuple[float, int, int]] = []
    for c_idx, cand in enumerate(candidates):
        if not cand["orig_text"] or not cand["trans_text"]:
            continue
        for b_idx, block in enumerate(blocks):
            sim = _text_similarity(cand["orig_text"], block.original_text)
            if sim >= 0.55:
                bonus = 0.05 if (cand["raw_id"] and cand["raw_id"].lower() == block.id.lower()) else 0.0
                sim_pairs.append((sim + bonus, c_idx, b_idx))

    sim_pairs.sort(key=lambda item: item[0], reverse=True)

    for score, c_idx, b_idx in sim_pairs:
        if c_idx not in matched_candidates and b_idx not in matched_blocks:
            assignments[b_idx] = candidates[c_idx]
            matched_blocks.add(b_idx)
            matched_candidates.add(c_idx)
            text_verified_blocks.add(b_idx)

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
    # Step 4a: Explicit reading_order matching
    for c_idx, cand in enumerate(candidates):
        if c_idx in matched_candidates or not cand["trans_text"]:
            continue
        c_clean = re.sub(r'^(?:block_|bubble_|b_?|#)', '', cand["raw_id"].strip().lower())
        if c_clean.isdigit():
            k = int(c_clean)
            for b_idx, block in enumerate(blocks):
                if b_idx in matched_blocks:
                    continue
                if block.reading_order > 0 and block.reading_order == k:
                    assignments[b_idx] = cand
                    matched_blocks.add(b_idx)
                    matched_candidates.add(c_idx)
                    break

    # Step 4b: Position-based matching (detect 0-based vs 1-based indexing to prevent off-by-one shifts)
    numeric_cands: List[Tuple[int, int]] = []
    for c_idx, cand in enumerate(candidates):
        if c_idx in matched_candidates or not cand["trans_text"]:
            continue
        c_clean = re.sub(r'^(?:block_|bubble_|b_?|#)', '', cand["raw_id"].strip().lower())
        if c_clean.isdigit():
            numeric_cands.append((c_idx, int(c_clean)))

    has_zero = any(k == 0 for _, k in numeric_cands)
    for c_idx, k in numeric_cands:
        if c_idx in matched_candidates:
            continue
        target_b_idx = (k if has_zero else k - 1)
        if 0 <= target_b_idx < len(blocks) and target_b_idx not in matched_blocks:
            assignments[target_b_idx] = candidates[c_idx]
            matched_blocks.add(target_b_idx)
            matched_candidates.add(c_idx)

    # Tier 4c: Sequential alignment for remaining unmatched candidates without explicit mismatched ID
    remaining_b_indices = [i for i in range(len(blocks)) if i not in matched_blocks]
    remaining_c_indices = [
        i for i in range(len(candidates))
        if i not in matched_candidates and candidates[i]["trans_text"] and not candidates[i]["raw_id"]
    ]
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
    # Detect cases where long dialogue and short label got swapped (bidirectional check)
    swapped_indices: Set[int] = set()
    for i in range(len(blocks)):
        if i in swapped_indices or i in text_verified_blocks:
            continue
        for j in range(i + 1, len(blocks)):
            if j in swapped_indices or j in text_verified_blocks:
                continue
            b1, b2 = blocks[i], blocks[j]
            o1, o2 = len(b1.original_text), len(b2.original_text)
            t1, t2 = len(b1.translated_text), len(b2.translated_text)
            # Check both directions (i long / j short OR i short / j long)
            is_anomaly = (
                (o1 >= 50 and o2 <= 25 and t1 <= 18 and t2 >= 35) or
                (o2 >= 50 and o1 <= 25 and t2 <= 18 and t1 >= 35)
            )
            if is_anomaly:
                b1.translated_text, b2.translated_text = b2.translated_text, b1.translated_text
                b1.type, b2.type = b2.type, b1.type
                swapped_indices.add(i)
                swapped_indices.add(j)
                break

    return blocks
