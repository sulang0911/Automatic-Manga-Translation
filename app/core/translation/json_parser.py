"""
app/core/translation/json_parser.py
Robust 7-stage JSON Parser for LLM manga translation outputs.
Handles thinking tags, markdown wrappers, trailing commas, truncated JSON, and syntax salvage.
"""
import json
import re
from typing import Any, List, Dict


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
        for item in parsed_data:
            if isinstance(item, dict) and "id" in item:
                result[str(item["id"])] = {
                    "translated_text": str(item.get("translated_text", "")).strip(),
                    "type": item.get("type", "bubble")
                }
    elif isinstance(parsed_data, dict):
        # Direct single-block dict handling: {"id": "...", "translated_text": "..."}
        if "id" in parsed_data and "translated_text" in parsed_data:
            return {
                str(parsed_data["id"]): {
                    "translated_text": str(parsed_data.get("translated_text", "")).strip(),
                    "type": parsed_data.get("type", "bubble")
                }
            }
        for key in ["translations", "blocks", "data", "results", "result", "items", "dialogues", "output", "response", "content"]:
            if key in parsed_data and isinstance(parsed_data[key], list):
                return extract_translation_map(parsed_data[key])
        # Fallback for any key that wraps a list of block dictionaries
        for val in parsed_data.values():
            if isinstance(val, list) and val and isinstance(val[0], dict) and ("id" in val[0] or "translated_text" in val[0]):
                return extract_translation_map(val)
        for k, v in parsed_data.items():
            if isinstance(v, dict) and "translated_text" in v:
                result[str(k)] = {
                    "translated_text": str(v.get("translated_text", "")).strip(),
                    "type": v.get("type", "bubble")
                }
            elif isinstance(v, str):
                result[str(k)] = {"translated_text": v.strip(), "type": "bubble"}
    return result
