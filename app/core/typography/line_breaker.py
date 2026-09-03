"""
app/core/typography/line_breaker.py
CJK Kinsoku Shori (JIS X 4051) Line-Breaking Rules and Western Word-Wrapping.
"""
import re
from typing import List, Protocol, Set


# JIS X 4051 Characters forbidden from beginning a line (Gyōtō Kinsoku / 行頭禁则)
GYOTO_KINSOKU: Set[str] = set(
    ")]}⟩》」』】〕’”\"'>）］｝〉》」』】〕’”）｠"
    "、。，．・：；？！‼⁇⁈⁉"
    "ー〜…‥ヽヾゝゞ々〻"
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ"
    "°′″℃％‰¢"
)

# JIS X 4051 Characters forbidden from ending a line (Gyōmatsu Kinsoku / 行末禁则)
GYOMATSU_KINSOKU: Set[str] = set(
    "([{⟨《「『【〔‘“\"'<（［｛〈《「『【〔‘“（｟"
    "￥＄￡€№§©®"
)


class TextWidthMeasurer(Protocol):
    """Protocol for querying text advance width in pixels."""
    def measure_width(self, text: str, font_size: float) -> float:
        ...


class DefaultCharWidthMeasurer:
    """Fallback proportional width measurer when font rendering context is not initialized."""
    def measure_width(self, text: str, font_size: float) -> float:
        width = 0.0
        for ch in text:
            # CJK characters are typically 1.0 em wide; Latin characters ~0.55 em
            if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', ch):
                width += font_size * 1.0
            elif ch in ".,'!;: ":
                width += font_size * 0.3
            elif ch in "mwMW":
                width += font_size * 0.85
            else:
                width += font_size * 0.55
        return width


class LineBreaker:
    """
    Manga dialogue line breaker supporting CJK character-level wrapping
    with Kinsoku Shori lookahead/rollback, and Western whitespace word-wrapping.
    """

    TOKEN_PATTERN = re.compile(
        r'[a-zA-Z0-9_\-]+(?:\'[a-zA-Z0-9_\-]+)*'  # Latin words & numbers
        r'|\s+'                                   # Whitespace
        r'|[\u4e00-\u9fff]'                       # CJK Unified Ideographs
        r'|[\u3040-\u309f]'                       # Hiragana
        r'|[\u30a0-\u30ff]'                       # Katakana
        r'|[\uac00-\ud7af]'                       # Hangul
        r'|[^\w\s]'                               # Punctuation & symbols
    )

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        tokens: List[str] = []
        pos = 0
        for match in cls.TOKEN_PATTERN.finditer(text):
            if match.start() > pos:
                tokens.extend(list(text[pos:match.start()]))
            tokens.append(match.group(0))
            pos = match.end()
        if pos < len(text):
            tokens.extend(list(text[pos:]))
        return [t for t in tokens if t]

    @classmethod
    def is_cjk(cls, text: str) -> bool:
        return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

    def wrap_text(
        self,
        text: str,
        max_width: float,
        font_size: float,
        measurer: TextWidthMeasurer
    ) -> List[str]:
        paragraphs = text.split("\n")
        all_lines: List[str] = []

        for p in paragraphs:
            trimmed = p.strip()
            if not trimmed:
                continue
            tokens = self.tokenize(trimmed)
            p_lines = self._wrap_paragraph_tokens(tokens, max_width, font_size, measurer)
            all_lines.extend(p_lines)

        return all_lines if all_lines else [text]

    def _wrap_paragraph_tokens(
        self,
        tokens: List[str],
        max_width: float,
        font_size: float,
        measurer: TextWidthMeasurer
    ) -> List[str]:
        lines: List[str] = []
        current_tokens: List[str] = []

        for token in tokens:
            test_tokens = current_tokens + [token]
            test_str = "".join(test_tokens).strip()
            test_w = measurer.measure_width(test_str, font_size)

            if test_w <= max_width or not current_tokens:
                current_tokens.append(token)
            else:
                # Line would overflow. Apply Kinsoku Shori rules before breaking:
                # 1. Gyōtō Kinsoku: If candidate token starts with a forbidden character
                if token and token[0] in GYOTO_KINSOKU:
                    # Oidashi (Push out): pull preceding token down to start next line
                    if len(current_tokens) > 1:
                        popped = current_tokens.pop()
                        lines.append("".join(current_tokens).strip())
                        current_tokens = [popped, token]
                        continue
                    else:
                        # Only 1 token exists; squeeze in (Oikomi)
                        current_tokens.append(token)
                        lines.append("".join(current_tokens).strip())
                        current_tokens = []
                        continue

                # 2. Gyōmatsu Kinsoku: If last token in current line ends with a forbidden character
                if current_tokens and current_tokens[-1][-1] in GYOMATSU_KINSOKU:
                    popped = current_tokens.pop()
                    if current_tokens:
                        lines.append("".join(current_tokens).strip())
                    current_tokens = [popped, token]
                    continue

                # Standard break
                lines.append("".join(current_tokens).strip())
                if token.strip():
                    current_tokens = [token]
                else:
                    current_tokens = []

        if current_tokens:
            final_str = "".join(current_tokens).strip()
            if final_str:
                lines.append(final_str)

        return lines
