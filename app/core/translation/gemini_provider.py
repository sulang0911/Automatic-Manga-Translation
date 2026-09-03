"""
app/core/translation/gemini_provider.py
Google Gemini Translation Provider (REST API, Vision Multimodal support).
"""
import time
import base64
import requests
from typing import List, Optional, Callable
from app.core.models import TranslationBlock, BlockType
from app.core.translation.base import (
    BaseTranslationProvider, ProviderConfig,
    TranslationContext, DiagnosticResult
)
from app.core.translation.prompt_templates import PromptTemplates
from app.core.translation.retry_handler import execute_http_request_with_retry
from app.core.translation.json_parser import parse_llm_json_response, extract_translation_map


class GeminiProvider(BaseTranslationProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        ep = (
            config.endpoint.strip().rstrip('/') if config.endpoint
            else "https://generativelanguage.googleapis.com"
        )
        if ep.endswith('/v1beta'):
            ep = ep[:-len('/v1beta')].rstrip('/')
        self.endpoint = ep
        self.model = config.model or "gemini-2.5-flash"

    def supports_vision(self) -> bool:
        return True

    def translate_text_blocks(
        self,
        blocks: List[TranslationBlock],
        source_lang: str,
        target_lang: str,
        context: Optional[TranslationContext] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[TranslationBlock]:
        if not blocks:
            return []

        if progress_callback:
            progress_callback(20, f"正在向 Gemini ({self.model}) 发送文本翻译请求...")

        sys_prompt = PromptTemplates.build_text_system_prompt(source_lang, target_lang, context)
        user_msg = PromptTemplates.build_text_user_message(blocks, source_lang, target_lang)

        # Build URL
        url = f"{self.endpoint}/v1beta/models/{self.model}:generateContent?key={self.config.api_key}"
        headers = {"Content-Type": "application/json"}

        body = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{sys_prompt}\n\n{user_msg}"}]}
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.config.temperature
            }
        }

        resp = execute_http_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_payload=body,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            provider_name="Gemini",
            progress_callback=progress_callback
        )

        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = parse_llm_json_response(raw_text)
        trans_map = extract_translation_map(parsed)

        for b in blocks:
            if b.id in trans_map:
                b.translated_text = trans_map[b.id]["translated_text"]
                b.type = trans_map[b.id]["type"]
            elif not b.translated_text:
                b.translated_text = b.original_text

        if progress_callback:
            progress_callback(100, f"Gemini 成功完成 {len(blocks)} 个气泡翻译")

        return blocks

    def translate_vision(
        self,
        image_bytes: bytes,
        target_lang: str,
        source_lang: str = "auto",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[TranslationBlock]:
        if progress_callback:
            progress_callback(30, f"正在向 Gemini ({self.model}) 发送视觉多模态识别与翻译请求...")

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        sys_prompt = PromptTemplates.build_vision_system_prompt(target_lang)

        url = f"{self.endpoint}/v1beta/models/{self.model}:generateContent?key={self.config.api_key}"
        headers = {"Content-Type": "application/json"}

        body = {
            "contents": [
                {
                    "parts": [
                        {"text": sys_prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}}
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
            }
        }

        resp = execute_http_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_payload=body,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            provider_name="Gemini Vision",
            progress_callback=progress_callback
        )

        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = parse_llm_json_response(raw_text)

        blocks_raw = parsed.get("blocks", []) if isinstance(parsed, dict) else parsed
        blocks: List[TranslationBlock] = []
        for idx, item in enumerate(blocks_raw):
            b_type = item.get("type", "bubble")
            blocks.append(TranslationBlock(
                id=f"gemini_vision_{idx}",
                original_text=item.get("original_text", ""),
                translated_text=item.get("translated_text", ""),
                xmin=float(item.get("xmin", 0.0)),
                ymin=float(item.get("ymin", 0.0)),
                xmax=float(item.get("xmax", 100.0)),
                ymax=float(item.get("ymax", 100.0)),
                text_color=item.get("text_color", "#000000"),
                bg_color=item.get("bg_color", "#FFFFFF"),
                type=b_type
            ))

        return blocks

    def test_connection(self) -> DiagnosticResult:
        start_time = time.perf_counter()
        url = f"{self.endpoint}/v1beta/models/{self.model}:generateContent?key={self.config.api_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{"parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 5}
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10.0)
            latency = (time.perf_counter() - start_time) * 1000
            if resp.status_code == 200:
                return DiagnosticResult(
                    success=True, provider="Gemini", model=self.model,
                    latency_ms=round(latency, 1), message="连接成功"
                )
            return DiagnosticResult(
                success=False, provider="Gemini", model=self.model,
                latency_ms=round(latency, 1), message=f"HTTP {resp.status_code}: {resp.text}"
            )
        except Exception as e:
            return DiagnosticResult(
                success=False, provider="Gemini", model=self.model, latency_ms=0,
                message=str(e), suggested_action="请确认 Gemini API Key 是否有效，国内环境请配置代理端点。"
            )
