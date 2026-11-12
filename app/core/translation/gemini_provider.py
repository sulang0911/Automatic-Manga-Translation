"""
app/core/translation/gemini_provider.py
Google Gemini Translation Provider (REST API, Vision Multimodal support, with automatic Antigravity proxy fallback).
"""
import time
import base64
import requests
from typing import List, Optional, Callable
from app.core.models import TranslationBlock, BlockType
from app.core.translation.base import (
    BaseTranslationProvider, ProviderConfig,
    TranslationContext, DiagnosticResult, TranslationError
)
from app.core.translation.prompt_templates import PromptTemplates
from app.core.translation.retry_handler import execute_http_request_with_retry
from app.core.translation.json_parser import parse_llm_json_response, extract_translation_map, align_translations_to_blocks
from app.core.translation.antigravity_provider import AntigravityGeminiProvider, extract_gemini_text_from_candidates


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

    def _is_custom_proxy(self) -> bool:
        return bool(self.config.endpoint and "generativelanguage.googleapis.com" not in self.config.endpoint.lower())

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

        # If endpoint explicitly references antigravity or /v1, delegate directly to AntigravityGeminiProvider
        raw_ep_lower = (self.config.endpoint or "").strip().rstrip("/").lower()
        if "/antigravity" in raw_ep_lower or raw_ep_lower.endswith("/v1") or raw_ep_lower.endswith("/chat/completions"):
            return AntigravityGeminiProvider(self.config).translate_text_blocks(
                blocks, source_lang, target_lang, context=context, progress_callback=progress_callback
            )

        if progress_callback:
            progress_callback(20, f"正在向 Gemini ({self.model}) 发送文本翻译请求...")

        sys_prompt = PromptTemplates.build_text_system_prompt(source_lang, target_lang, context)
        user_msg = PromptTemplates.build_text_user_message(blocks, source_lang, target_lang)

        # Build URL
        url = f"{self.endpoint}/v1beta/models/{self.model}:generateContent?key={self.config.api_key}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["x-goog-api-key"] = self.config.api_key

        body = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{sys_prompt}\n\n{user_msg}"}]}
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.config.temperature
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        try:
            resp = execute_http_request_with_retry(
                method="POST",
                url=url,
                headers=headers,
                json_payload=body,
                timeout=self.config.timeout_seconds,
                max_retries=1 if self._is_custom_proxy() else self.config.max_retries,
                provider_name="Gemini",
                progress_callback=progress_callback,
                proxies=self.resolve_proxies()
            )
        except Exception as e:
            err_str = str(e).lower()
            if self._is_custom_proxy() or "platform is not gemini" in err_str or "antigravity" in err_str or "400" in err_str or "404" in err_str:
                if progress_callback:
                    progress_callback(25, "检测到反代兼容端点，正在自动切换 Antigravity Gemini 兼容通道...")
                return AntigravityGeminiProvider(self.config).translate_text_blocks(
                    blocks, source_lang, target_lang, context=context, progress_callback=progress_callback
                )
            raise e

        data = resp.json()
        candidates = data.get("candidates", [])
        raw_text = extract_gemini_text_from_candidates(candidates)
        from app.core.translation.json_parser import is_llm_refusal
        if not raw_text or is_llm_refusal(raw_text):
            return AntigravityGeminiProvider(self.config).translate_text_blocks(
                blocks, source_lang, target_lang, context=context, progress_callback=progress_callback
            )
        try:
            parsed = parse_llm_json_response(raw_text)
            blocks = align_translations_to_blocks(parsed, blocks)
        except Exception:
            return AntigravityGeminiProvider(self.config).translate_text_blocks(
                blocks, source_lang, target_lang, context=context, progress_callback=progress_callback
            )

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
        raw_ep_lower = (self.config.endpoint or "").strip().rstrip("/").lower()
        if "/antigravity" in raw_ep_lower or raw_ep_lower.endswith("/v1") or raw_ep_lower.endswith("/chat/completions"):
            return AntigravityGeminiProvider(self.config).translate_vision(
                image_bytes, target_lang, source_lang=source_lang, progress_callback=progress_callback
            )

        if progress_callback:
            progress_callback(30, f"正在向 Gemini ({self.model}) 发送视觉多模态识别与翻译请求...")

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        sys_prompt = PromptTemplates.build_vision_system_prompt(target_lang)

        url = f"{self.endpoint}/v1beta/models/{self.model}:generateContent?key={self.config.api_key}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["x-goog-api-key"] = self.config.api_key

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
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        try:
            resp = execute_http_request_with_retry(
                method="POST",
                url=url,
                headers=headers,
                json_payload=body,
                timeout=self.config.timeout_seconds,
                max_retries=1 if self._is_custom_proxy() else self.config.max_retries,
                provider_name="Gemini Vision",
                progress_callback=progress_callback,
                proxies=self.resolve_proxies()
            )
        except Exception as e:
            err_str = str(e).lower()
            if self._is_custom_proxy() or "platform is not gemini" in err_str or "antigravity" in err_str or "400" in err_str or "404" in err_str:
                return AntigravityGeminiProvider(self.config).translate_vision(
                    image_bytes, target_lang, source_lang=source_lang, progress_callback=progress_callback
                )
            raise e

        data = resp.json()
        candidates = data.get("candidates", [])
        raw_text = extract_gemini_text_from_candidates(candidates)
        if not raw_text:
            finish_reason = candidates[0].get("finishReason", "UNKNOWN") if candidates else "NO_CANDIDATE"
            raise TranslationError(
                f"Gemini Vision 未返回有效的识别内容 (finishReason: {finish_reason})",
                provider="Gemini Vision",
                suggested_action="图像可能被安全审核拦截，建议切换为文字模式或更换模型。"
            )
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
        if self.config.api_key:
            headers["x-goog-api-key"] = self.config.api_key
        body = {
            "contents": [{"parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 5}
        }
        try:
            px = self.resolve_proxies()
            resp = requests.post(url, headers=headers, json=body, timeout=10.0,
                                 **({} if px is None else {"proxies": px}))
            latency = (time.perf_counter() - start_time) * 1000
            if resp.status_code == 200:
                return DiagnosticResult(
                    success=True, provider="Gemini", model=self.model,
                    latency_ms=round(latency, 1), message="连接成功"
                )
            if self._is_custom_proxy() or "platform is not gemini" in resp.text.lower():
                return AntigravityGeminiProvider(self.config).test_connection()
            return DiagnosticResult(
                success=False, provider="Gemini", model=self.model,
                latency_ms=round(latency, 1), message=f"HTTP {resp.status_code}: {resp.text}"
            )
        except Exception as e:
            if self._is_custom_proxy():
                return AntigravityGeminiProvider(self.config).test_connection()
            return DiagnosticResult(
                success=False, provider="Gemini", model=self.model, latency_ms=0,
                message=str(e), suggested_action="请确认 Gemini API Key 是否有效，国内环境请配置代理端点。"
            )
