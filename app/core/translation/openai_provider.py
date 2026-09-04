"""
app/core/translation/openai_provider.py
OpenAI Translation Provider (GPT-4o, GPT-4o-mini Chat & Vision Multimodal).
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
from app.core.translation.json_parser import parse_llm_json_response, extract_translation_map, align_translations_to_blocks


class OpenAIProvider(BaseTranslationProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        ep = config.endpoint.strip().rstrip('/') if config.endpoint else "https://api.openai.com/v1"
        if ep.endswith('/chat/completions'):
            ep = ep[:-len('/chat/completions')].rstrip('/')
        self.endpoint = ep
        self.model = config.model or "gpt-4o-mini"

    def supports_vision(self) -> bool:
        return any(v in self.model.lower() for v in ["gpt-4o", "vision", "gpt-4-turbo"])

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
            progress_callback(20, f"正在向 OpenAI ({self.model}) 发送文本翻译请求...")

        sys_prompt = PromptTemplates.build_text_system_prompt(source_lang, target_lang, context)
        user_msg = PromptTemplates.build_text_user_message(blocks, source_lang, target_lang)

        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg}
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature
        }

        resp = execute_http_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_payload=body,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            provider_name="OpenAI",
            progress_callback=progress_callback,
            proxies=self.resolve_proxies()
        )

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_llm_json_response(raw_text)
        blocks = align_translations_to_blocks(parsed, blocks)

        if progress_callback:
            progress_callback(100, f"OpenAI 成功完成 {len(blocks)} 个气泡翻译")

        return blocks

    def translate_vision(
        self,
        image_bytes: bytes,
        target_lang: str,
        source_lang: str = "auto",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[TranslationBlock]:
        if progress_callback:
            progress_callback(30, f"正在向 OpenAI ({self.model}) 发送多模态视觉翻译请求...")

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        sys_prompt = PromptTemplates.build_vision_system_prompt(target_lang)

        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Detect and translate all manga text to {target_lang}."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }

        resp = execute_http_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_payload=body,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            provider_name="OpenAI Vision",
            progress_callback=progress_callback,
            proxies=self.resolve_proxies()
        )

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_llm_json_response(raw_text)

        blocks_raw = parsed.get("blocks", []) if isinstance(parsed, dict) else parsed
        blocks: List[TranslationBlock] = []
        for idx, item in enumerate(blocks_raw):
            b_type = item.get("type", "bubble")
            blocks.append(TranslationBlock(
                id=f"vision_block_{idx}",
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
        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5
        }
        try:
            px = self.resolve_proxies()
            resp = requests.post(url, headers=headers, json=body, timeout=10.0,
                                 **({} if px is None else {"proxies": px}))
            latency = (time.perf_counter() - start_time) * 1000
            if resp.status_code == 200:
                return DiagnosticResult(
                    success=True, provider="OpenAI", model=self.model,
                    latency_ms=round(latency, 1), message="连接成功"
                )
            return DiagnosticResult(
                success=False, provider="OpenAI", model=self.model,
                latency_ms=round(latency, 1), message=f"HTTP {resp.status_code}: {resp.text}"
            )
        except Exception as e:
            return DiagnosticResult(
                success=False, provider="OpenAI", model=self.model, latency_ms=0,
                message=str(e), suggested_action="请检查 API Key 是否有效，或检查代理/网络连接。"
            )
