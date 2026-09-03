"""
app/core/translation/custom_provider.py
Universal OpenAI-compatible Custom Provider (Ollama, vLLM, LMStudio, OneAPI, SiliconFlow).
"""
import re
import time
import requests
from typing import List, Optional, Callable
from app.core.models import TranslationBlock
from app.core.translation.base import (
    BaseTranslationProvider, ProviderConfig,
    TranslationContext, DiagnosticResult
)
from app.core.translation.prompt_templates import PromptTemplates
from app.core.translation.retry_handler import execute_http_request_with_retry
from app.core.translation.json_parser import parse_llm_json_response, extract_translation_map


class CustomOpenAIProvider(BaseTranslationProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.endpoint = self._normalize_endpoint(config.endpoint)
        self.model = config.model or "default-model"

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        if not endpoint or not endpoint.strip():
            return "http://localhost:11434/v1/chat/completions"
        ep = endpoint.strip().rstrip('/')
        if ep.endswith("/chat/completions"):
            ep = ep[:-len("/chat/completions")].rstrip('/')
        if not re.search(r'/v\d+$', ep):
            ep = f"{ep}/v1"
        return f"{ep}/chat/completions"

    def supports_vision(self) -> bool:
        return any(v in self.model.lower() for v in ["vl", "vision", "llava", "qvq", "gpt-4o"])

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
            progress_callback(20, f"正在向自定义 API ({self.model}) 发送翻译请求...")

        sys_prompt = PromptTemplates.build_text_system_prompt(source_lang, target_lang, context)
        user_msg = PromptTemplates.build_text_user_message(blocks, source_lang, target_lang)

        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key.lower() not in ["none", ""]:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg}
            ],
            "temperature": self.config.temperature
        }

        # Attempt with response_format, auto-degrade if unsupported
        try:
            body_with_fmt = {**body, "response_format": {"type": "json_object"}}
            resp = execute_http_request_with_retry(
                method="POST",
                url=self.endpoint,
                headers=headers,
                json_payload=body_with_fmt,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
                provider_name="Custom API",
                progress_callback=progress_callback
            )
        except Exception as e:
            # If 400 Bad Request indicates response_format is unsupported, retry without it
            if "response_format" in str(e).lower() or "400" in str(e):
                resp = execute_http_request_with_retry(
                    method="POST",
                    url=self.endpoint,
                    headers=headers,
                    json_payload=body,
                    timeout=self.config.timeout_seconds,
                    max_retries=self.config.max_retries,
                    provider_name="Custom API",
                    progress_callback=progress_callback
                )
            else:
                raise e

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = parse_llm_json_response(raw_text)
        trans_map = extract_translation_map(parsed)

        for b in blocks:
            if b.id in trans_map:
                b.translated_text = trans_map[b.id]["translated_text"]
                b.type = trans_map[b.id]["type"]
            elif not b.translated_text:
                b.translated_text = b.original_text

        if progress_callback:
            progress_callback(100, f"自定义 API 成功完成 {len(blocks)} 个气泡翻译")

        return blocks

    def test_connection(self) -> DiagnosticResult:
        start_time = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key.lower() not in ["none", ""]:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5
        }
        try:
            resp = requests.post(self.endpoint, headers=headers, json=body, timeout=10.0)
            latency = (time.perf_counter() - start_time) * 1000
            if resp.status_code == 200:
                return DiagnosticResult(
                    success=True, provider="Custom", model=self.model,
                    latency_ms=round(latency, 1), message="连接成功"
                )
            return DiagnosticResult(
                success=False, provider="Custom", model=self.model,
                latency_ms=round(latency, 1), message=f"HTTP {resp.status_code}: {resp.text}"
            )
        except Exception as e:
            return DiagnosticResult(
                success=False, provider="Custom", model=self.model, latency_ms=0,
                message=str(e), suggested_action="请检查自定义端点 URL 是否可访问（如是否包含 /v1/chat/completions）。"
            )
