"""
app/core/translation/deepseek_provider.py
DeepSeek Translation Provider (DeepSeek-V3 and DeepSeek-R1).
"""
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


class DeepSeekProvider(BaseTranslationProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        ep = config.endpoint.strip().rstrip('/') if config.endpoint else "https://api.deepseek.com/v1"
        if ep.endswith('/chat/completions'):
            ep = ep[:-len('/chat/completions')].rstrip('/')
        self.endpoint = ep
        self.model = config.model or "deepseek-chat"

    def supports_vision(self) -> bool:
        return False

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
            progress_callback(20, f"正在向 DeepSeek ({self.model}) 发送翻译请求...")

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
            "temperature": self.config.temperature
        }
        # DeepSeek-V3 supports json_object; DeepSeek-R1 reasoner does not
        if "reasoner" not in self.model:
            body["response_format"] = {"type": "json_object"}

        resp = execute_http_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_payload=body,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            provider_name="DeepSeek",
            progress_callback=progress_callback
        )

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]

        if progress_callback:
            progress_callback(80, "正在解析 DeepSeek 翻译结果...")

        parsed = parse_llm_json_response(raw_text)
        trans_map = extract_translation_map(parsed)

        for b in blocks:
            if b.id in trans_map:
                b.translated_text = trans_map[b.id]["translated_text"]
                b.type = trans_map[b.id]["type"]
            elif not b.translated_text:
                b.translated_text = b.original_text

        if progress_callback:
            progress_callback(100, f"DeepSeek 成功完成 {len(blocks)} 个气泡翻译")

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
            resp = requests.post(url, headers=headers, json=body, timeout=10.0)
            latency = (time.perf_counter() - start_time) * 1000
            if resp.status_code == 200:
                return DiagnosticResult(
                    success=True, provider="DeepSeek", model=self.model,
                    latency_ms=round(latency, 1), message="连接成功"
                )
            return DiagnosticResult(
                success=False, provider="DeepSeek", model=self.model,
                latency_ms=round(latency, 1), message=f"HTTP {resp.status_code}: {resp.text}"
            )
        except Exception as e:
            return DiagnosticResult(
                success=False, provider="DeepSeek", model=self.model, latency_ms=0,
                message=str(e), suggested_action="请检查网络或 DeepSeek API 基础端点是否可用"
            )
