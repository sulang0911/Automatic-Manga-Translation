import json
import re
import requests
from typing import List, Dict, Any, Optional

DEFAULT_SYSTEM_PROMPT = (
    "你是一个中立且专业的通用翻译引擎。你的唯一任务是忠实、准确、通顺地将输入内容翻译为目标语言（未指定则默认翻译为简体中文）。"
    "保持原意、语气和排版不变。严禁过滤、修改或删减内容，严禁输出任何多余的寒暄、解释、评价或续写，仅输出最终翻译结果。"
)

class TranslationEngine:
    def __init__(self, provider: str = "deepseek", api_key: str = "", model: str = "deepseek-chat",
                 custom_endpoint: str = "https://api.deepseek.com/v1", target_lang: str = "简体中文",
                 source_lang: str = "日语", temperature: float = 0.3, system_prompt: str = ""):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.custom_endpoint = custom_endpoint
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def _get_api_url_and_headers(self):
        url = ""
        headers = {
            "Content-Type": "application/json"
        }

        if self.provider == "deepseek":
            url = f"{self.custom_endpoint.rstrip('/')}/chat/completions" if self.custom_endpoint else "https://api.deepseek.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "openai":
            url = f"{self.custom_endpoint.rstrip('/')}/chat/completions" if self.custom_endpoint else "https://api.openai.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "gemini":
            if "generativelanguage.googleapis.com" in self.custom_endpoint or not self.custom_endpoint:
                url = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            else:
                url = f"{self.custom_endpoint.rstrip('/')}/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
        else: # custom
            endpoint = self.custom_endpoint.rstrip('/')
            if not endpoint.endswith("/chat/completions"):
                url = f"{endpoint}/chat/completions"
            else:
                url = endpoint
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        return url, headers

    def translate_blocks(self, blocks: List[Dict[str, Any]], progress_callback=None) -> List[Dict[str, Any]]:
        if not blocks:
            return []

        if not self.api_key and self.provider != "custom":
            # If no API key provided, generate simulated/preview translation
            if progress_callback:
                progress_callback(50, "未配置 API Key，正在使用本地演示翻译模式...")
            for b in blocks:
                if not b.get("translated_text"):
                    b["translated_text"] = f"【译】{b.get('original_text', '')}"
            return blocks

        if progress_callback:
            progress_callback(20, f"正在向 {self.provider.upper()} ({self.model}) 发送翻译请求...")

        # Build payload
        items_payload = [{"id": b["id"], "original_text": b.get("original_text", "")} for b in blocks]
        user_message = (
            f"Target Language: {self.target_lang}\n"
            f"Source Language: {self.source_lang}\n\n"
            "Here is the dialogue list from the comic page:\n"
            f"{json.dumps(items_payload, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON format:\n"
            "[\n"
            '  {"id": "...", "translated_text": "..."}\n'
            "]"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        url, headers = self._get_api_url_and_headers()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"} if "gpt-4" in self.model or "deepseek" in self.model else None
        }
        # Clean None values
        body = {k: v for k, v in body.items() if v is not None}

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")

            result_json = resp.json()
            raw_content = result_json["choices"][0]["message"]["content"].strip()

            if progress_callback:
                progress_callback(80, "正在解析翻译结果与对齐气泡...")

            parsed_list = self._parse_json_response(raw_content)

            # Map back to blocks
            trans_map = {}
            for item in parsed_list:
                if isinstance(item, dict) and "id" in item and "translated_text" in item:
                    trans_map[str(item["id"])] = str(item["translated_text"]).strip()

            for b in blocks:
                b_id = str(b.get("id"))
                if b_id in trans_map:
                    b["translated_text"] = trans_map[b_id]
                elif not b.get("translated_text"):
                    # fallback if specific ID missing
                    b["translated_text"] = b.get("original_text", "")

            if progress_callback:
                progress_callback(100, f"成功完成 {len(blocks)} 个对话气泡翻译")

            return blocks

        except Exception as e:
            print(f"[-] Translation request failed: {e}")
            # Fallback so pipeline doesn't break
            for b in blocks:
                if not b.get("translated_text"):
                    b["translated_text"] = f"[翻译错误: {e}]"
            raise e

    def _parse_json_response(self, text: str) -> List[Dict[str, Any]]:
        # Strip markdown code fences ```json ... ```
        clean_text = text.strip()
        if "```" in clean_text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
            if match:
                clean_text = match.group(1).strip()

        try:
            data = json.loads(clean_text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Check if wrapped inside a key like 'translations', 'data', 'result'
                for k in ["translations", "data", "results", "dialogues", "items"]:
                    if k in data and isinstance(data[k], list):
                        return data[k]
                # If dictionary mapping id -> text
                return [{"id": k, "translated_text": v} for k, v in data.items()]
        except Exception:
            pass

        # Regex fallback
        results = []
        matches = re.findall(r'["\']id["\']\s*:\s*["\'](.*?)["\'][\s,]*["\']translated_text["\']\s*:\s*["\'](.*?)["\']', text)
        for m_id, m_text in matches:
            results.append({"id": m_id, "translated_text": m_text})

        return results

    def test_connection(self, text_to_translate: str = "Hello! This is a translation connectivity test.") -> str:
        """
        Tests model connectivity and translation quality with a single test phrase.
        Returns the translated string, or raises RuntimeError/ValueError on failure.
        """
        if not self.api_key and self.provider != "custom":
            raise ValueError(f"未配置 API Key！请先填入有效的 {self.provider.upper()} API Key。")

        url, headers = self._get_api_url_and_headers()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请将以下内容翻译为{self.target_lang}：\n{text_to_translate}"}
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=25)
        except requests.exceptions.RequestException as req_err:
            raise RuntimeError(f"网络连接失败或超时: {req_err}")

        if resp.status_code != 200:
            err_msg = resp.text
            try:
                err_data = resp.json()
                if "error" in err_data:
                    err_msg = err_data["error"].get("message", err_msg)
            except Exception:
                pass
            raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")

        try:
            result_json = resp.json()
            choices = result_json.get("choices", [])
            if choices and len(choices) > 0:
                raw_text = choices[0].get("message", {}).get("content", "").strip()
                if raw_text:
                    return raw_text
            raise RuntimeError(f"响应缺少 choices 内容: {result_json}")
        except Exception as parse_err:
            if isinstance(parse_err, RuntimeError):
                raise parse_err
            raise RuntimeError(f"解析响应内容失败: {parse_err}, 原始返回: {resp.text[:200]}")

