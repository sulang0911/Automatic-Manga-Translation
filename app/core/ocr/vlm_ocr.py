"""
app/core/ocr/vlm_ocr.py
Optional multimodal Vision-LLM high-precision crop recognizer fallback.
Reuses configured LLM provider credentials, endpoint normalization, and proxy settings.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


def _encode_crop_jpeg_b64(crop_bgr: np.ndarray) -> Optional[str]:
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return None
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _clean_vlm_response(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")
        ).strip()
    if cleaned.upper() in {"NONE", "EMPTY", "NO TEXT", "N/A"}:
        return ""
    return cleaned


def is_vlm_ocr_configured(llm_cfg: Any) -> bool:
    """Checks whether an LLM config has enough connection details for vision fallback."""
    if llm_cfg is None:
        return False
    provider = str(getattr(llm_cfg, "provider", "") or "").lower()
    api_key = str(getattr(llm_cfg, "api_key", "") or "").strip()
    endpoint = str(getattr(llm_cfg, "endpoint", "") or "").strip()
    return bool(api_key or (provider == "custom" and endpoint))


def recognize_crop_with_vlm(
    crop_bgr: np.ndarray,
    llm_cfg: Any,
    source_lang: str = "auto",
    timeout_seconds: float = 15.0,
) -> Tuple[str, float]:
    """
    Performs verbatim OCR on a single bubble crop using the user's configured Vision LLM.
    Returns (text, confidence) and fails softly with ("", 0.0) if offline/unsupported.
    """
    if not is_vlm_ocr_configured(llm_cfg):
        return "", 0.0
    b64 = _encode_crop_jpeg_b64(crop_bgr)
    if not b64:
        return "", 0.0

    lang_hint = f" (language hint: {source_lang})" if source_lang and source_lang != "auto" else ""
    prompt = (
        f"Transcribe ONLY the exact text printed in this comic speech bubble crop{lang_hint}. "
        "Preserve exact spelling, punctuation, quotes, and any speaker prefix like (Name). "
        "Do NOT translate, explain, or add formatting. If no text is legible, reply empty."
    )

    try:
        from app.core.translation.base import ProviderConfig
        from app.core.translation.gemini_provider import GeminiProvider, extract_gemini_text_from_candidates
        from app.core.translation.openai_provider import OpenAIProvider

        p_cfg = ProviderConfig(
            provider_name=str(getattr(llm_cfg, "provider", "openai") or "openai"),
            api_key=str(getattr(llm_cfg, "api_key", "") or ""),
            model=str(getattr(llm_cfg, "model", "") or ""),
            endpoint=str(getattr(llm_cfg, "endpoint", "") or ""),
            timeout_seconds=timeout_seconds,
            proxy_url=str(getattr(llm_cfg, "proxy_url", "") or ""),
        )

        if p_cfg.provider_name.lower() == "gemini":
            raw_ep_lower = p_cfg.endpoint.strip().rstrip("/").lower()
            is_openai_compat_proxy = (
                "/antigravity" not in raw_ep_lower
                and (raw_ep_lower.endswith("/v1") or raw_ep_lower.endswith("/chat/completions"))
            )
            if not is_openai_compat_proxy:
                helper = GeminiProvider(p_cfg)
                if helper._is_custom_proxy():
                    url = f"{helper.endpoint}/v1beta/models/{helper.model}:generateContent"
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {p_cfg.api_key}"}
                else:
                    url = f"{helper.endpoint}/v1beta/models/{helper.model}:generateContent?key={p_cfg.api_key}"
                    headers = {"Content-Type": "application/json"}
                    if p_cfg.api_key:
                        headers["x-goog-api-key"] = p_cfg.api_key

                body = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]
                    }],
                    "generationConfig": {"temperature": 0.0},
                }
                px = helper.resolve_proxies()
                resp = requests.post(
                    url, headers=headers, json=body, timeout=timeout_seconds,
                    **({} if px is None else {"proxies": px})
                )
                if resp.status_code != 200:
                    logger.warning("VLM OCR Gemini HTTP %s: %s", resp.status_code, resp.text[:200])
                    return "", 0.0
                raw = extract_gemini_text_from_candidates(resp.json().get("candidates", []))
                cleaned = _clean_vlm_response(raw)
                return (cleaned, 0.92) if cleaned else ("", 0.0)

        if p_cfg.provider_name.lower() == "custom":
            from app.core.translation.custom_provider import CustomOpenAIProvider
            helper = CustomOpenAIProvider(p_cfg)
            url = helper.endpoint
        else:
            helper = OpenAIProvider(p_cfg)
            url = f"{helper.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if p_cfg.api_key:
            headers["Authorization"] = f"Bearer {p_cfg.api_key}"
        body = {
            "model": helper.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        px = helper.resolve_proxies()
        resp = requests.post(
            url, headers=headers, json=body, timeout=timeout_seconds,
            **({} if px is None else {"proxies": px})
        )
        if resp.status_code != 200:
            logger.warning("VLM OCR OpenAI HTTP %s: %s", resp.status_code, resp.text[:200])
            return "", 0.0
        choices = resp.json().get("choices") or []
        raw = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        cleaned = _clean_vlm_response(raw)
        return (cleaned, 0.92) if cleaned else ("", 0.0)
    except Exception as e:
        logger.warning("VLM OCR fallback failed cleanly: %s", e)
        return "", 0.0
