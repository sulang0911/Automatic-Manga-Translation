"""
app/core/translation/retry_handler.py
Robust exponential backoff retry handler with HTTP 429 rate-limit and quota exhaustion detection.
"""
import time
import random
import requests
from typing import Dict, Any, Optional, Callable
from app.core.translation.base import (
    RateLimitError, QuotaExhaustedError, AuthenticationError,
    ModelNotFoundError, TranslationError
)


def execute_http_request_with_retry(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    provider_name: str = "generic",
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> requests.Response:
    """
    Executes HTTP request with exponential backoff and jitter.
    Distinguishes immediately fatal errors (quota exhaustion, invalid auth) from transient retries.
    """
    attempt = 0
    while True:
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json_payload,
                timeout=timeout
            )

            # Success
            if resp.status_code == 200:
                return resp

            # 401 / 403: Fatal Authentication Failure
            if resp.status_code in [401, 403]:
                raise AuthenticationError(
                    f"HTTP {resp.status_code}: {resp.text}",
                    provider=provider_name
                )

            # 404: Model Not Found or Invalid Endpoint
            if resp.status_code == 404:
                raise ModelNotFoundError(
                    f"HTTP 404: {resp.text}",
                    provider=provider_name
                )

            # 429: Rate Limit OR Quota Exhaustion
            if resp.status_code == 429:
                err_lower = resp.text.lower()
                # Check for fatal quota depletion markers
                if any(k in err_lower for k in [
                    "insufficient_quota", "insufficient_balance", "quota_exceeded",
                    "resource_exhausted", "billing_not_active", "credit_exhausted", "欠费", "余额不足"
                ]):
                    raise QuotaExhaustedError(
                        f"账户配额/余额耗尽 (HTTP 429): {resp.text}",
                        provider=provider_name
                    )

                # Transient rate limit: parse Retry-After header if provided
                retry_after_hdr = resp.headers.get("Retry-After")
                retry_seconds = float(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else None

                if attempt >= max_retries:
                    raise RateLimitError(
                        f"超过最大重试次数 (HTTP 429): {resp.text}",
                        provider=provider_name,
                        retry_after=retry_seconds
                    )

                sleep_time = retry_seconds if retry_seconds else (
                    initial_delay * (backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                )
                attempt += 1
                if progress_callback:
                    progress_callback(
                        25, f"触发请求频率限制 (429)，等待 {sleep_time:.1f} 秒后进行第 {attempt}/{max_retries} 次重试..."
                    )
                time.sleep(sleep_time)
                continue

            # 5xx Server Errors: Transient Glitch
            if 500 <= resp.status_code < 600:
                if attempt >= max_retries:
                    raise TranslationError(
                        f"服务商服务器错误 (HTTP {resp.status_code}): {resp.text}",
                        provider=provider_name,
                        status_code=resp.status_code,
                        retryable=False
                    )
                sleep_time = initial_delay * (backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                attempt += 1
                if progress_callback:
                    progress_callback(
                        25, f"服务商响应异常 (HTTP {resp.status_code})，等待 {sleep_time:.1f} 秒重试..."
                    )
                time.sleep(sleep_time)
                continue

            # Other 4xx Errors (e.g. 400 Bad Request)
            raise TranslationError(
                f"请求错误 (HTTP {resp.status_code}): {resp.text}",
                provider=provider_name,
                status_code=resp.status_code
            )

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
            if attempt >= max_retries:
                raise TranslationError(
                    f"网络连接超时或失败: {str(net_err)}",
                    provider=provider_name,
                    retryable=False,
                    suggested_action="请检查本地网络连接、代理端点或防火墙设置。"
                )
            sleep_time = initial_delay * (backoff_factor ** attempt) + random.uniform(0.2, 0.6)
            attempt += 1
            if progress_callback:
                progress_callback(
                    25, f"网络波动 ({type(net_err).__name__})，正在进行第 {attempt}/{max_retries} 次重试..."
                )
            time.sleep(sleep_time)
