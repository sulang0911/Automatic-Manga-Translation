"""
scripts/test_antigravity_api.py
Standalone diagnostic script for testing Antigravity / Sub2API Gemini connectivity
and manga JSON translation using Python stdlib only.

Usage:
  1) With environment variables (takes precedence):
       export GOOGLE_GEMINI_BASE_URL="https://sub2api.emberanvil.dev/antigravity"
       export GEMINI_API_KEY="sk-..."
       export GEMINI_MODEL="gemini-3.8-flash"
       python scripts/test_antigravity_api.py

  2) Or automatically reading local desktop_config.json (model defaults to gemini-3.8-flash).
"""
import json
import os
import time
import urllib.error
import urllib.request


def _load_config():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "desktop_config.json",
    )
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm"), dict) else {}

    base_url = (
        os.environ.get("GOOGLE_GEMINI_BASE_URL")
        or llm_cfg.get("endpoint")
        or cfg.get("custom_endpoint")
        or ""
    ).strip().rstrip("/")
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or llm_cfg.get("api_key")
        or cfg.get("api_key")
        or ""
    ).strip()
    model = (
        os.environ.get("GEMINI_MODEL")
        or llm_cfg.get("model")
        or cfg.get("model")
        or "gemini-3.8-flash"
    ).strip()
    timeout_s = max(120.0, float(llm_cfg.get("timeout_seconds", 120.0) or 120.0))

    if not base_url.lower().endswith("/antigravity"):
        # Match /antigravity/v1beta/models/... unless /v1beta already supplied
        if not base_url.lower().endswith("/v1beta"):
            base_url = f"{base_url}/antigravity"
    return base_url, api_key, model, timeout_s


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def _extract_reply(result: dict) -> str:
    candidates = result.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    non_thought = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text") and not p.get("thought")]
    all_txt = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    return "\n".join(non_thought or all_txt).strip()


def _post(url: str, api_key: str, payload: dict, timeout_s: float, bypass_proxy: bool):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.4.0",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if bypass_proxy
        else urllib.request.build_opener()
    )
    t0 = time.perf_counter()
    with opener.open(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, time.perf_counter() - t0, json.loads(raw)


def main():
    base_url, api_key, model, timeout_s = _load_config()
    if not base_url or not api_key:
        print("[-] 缺少 BASE_URL 或 API_KEY，请设置环境变量或检查 desktop_config.json")
        return 1

    if base_url.lower().endswith("/v1beta"):
        url = f"{base_url}/models/{model}:generateContent"
    else:
        url = f"{base_url}/v1beta/models/{model}:generateContent"

    print(f"[*] Endpoint : {url}")
    print(f"[*] Model    : {model}")
    print(f"[*] API Key  : {_mask_key(api_key)}")
    print(f"[*] Timeout  : {timeout_s:.0f}s")
    print(f"[*] Env Proxy: {urllib.request.getproxies() or 'none'}")

    tests = [
        (
            "1) 连通性测试",
            {"contents": [{"role": "user", "parts": [{"text": "你好，请回复'连通成功'"}]}]},
        ),
        (
            "2) 漫画 JSON 翻译测试",
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    'Translate to Simplified Chinese and return ONLY a JSON array: '
                                    '[{"id": "b1", "translated_text": "你好！"}] for: '
                                    '[{"id": "b1", "original_text": "When will this procedure be finished?"}]'
                                )
                            }
                        ],
                    }
                ],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
            },
        ),
    ]

    for label, payload in tests:
        print(f"\n--- {label} (直连绕过本地系统代理 / 含思考延迟) ---")
        try:
            status, dt, data = _post(url, api_key, payload, timeout_s=timeout_s, bypass_proxy=True)
            print(f"[+] OK status={status}, 耗时={dt:.2f}s, modelVersion={data.get('modelVersion')}")
            print(f"[+] usageMetadata={data.get('usageMetadata')}")
            print(f"[+] reply={_extract_reply(data)}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            print(f"[-] HTTP {e.code}: {err_body[:400]}")
            return 1
        except Exception as e:
            print(f"[-] 请求失败: {type(e).__name__}: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
