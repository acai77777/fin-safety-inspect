"""One-shot DeepSeek connectivity check. Run with: python test_relay.py"""

import os

import httpx

key = os.environ.get("OPENAI_API_KEY", "")
base = os.environ.get("OPENAI_BASE_URL", "")

print(f"KEY:  {key[:10]}... (len={len(key)})")
print(f"URL:  {base}")
print()

if not key or not base:
    raise SystemExit("Missing OPENAI_API_KEY or OPENAI_BASE_URL")

url = base.rstrip("/") + "/chat/completions"
print(f"POST  {url}")

r = httpx.post(
    url,
    headers={"Authorization": f"Bearer {key}"},
    json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "say hi in one word"}],
        "max_tokens": 10,
    },
    timeout=30,
)

print(f"STATUS: {r.status_code}")
print(f"CT:     {r.headers.get('content-type', '')}")
print(f"BODY:   {r.text[:400]}")
