"""LLM Enhancer — optional AI enhancement of analysis results.

Disabled by default. Enabled via --llm flag.
Supported backends: DeepSeek, OpenAI, Ollama (local).

The LLM never receives the raw image — only features, OCR text, and layout regions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class LLMEnhancerError(Exception):
    pass


def build_prompt(
    features: list,
    ocr_text: str,
    layout_labels: list,
    current_result: Dict[str, Any],
) -> str:
    """Build the prompt sent to the LLM."""
    return f"""You are a screenshot analysis assistant. Based on the extracted features below, improve the structured description of the screenshot.

## Extracted Data

**Detected features:** {', '.join(features) if features else 'none'}
**Layout regions:** {', '.join(layout_labels) if layout_labels else 'none'}
**OCR text snippet (first 500 chars):**
{ocr_text[:500] if ocr_text else '(no text detected)'}

## Current Analysis
```json
{json.dumps(current_result, indent=2)}
```

## Task
Return a JSON object with these fields only:
- "application": string — application name (e.g. "VS Code", "Chrome", "Terminal")
- "screen_type": string — snake_case type (e.g. "python_error", "login_form")
- "description": string — one clear sentence describing what is visible
- "confidence": float — 0.0 to 1.0
- "visible_elements": array of strings

Respond with raw JSON only, no markdown code fences.
"""


def enhance_with_llm(
    features: list,
    ocr_text: str,
    layout_labels: list,
    current_result: Dict[str, Any],
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the configured LLM and return an enhanced result dict.

    Backend selection order:
    1. `backend` argument
    2. SCREENKB_LLM env var (deepseek | openai | ollama)
    3. Auto-detect from available API keys
    """
    prompt = build_prompt(features, ocr_text, layout_labels, current_result)

    selected = backend or os.environ.get("SCREENKB_LLM", "").lower()
    if not selected:
        # Auto-detect
        if os.environ.get("DEEPSEEK_API_KEY"):
            selected = "deepseek"
        elif os.environ.get("OPENAI_API_KEY"):
            selected = "openai"
        else:
            selected = "ollama"

    if selected == "deepseek":
        return _call_openai_compatible(
            prompt=prompt,
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            model=os.environ.get("SCREENKB_LLM_MODEL", "deepseek-chat"),
        )
    elif selected == "openai":
        return _call_openai_compatible(
            prompt=prompt,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            model=os.environ.get("SCREENKB_LLM_MODEL", "gpt-4o-mini"),
        )
    elif selected == "ollama":
        return _call_ollama(
            prompt=prompt,
            model=os.environ.get("SCREENKB_LLM_MODEL", "llama3"),
        )
    else:
        raise LLMEnhancerError(f"Unknown LLM backend: '{selected}'. Use deepseek, openai, or ollama.")


def _call_openai_compatible(
    prompt: str,
    base_url: str,
    api_key_env: str,
    model: str,
) -> Dict[str, Any]:
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        raise LLMEnhancerError("urllib not available")

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise LLMEnhancerError(
            f"API key not found. Set the {api_key_env} environment variable."
        )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 400,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except urllib.error.HTTPError as e:
        raise LLMEnhancerError(f"HTTP {e.code}: {e.read().decode()[:200]}")
    except json.JSONDecodeError as e:
        raise LLMEnhancerError(f"LLM returned invalid JSON: {e}")


def _call_ollama(prompt: str, model: str) -> Dict[str, Any]:
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        raise LLMEnhancerError("urllib not available")

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        content = data.get("response", "")
        return json.loads(content)
    except urllib.error.URLError as e:
        raise LLMEnhancerError(
            f"Cannot reach Ollama at {base_url}. Is it running? Error: {e}"
        )
    except json.JSONDecodeError as e:
        raise LLMEnhancerError(f"Ollama returned invalid JSON: {e}")
