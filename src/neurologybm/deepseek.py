"""DeepSeek API helpers for private case evaluation runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


PRIVATE_PATH_MARKER = "DO NOT COMMIT TO GITHUB"
DEFAULT_PRIVATE_ROOT = Path("docs") / PRIVATE_PATH_MARKER
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    light_model: str = "deepseek-chat"
    pro_model: str = "deepseek-reasoner"
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            light_model=os.getenv("DEEPSEEK_LIGHT_MODEL", "deepseek-chat"),
            pro_model=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-reasoner"),
            timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")),
        )

    def model_for_tier(self, tier: str) -> str:
        if tier == "light":
            return self.light_model
        if tier == "pro":
            return self.pro_model
        raise ValueError(f"Unsupported model tier: {tier}")


class DeepSeekClient:
    """Minimal OpenAI-compatible chat completions client."""

    def __init__(self, config: DeepSeekConfig | None = None) -> None:
        self.config = config or DeepSeekConfig.from_env()

    def chat_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.config.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for non-dry-run DeepSeek calls.")

        payload = {
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if extra_body:
            provider_body = dict(extra_body)
            if provider_body.pop("_disable_response_format", False):
                payload.pop("response_format", None)
            payload.update(provider_body)
        endpoint = self.config.base_url.rstrip("/") + "/v1/chat/completions"
        started = time.time()
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else ""
            code = exc.response.status_code if exc.response is not None else "unknown"
            raise ValueError(f"DeepSeek API HTTP {code}: {body}") from exc
        except requests.RequestException as exc:
            raise ValueError(f"DeepSeek API request failed: {exc}") from exc

        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed_content = parse_json_content(content)
        return {
            "model": model,
            "elapsed_seconds": round(time.time() - started, 3),
            "request": redact_request_payload(payload),
            "response": response_json,
            "parsed_content": parsed_content,
        }


def assert_private_path(path: Path) -> None:
    parts = set(path.resolve().parts)
    if PRIVATE_PATH_MARKER not in parts:
        raise ValueError(f"Refusing to write case text/API artifacts outside private path: {path}")


def redact_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    redacted["messages"] = [
        {
            "role": message.get("role"),
            "content": f"<redacted {len(str(message.get('content', '')))} chars>",
        }
        for message in payload.get("messages", [])
    ]
    return redacted


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = _parse_json_object_from_text(content)
    return parsed if isinstance(parsed, dict) else {"raw_content": content}


def _parse_json_object_from_text(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {"raw_content": content}
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
            return parsed if isinstance(parsed, dict) else {"raw_content": content}
        except json.JSONDecodeError:
            pass
    return {"raw_content": content}
