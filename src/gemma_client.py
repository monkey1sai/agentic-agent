from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from config import Settings


class GemmaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        if not self._settings.api_key:
            raise ValueError("GEMMA_API_KEY is required to call the teacher model.")

        payload = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self._settings.base_url}/chat/completions"
        http_request = request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._settings.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self._settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemma API HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Gemma API connection failed: {exc}") from exc

    @staticmethod
    def extract_content(response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("Gemma response has no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Gemma response content is empty.")
        return content
