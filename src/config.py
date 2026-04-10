from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 120
    prompt_version: str = "teacher_agentic_v1"


def load_settings() -> Settings:
    base_url = os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("GEMMA_API_KEY", "")
    model = os.environ.get("GEMMA_MODEL", "gemma-4-26B-A4B-it-Q8_0.gguf")
    timeout_seconds = int(os.environ.get("GEMMA_TIMEOUT_SECONDS", "120"))
    prompt_version = os.environ.get("PROMPT_VERSION", "teacher_agentic_v1")
    return Settings(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        prompt_version=prompt_version,
    )


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
