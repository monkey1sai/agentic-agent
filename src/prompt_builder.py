from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROMPT_PLACEHOLDER = "{{TASK_JSON}}"


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_teacher_messages(task: dict[str, Any], prompt_template: str) -> list[dict[str, str]]:
    task_json = json.dumps(task, ensure_ascii=False, indent=2)
    user_prompt = prompt_template.replace(PROMPT_PLACEHOLDER, task_json)
    return [
        {
            "role": "system",
            "content": "你是結構化 agentic coding 教師模型，只輸出合法 JSON。",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
