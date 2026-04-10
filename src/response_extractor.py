from __future__ import annotations

import json
from typing import Any


def extract_json_block(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _find_balanced_json_object(stripped)
        return json.loads(candidate)


def normalize_teacher_trace(payload: dict[str, Any]) -> dict[str, Any]:
    normalized_steps = []
    for index, step in enumerate(payload.get("steps") or [], start=1):
        normalized_steps.append(
            {
                "step_id": int(step.get("step_id", index)),
                "goal": _as_text(step.get("goal")),
                "thought_summary": _as_text(step.get("thought_summary")),
                "action_type": _as_text(step.get("action_type"), default="other"),
                "action_input": _as_text(step.get("action_input")),
                "expected_observation": _as_text(step.get("expected_observation")),
                "actual_observation": _as_text(step.get("actual_observation")),
                "decision": _as_text(step.get("decision")),
                "produced_artifact": _as_text(step.get("produced_artifact")),
            }
        )

    final_output = payload.get("final_output") or {}
    normalized = {
        "task_summary": _as_text(payload.get("task_summary")),
        "task_type": _as_text(payload.get("task_type"), default="other"),
        "assumptions": _as_text_list(payload.get("assumptions")),
        "plan": _as_text_list(payload.get("plan")),
        "steps": normalized_steps,
        "final_output": {
            "answer": _as_text(final_output.get("answer")),
            "artifact_summary": _as_text(final_output.get("artifact_summary")),
            "stop_reason": _as_text(final_output.get("stop_reason")),
        },
        "quality_checks": _as_text_list(payload.get("quality_checks")),
    }
    return normalized


def _strip_code_fence(raw_text: str) -> str:
    lines = raw_text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _find_balanced_json_object(raw_text: str) -> str:
    start = raw_text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in teacher response.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw_text)):
        char = raw_text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : index + 1]
    raise ValueError("JSON object is not balanced in teacher response.")


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
