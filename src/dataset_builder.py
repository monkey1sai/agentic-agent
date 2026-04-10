from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def validate_teacher_trace(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_root = ["task_summary", "task_type", "assumptions", "plan", "steps", "final_output", "quality_checks"]
    for key in required_root:
        if key not in trace:
            errors.append(f"missing root field: {key}")

    if not isinstance(trace.get("steps"), list) or not trace.get("steps"):
        errors.append("steps must be a non-empty list")

    final_output = trace.get("final_output") or {}
    for key in ["answer", "artifact_summary", "stop_reason"]:
        if not str(final_output.get(key, "")).strip():
            errors.append(f"final_output.{key} is required")

    for step in trace.get("steps") or []:
        for key in [
            "step_id",
            "goal",
            "thought_summary",
            "action_type",
            "action_input",
            "expected_observation",
            "actual_observation",
            "decision",
            "produced_artifact",
        ]:
            if key not in step or not str(step.get(key, "")).strip():
                errors.append(f"step field missing or empty: {key}")
    return errors


def score_sample_quality(trace: dict[str, Any], validation_errors: list[str]) -> float:
    score = 1.0
    if validation_errors:
        score -= min(0.5, len(validation_errors) * 0.05)

    steps = trace.get("steps") or []
    if len(steps) < 2:
        score -= 0.1
    if len(steps) > 12:
        score -= 0.05

    for step in steps:
        if len(step.get("thought_summary", "")) < 8:
            score -= 0.03
        if step.get("actual_observation", "") == step.get("expected_observation", ""):
            score -= 0.02

    return max(0.0, round(score, 4))


def build_training_record(
    task: dict[str, Any],
    trace: dict[str, Any],
    raw_response: str,
    teacher_model: str,
    prompt_version: str,
    validation_errors: list[str],
    quality_score: float,
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "task")
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return {
        "sample_id": f"{task_id}::teacher::{timestamp}",
        "task": task,
        "teacher_plan": {
            "task_summary": trace["task_summary"],
            "task_type": trace["task_type"],
            "assumptions": trace["assumptions"],
            "plan": trace["plan"],
        },
        "steps": trace["steps"],
        "final_output": trace["final_output"],
        "metadata": {
            "teacher_model": teacher_model,
            "prompt_version": prompt_version,
            "quality_score": quality_score,
            "validation_errors": validation_errors,
            "raw_response": raw_response,
        },
    }
