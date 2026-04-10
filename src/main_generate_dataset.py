from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import load_settings, project_root
from dataset_builder import build_training_record, score_sample_quality, validate_teacher_trace
from gemma_client import GemmaClient
from prompt_builder import build_teacher_messages, load_prompt_template
from response_extractor import extract_json_block, normalize_teacher_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate agentic training data from Gemma teacher outputs.")
    parser.add_argument("--tasks", required=True, help="Path to a JSON or JSONL task file.")
    parser.add_argument("--output", default="data/processed/agentic_train.jsonl", help="Processed output JSONL path.")
    parser.add_argument("--raw-output", default="data/raw/teacher_generations.jsonl", help="Raw teacher response JSONL path.")
    parser.add_argument("--prompt", default="prompts/teacher_agentic.md", help="Prompt template path.")
    parser.add_argument("--response-file", help="Optional mock teacher response file for local validation.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first request payload without calling the API.")
    return parser.parse_args()


def load_tasks(tasks_path: Path) -> list[dict[str, Any]]:
    if tasks_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Task input must be a JSON object, array, or JSONL.")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_mock_response(response_path: Path) -> str:
    payload = response_path.read_text(encoding="utf-8").strip()
    if not payload:
        raise ValueError("Mock response file is empty.")
    return payload


def main() -> None:
    args = parse_args()
    root = project_root()
    tasks_path = (root / args.tasks).resolve() if not Path(args.tasks).is_absolute() else Path(args.tasks)
    output_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    raw_output_path = (root / args.raw_output).resolve() if not Path(args.raw_output).is_absolute() else Path(args.raw_output)
    prompt_path = (root / args.prompt).resolve() if not Path(args.prompt).is_absolute() else Path(args.prompt)
    response_path = None
    if args.response_file:
        response_path = (root / args.response_file).resolve() if not Path(args.response_file).is_absolute() else Path(args.response_file)

    settings = load_settings()
    prompt_template = load_prompt_template(prompt_path)
    tasks = load_tasks(tasks_path)

    if not tasks:
        raise ValueError("No tasks found in input file.")

    client = GemmaClient(settings)

    for index, task in enumerate(tasks):
        messages = build_teacher_messages(task, prompt_template)
        if args.dry_run:
            print(json.dumps({"messages": messages, "model": settings.model}, ensure_ascii=False, indent=2))
            if index == 0:
                return
            continue

        if response_path is not None:
            raw_content = load_mock_response(response_path)
        else:
            response = client.create_chat_completion(messages=messages)
            raw_content = client.extract_content(response)
        raw_record = {
            "task_id": task.get("task_id", f"task_{index + 1}"),
            "model": settings.model,
            "response": raw_content,
        }
        append_jsonl(raw_output_path, raw_record)

        extracted = extract_json_block(raw_content)
        normalized = normalize_teacher_trace(extracted)
        validation_errors = validate_teacher_trace(normalized)
        quality_score = score_sample_quality(normalized, validation_errors)
        training_record = build_training_record(
            task=task,
            trace=normalized,
            raw_response=raw_content,
            teacher_model=settings.model,
            prompt_version=settings.prompt_version,
            validation_errors=validation_errors,
            quality_score=quality_score,
        )
        append_jsonl(output_path, training_record)

        print(
            json.dumps(
                {
                    "task_id": task.get("task_id", f"task_{index + 1}"),
                    "quality_score": quality_score,
                    "validation_errors": validation_errors,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
