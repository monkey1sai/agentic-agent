from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from config import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build domain-specific SFT datasets from agentic training data.")
    parser.add_argument("--input", required=True, help="Path to agentic JSONL input.")
    parser.add_argument("--output-train", default="data/processed/domain_train.jsonl")
    parser.add_argument("--output-valid", default="data/processed/domain_valid.jsonl")
    parser.add_argument("--output-test", default="data/processed/domain_test.jsonl")
    parser.add_argument("--domain", default="coding")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_assistant_content(sample: dict[str, Any], max_steps: int) -> str:
    teacher_plan = sample.get("teacher_plan") or {}
    steps = (sample.get("steps") or [])[:max_steps]
    final_output = sample.get("final_output") or {}
    parts: list[str] = []
    summary = teacher_plan.get("task_summary")
    if summary:
        parts.append(f"摘要: {summary}")
    plan = teacher_plan.get("plan") or []
    if plan:
        parts.append("計畫:")
        parts.extend(f"- {item}" for item in plan[:max_steps])
    if steps:
        parts.append("執行:")
        for step in steps:
            goal = step.get("goal", "")
            decision = step.get("decision", "")
            produced = step.get("produced_artifact", "")
            parts.append(f"- {goal}; 決策: {decision}; 產出: {produced}")
    answer = final_output.get("answer")
    if answer:
        parts.append("最終答案:")
        parts.append(answer)
    return "\n".join(parts).strip()


def format_user_content(task: dict[str, Any]) -> str:
    lines = [f"任務: {task.get('instruction', '')}"]
    context = task.get("context") or {}
    language = context.get("language")
    if language:
        lines.append(f"語言: {language}")
    snippet = context.get("snippet")
    if snippet:
        lines.append("程式碼:")
        lines.append(snippet)
    constraints = task.get("constraints") or []
    if constraints:
        lines.append("限制:")
        lines.extend(f"- {item}" for item in constraints)
    artifacts = task.get("expected_artifact") or []
    if artifacts:
        lines.append("預期產出:")
        lines.extend(f"- {item}" for item in artifacts)
    return "\n".join(lines).strip()


def convert_agentic_to_sft(sample: dict[str, Any], max_steps: int) -> dict[str, Any]:
    task = sample.get("task") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "messages": [
            {
                "role": "system",
                "content": "你是專注於 coding 任務的助理，請直接給出正確、簡潔且可執行的最終答案。",
            },
            {
                "role": "user",
                "content": format_user_content(task),
            },
            {
                "role": "assistant",
                "content": compact_assistant_content(sample, max_steps=max_steps),
            },
        ],
        "metadata": {
            "domain": task.get("domain", "coding"),
            "source": "agentic_teacher",
            "format": "chat-sft",
        },
    }


def split_records(records: list[dict[str, Any]], valid_ratio: float, test_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generator = random.Random(seed)
    shuffled = records[:]
    generator.shuffle(shuffled)
    total = len(shuffled)
    valid_count = int(total * valid_ratio)
    test_count = int(total * test_ratio)
    valid_records = shuffled[:valid_count]
    test_records = shuffled[valid_count:valid_count + test_count]
    train_records = shuffled[valid_count + test_count:]
    return train_records, valid_records, test_records


def main() -> None:
    args = parse_args()
    root = project_root()
    input_path = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    train_path = (root / args.output_train).resolve() if not Path(args.output_train).is_absolute() else Path(args.output_train)
    valid_path = (root / args.output_valid).resolve() if not Path(args.output_valid).is_absolute() else Path(args.output_valid)
    test_path = (root / args.output_test).resolve() if not Path(args.output_test).is_absolute() else Path(args.output_test)

    records = load_jsonl(input_path)
    filtered = [record for record in records if (record.get("task") or {}).get("domain", args.domain) == args.domain]
    sft_records = [convert_agentic_to_sft(record, max_steps=args.max_steps) for record in filtered]
    train_records, valid_records, test_records = split_records(sft_records, args.valid_ratio, args.test_ratio, args.seed)
    write_jsonl(train_path, train_records)
    write_jsonl(valid_path, valid_records)
    write_jsonl(test_path, test_records)
    print(json.dumps({"train": len(train_records), "valid": len(valid_records), "test": len(test_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
