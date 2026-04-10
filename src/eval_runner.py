from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import load_settings, project_root
from gemma_client import GemmaClient
from metrics import load_jsonl, run_simple_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base/teacher/student evaluation on coding benchmarks.")
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--predictions", help="Existing predictions JSONL to score.")
    parser.add_argument("--output", default="reports/latest_eval_summary.json")
    parser.add_argument("--mode", choices=["score-only", "teacher"], default="score-only")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def score_existing(benchmarks_path: Path, predictions_path: Path) -> dict[str, Any]:
    benchmarks = load_jsonl(benchmarks_path)
    predictions = load_jsonl(predictions_path)
    return run_simple_benchmark(predictions, benchmarks)


def run_teacher_eval(benchmarks_path: Path) -> list[dict[str, Any]]:
    settings = load_settings()
    client = GemmaClient(settings)
    benchmarks = load_jsonl(benchmarks_path)
    outputs: list[dict[str, Any]] = []
    for item in benchmarks:
        instruction = item.get("instruction", "")
        response = client.create_chat_completion(messages=[
            {"role": "system", "content": "你是專注於 coding 任務的助理，請直接輸出最終答案。"},
            {"role": "user", "content": instruction},
        ], temperature=0.2, max_tokens=1200)
        answer = client.extract_content(response)
        outputs.append({"benchmark_id": item.get("benchmark_id"), "answer": answer})
    return outputs


def main() -> None:
    args = parse_args()
    root = project_root()
    benchmarks_path = (root / args.benchmarks).resolve() if not Path(args.benchmarks).is_absolute() else Path(args.benchmarks)
    output_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    if args.mode == "score-only":
        if not args.predictions:
            raise ValueError("--predictions is required in score-only mode")
        predictions_path = (root / args.predictions).resolve() if not Path(args.predictions).is_absolute() else Path(args.predictions)
        summary = score_existing(benchmarks_path, predictions_path)
        write_json(output_path, summary)
        print(json.dumps(summary, ensure_ascii=False))
        return

    predictions = run_teacher_eval(benchmarks_path)
    score = run_simple_benchmark(predictions, load_jsonl(benchmarks_path))
    payload = {"predictions": predictions, "summary": score}
    write_json(output_path, payload)
    print(json.dumps(score, ensure_ascii=False))


if __name__ == "__main__":
    main()
