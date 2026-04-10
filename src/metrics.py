from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contains_checks(answer: str, must_include: list[str], must_not_include: list[str]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for item in must_include:
        if item not in answer:
            failures.append(f"missing required fragment: {item}")
    for item in must_not_include:
        if item in answer:
            failures.append(f"contains forbidden fragment: {item}")
    return not failures, failures


def run_simple_benchmark(predictions: list[dict[str, Any]], benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    pred_map = {item.get("benchmark_id"): item for item in predictions}
    passed = 0
    details: list[dict[str, Any]] = []
    for bench in benchmarks:
        bench_id = bench.get("benchmark_id")
        prediction = pred_map.get(bench_id, {})
        answer = str(prediction.get("answer", ""))
        reference = bench.get("reference") or {}
        bench_pass = False
        failures: list[str] = []
        if reference.get("type") == "contains":
            bench_pass, failures = contains_checks(answer, reference.get("must_include") or [], reference.get("must_not_include") or [])
        elif reference.get("type") == "unit-tests":
            for test_case in reference.get("tests") or []:
                expected = str(test_case.get("expected"))
                if expected not in answer:
                    failures.append(f"expected literal not found: {expected}")
            bench_pass = not failures
        if bench_pass:
            passed += 1
        details.append({"benchmark_id": bench_id, "pass": bench_pass, "failures": failures})
    total = len(benchmarks)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "details": details,
    }
