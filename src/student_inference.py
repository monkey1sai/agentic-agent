"""學生模型推論與 base/teacher/student 比較評測"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import load_settings, project_root
from gemma_client import GemmaClient
from metrics import load_jsonl, run_simple_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on benchmark and compare against base/teacher.")
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--student-model", help="Path to fine-tuned student model (local or HF repo).")
    parser.add_argument("--base-model", default="google/gemma-4-E4B-it")
    parser.add_argument("--output", default="reports/comparison_results.json")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit inference to N samples (0=all).")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_inference(model_id: str, benchmarks: list[dict[str, Any]], mode: str = "teacher", base_model_id: str | None = None) -> list[dict[str, Any]]:
    """
    在 benchmark 上跑推論。
    mode 可為 'teacher'（即時呼叫教師 API）或 'student'（需要本地模型）。
    """
    if mode == "teacher":
        settings = load_settings()
        client = GemmaClient(settings)
        outputs: list[dict[str, Any]] = []
        for item in benchmarks:
            instruction = item.get("instruction", "")
            try:
                response = client.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "你是專注於 coding 任務的助理，請直接輸出最終答案。"},
                        {"role": "user", "content": instruction},
                    ],
                    temperature=0.2,
                    max_tokens=1200,
                )
                answer = client.extract_content(response)
            except Exception as e:
                answer = f"[Error: {str(e)}]"
            outputs.append({"benchmark_id": item.get("benchmark_id"), "answer": answer})
        return outputs
    
    # mode == 'student': 使用本地模型推論（需要 transformers + torch）
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        return [{"benchmark_id": item.get("benchmark_id"), "answer": "[transformers/torch not installed]"} for item in benchmarks]
    
    try:
        model_path = Path(model_id)
        if model_path.exists() and (model_path / "adapter_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            if not base_model_id:
                raise ValueError("base_model_id is required when loading a PEFT adapter directory")
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", load_in_4bit=True)
            model = PeftModel.from_pretrained(base_model, model_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
    except Exception as e:
        return [{"benchmark_id": item.get("benchmark_id"), "answer": f"[Model load error: {str(e)}]"} for item in benchmarks]
    
    outputs: list[dict[str, Any]] = []
    for item in benchmarks:
        instruction = item.get("instruction", "")
        try:
            inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1200, temperature=0.2, do_sample=True)
            answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            answer = f"[Inference error: {str(e)}]"
        outputs.append({"benchmark_id": item.get("benchmark_id"), "answer": answer})
    
    return outputs


def main() -> None:
    args = parse_args()
    root = project_root()
    benchmarks_path = (root / args.benchmarks).resolve() if not Path(args.benchmarks).is_absolute() else Path(args.benchmarks)
    output_path = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    benchmarks = load_jsonl(benchmarks_path)
    if args.max_samples > 0:
        benchmarks = benchmarks[:args.max_samples]

    results: dict[str, Any] = {
        "benchmark_count": len(benchmarks),
        "models": {},
    }

    # 教師模型推論
    print("[*] Running teacher model inference...")
    teacher_outputs = run_inference(args.base_model, benchmarks, mode="teacher", base_model_id=args.base_model)
    teacher_score = run_simple_benchmark(teacher_outputs, benchmarks)
    results["models"]["teacher"] = {
        "pass_rate": teacher_score.get("pass_rate", 0.0),
        "passed": teacher_score.get("passed", 0),
        "total": teacher_score.get("total", 0),
    }
    print(f"[✓] Teacher: {results['models']['teacher']['pass_rate']:.1%}")

    # Base 模型推論（如果提供 student_model，就用它；否則跳過）
    if args.student_model:
        print("[*] Running student model inference...")
        try:
            student_outputs = run_inference(args.student_model, benchmarks, mode="student", base_model_id=args.base_model)
            student_score = run_simple_benchmark(student_outputs, benchmarks)
            results["models"]["student"] = {
                "pass_rate": student_score.get("pass_rate", 0.0),
                "passed": student_score.get("passed", 0),
                "total": student_score.get("total", 0),
            }
            print(f"[✓] Student: {results['models']['student']['pass_rate']:.1%}")
        except Exception as e:
            results["models"]["student"] = {"error": str(e)}
            print(f"[✗] Student error: {str(e)}")

    write_json(output_path, results)
    print(f"\n[✓] Results written to {output_path}")


if __name__ == "__main__":
    main()
