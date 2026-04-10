"""批量生成 coding domain teacher 資料，無需外部 API（用 mock/本地推理）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate coding domain teacher training data.")
    parser.add_argument("--task-list", required=True, help="JSONL with task definitions (instruction, context, constraints).")
    parser.add_argument("--output-raw", default="data/raw/coding_teacher_batch.jsonl")
    parser.add_argument("--output-processed", default="data/processed/coding_batch_train.jsonl")
    parser.add_argument("--use-api", action="store_true", help="Query real teacher API instead of using mock.")
    parser.add_argument("--mock-ratio", type=float, default=0.8, help="Fraction of samples to sample from mock patterns.")
    parser.add_argument("--seed", type=int, default=42)
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


def build_trace(plan: list[str], answer: str, decisions: list[str] | None = None) -> list[dict[str, str]]:
    trace_steps: list[dict[str, str]] = []
    decisions = decisions or []
    for index, goal in enumerate(plan):
        decision = decisions[index] if index < len(decisions) else f"完成步驟: {goal}"
        artifact = answer if index == len(plan) - 1 else goal
        trace_steps.append(
            {
                "goal": goal,
                "decision": decision,
                "produced_artifact": artifact,
            }
        )
    return trace_steps


def mock_teacher_response(task: dict[str, Any], sample_id: str) -> dict[str, Any]:
    """生成模擬 teacher 軌跡（用於快速測試）。"""
    instruction = task.get("instruction", "")
    lowered = instruction.lower()
    
    # 簡單啟發式：根據任務關鍵詞產生合理的假軌跡
    if "longest" in lowered or "最長子串" in instruction or "無重複字符" in instruction:
        plan = ["分析問題需要滑動窗口", "選定兩指針策略", "編寫核心迴圈", "驗證邊界"]
        steps = [
            {
                "goal": "理解問題",
                "decision": "使用滑動窗口 (sliding window)",
                "produced_artifact": "認識到需要 seen dict 與雙指針",
            },
            {
                "goal": "實現主迴圈",
                "decision": "遍歷 right pointer，更新 seen 與 left",
                "produced_artifact": "def length_of_longest_substring(s):\n    seen = {}\n    left = 0\n    best = 0",
            },
            {
                "goal": "完整解法",
                "decision": "加入邊界檢查與返回值",
                "produced_artifact": "完整可運行方案",
            },
        ]
        answer = "def length_of_longest_substring(s):\n    left = 0\n    seen = {}\n    best = 0\n    for right, ch in enumerate(s):\n        if ch in seen and seen[ch] >= left:\n            left = seen[ch] + 1\n        seen[ch] = right\n        best = max(best, right - left + 1)\n    return best"
    elif "average" in lowered or "平均值" in instruction:
        plan = ["檢查空列表", "計算和", "除以長度"]
        steps = [
            {
                "goal": "邊界條件",
                "decision": "若 nums 為空返回 0",
                "produced_artifact": "if not nums: return 0",
            },
            {
                "goal": "主邏輯",
                "decision": "sum(nums) / len(nums)",
                "produced_artifact": "def average(nums):\n    if not nums:\n        return 0\n    return sum(nums) / len(nums)",
            },
        ]
        answer = "def average(nums):\n    if not nums:\n        return 0\n    return sum(nums) / len(nums)"
    elif "括號" in instruction or "bracket" in lowered or "parentheses" in lowered:
        plan = ["建立括號映射", "遍歷字串維護 stack", "處理非法字符與剩餘 stack"]
        answer = "def is_valid_brackets(text):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in text:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n        else:\n            return False\n    return not stack"
        steps = build_trace(plan, answer, ["用 dict 表示對應括號", "左括號 push，右括號比對", "遇到非法字符直接失敗"]) 
    elif "二分" in instruction or "binary search" in lowered:
        plan = ["設定左右邊界", "迴圈縮小搜尋區間", "找不到返回 -1"]
        answer = "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        steps = build_trace(plan, answer, ["使用閉區間 [left, right]", "根據比較結果移動指標", "完整覆蓋 target 不存在情況"])
    elif "合並" in instruction or "merge" in lowered:
        plan = ["建立雙指針", "依序取較小值", "接上剩餘元素"]
        answer = "def merge_sorted_lists(left, right):\n    i = 0\n    j = 0\n    merged = []\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            merged.append(left[i])\n            i += 1\n        else:\n            merged.append(right[j])\n            j += 1\n    merged.extend(left[i:])\n    merged.extend(right[j:])\n    return merged"
        steps = build_trace(plan, answer)
    elif "反轉" in instruction or "reverse" in lowered:
        plan = ["確認輸入型別", "使用切片或雙指針反轉", "返回結果"]
        answer = "def reverse_string(text):\n    return text[::-1]"
        steps = build_trace(plan, answer)
    elif "去重" in instruction or "dedup" in lowered or "duplicate" in lowered:
        plan = ["建立 seen 集合", "保持原順序遍歷", "只加入首次出現元素"]
        answer = "def deduplicate(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"
        steps = build_trace(plan, answer)
    elif "階乘" in instruction or "factorial" in lowered:
        plan = ["處理負數與 0", "用迴圈累乘", "返回結果"]
        answer = "def factorial(n):\n    if n < 0:\n        return None\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result"
        steps = build_trace(plan, answer)
    elif "two sum" in lowered or "兩數之和" in instruction:
        plan = ["建立值到索引的映射", "逐步查找補數", "找到即返回索引"]
        answer = "def two_sum(nums, target):\n    seen = {}\n    for index, value in enumerate(nums):\n        need = target - value\n        if need in seen:\n            return [seen[need], index]\n        seen[value] = index\n    return []"
        steps = build_trace(plan, answer)
    elif "palindrome" in lowered or "回文" in instruction:
        plan = ["雙指針從兩端靠攏", "逐步比較字符", "全部相等則為回文"]
        answer = "def is_palindrome(text):\n    left, right = 0, len(text) - 1\n    while left < right:\n        if text[left] != text[right]:\n            return False\n        left += 1\n        right -= 1\n    return True"
        steps = build_trace(plan, answer)
    elif "fibonacci" in lowered or "費波那契" in instruction:
        plan = ["處理小於 2 的情況", "迭代更新前兩項", "返回第 n 項"]
        answer = "def fibonacci(n):\n    if n < 0:\n        return None\n    if n < 2:\n        return n\n    first, second = 0, 1\n    for _ in range(2, n + 1):\n        first, second = second, first + second\n    return second"
        steps = build_trace(plan, answer)
    elif "單詞" in instruction or "word" in lowered and "count" in lowered:
        plan = ["正規化空白", "依空白切分", "計算出現次數"]
        answer = "def count_words(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"
        steps = build_trace(plan, answer)
    elif "flatten" in lowered or "扁平" in instruction:
        plan = ["建立結果列表", "遞迴展開巢狀列表", "累積所有元素"]
        answer = "def flatten(items):\n    result = []\n    for item in items:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"
        steps = build_trace(plan, answer)
    elif "unique" in lowered or "唯一" in instruction:
        plan = ["比較集合大小", "若集合大小相同則唯一", "返回布林值"]
        answer = "def all_unique(items):\n    return len(items) == len(set(items))"
        steps = build_trace(plan, answer)
    elif "最大值" in instruction:
        plan = ["處理空列表", "遍歷更新最大值", "返回結果"]
        answer = "def find_max(nums):\n    if not nums:\n        return None\n    current = nums[0]\n    for value in nums[1:]:\n        if value > current:\n            current = value\n    return current"
        steps = build_trace(plan, answer)
    elif "最小值" in instruction:
        plan = ["處理空列表", "遍歷更新最小值", "返回結果"]
        answer = "def find_min(nums):\n    if not nums:\n        return None\n    current = nums[0]\n    for value in nums[1:]:\n        if value < current:\n            current = value\n    return current"
        steps = build_trace(plan, answer)
    elif "列表總和" in instruction:
        plan = ["處理空列表", "逐步累加元素", "返回總和"]
        answer = "def total(nums):\n    result = 0\n    for value in nums:\n        result += value\n    return result"
        steps = build_trace(plan, answer)
    elif "攝氏" in instruction and "華氏" in instruction:
        plan = ["套用轉換公式", "計算華氏溫度", "返回 float"]
        answer = "def celsius_to_fahrenheit(celsius):\n    return celsius * 9 / 5 + 32"
        steps = build_trace(plan, answer)
    elif "偶數" in instruction:
        plan = ["對 2 取餘", "比較是否為 0", "返回布林值"]
        answer = "def is_even(value):\n    return value % 2 == 0"
        steps = build_trace(plan, answer)
    elif "質數" in instruction:
        plan = ["排除小於 2", "只檢查到平方根", "找到因數即返回 False"]
        answer = "def is_prime(n):\n    if n < 2:\n        return False\n    divisor = 2\n    while divisor * divisor <= n:\n        if n % divisor == 0:\n            return False\n        divisor += 1\n    return True"
        steps = build_trace(plan, answer)
    elif "空白改成底線" in instruction:
        plan = ["使用字串 replace", "替換所有空白", "返回結果"]
        answer = "def spaces_to_underscores(text):\n    return text.replace(' ', '_')"
        steps = build_trace(plan, answer)
    elif "元音" in instruction:
        plan = ["建立元音集合", "遍歷字串計數", "返回總數"]
        answer = "def count_vowels(text):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for ch in text if ch in vowels)"
        steps = build_trace(plan, answer)
    elif "第二大" in instruction:
        plan = ["移除重複值", "確認至少有兩個不同值", "返回排序後倒數第二個"]
        answer = "def second_largest(nums):\n    unique = sorted(set(nums))\n    if len(unique) < 2:\n        return None\n    return unique[-2]"
        steps = build_trace(plan, answer)
    elif "共同前綴" in instruction:
        plan = ["從最短長度開始比較", "逐位檢查是否一致", "累積共同前綴"]
        answer = "def common_prefix(left, right):\n    result = []\n    for a, b in zip(left, right):\n        if a != b:\n            break\n        result.append(a)\n    return ''.join(result)"
        steps = build_trace(plan, answer)
    elif "單詞順序反轉" in instruction:
        plan = ["切分單詞", "反轉列表", "重新組合"]
        answer = "def reverse_words(text):\n    return ' '.join(reversed(text.split()))"
        steps = build_trace(plan, answer)
    elif "anagram" in lowered:
        plan = ["比較排序後字串", "若相同則為 anagram", "返回布林值"]
        answer = "def is_anagram(left, right):\n    return sorted(left) == sorted(right)"
        steps = build_trace(plan, answer)
    elif "none 值" in lowered or "none 值" in instruction or "none" in lowered and "移除" in instruction:
        plan = ["遍歷列表", "過濾 None", "保持原順序返回"]
        answer = "def remove_none(items):\n    return [item for item in items if item is not None]"
        steps = build_trace(plan, answer)
    elif "每個字符" in instruction and "出現次數" in instruction:
        plan = ["建立計數 dict", "遍歷字串更新計數", "返回結果"]
        answer = "def char_counts(text):\n    counts = {}\n    for ch in text:\n        counts[ch] = counts.get(ch, 0) + 1\n    return counts"
        steps = build_trace(plan, answer)
    elif "是否已排序" in instruction:
        plan = ["比較相鄰元素", "若出現降序則失敗", "否則返回 True"]
        answer = "def is_sorted(nums):\n    return all(nums[index] <= nums[index + 1] for index in range(len(nums) - 1))"
        steps = build_trace(plan, answer)
    elif "平方" in instruction:
        plan = ["遍歷列表", "對每個元素平方", "返回新列表"]
        answer = "def square_list(nums):\n    return [value * value for value in nums]"
        steps = build_trace(plan, answer)
    elif "缺失數字" in instruction:
        plan = ["利用理論總和", "減去實際總和", "得到缺失值"]
        answer = "def missing_number(nums):\n    expected = len(nums) * (len(nums) + 1) // 2\n    return expected - sum(nums)"
        steps = build_trace(plan, answer)
    elif "合併兩個 dict" in instruction or "merge two dict" in lowered:
        plan = ["先複製第一個 dict", "更新第二個 dict", "返回新 dict"]
        answer = "def merge_dicts(left, right):\n    merged = dict(left)\n    merged.update(right)\n    return merged"
        steps = build_trace(plan, answer)
    else:
        plan = ["分析輸入輸出", "選擇直接且可讀的實作", "補上邊界處理"]
        answer = "def solve(*args, **kwargs):\n    raise NotImplementedError('需要為此題型補充專用 mock')"
        steps = build_trace(plan, answer)

    return {
        "sample_id": sample_id,
        "task": task,
        "teacher_plan": {
            "task_summary": instruction[:80],
            "plan": plan,
        },
        "steps": steps,
        "final_output": {"answer": answer},
        "metadata": {
            "domain": "coding",
            "source": "mock_teacher",
            "quality_score": 0.95,
            "validation_errors": [],
        },
    }


def batch_generate(tasks: list[dict[str, Any]], use_api: bool = False, mock_ratio: float = 0.8, seed: int = 42) -> list[dict[str, Any]]:
    """批量產生 teacher 軌跡。"""
    import random
    
    generator = random.Random(seed)
    results: list[dict[str, Any]] = []

    for idx, task in enumerate(tasks):
        sample_id = task.get("task_id") or f"coding_sample_{idx:06d}"
        
        # 大部分使用 mock，少數可改用真實 API（未實裝）
        if generator.random() < mock_ratio or not use_api:
            result = mock_teacher_response(task, sample_id)
        else:
            # 如果要真正呼叫 API，可在這裡加入 GemmaClient 邏輯
            result = mock_teacher_response(task, sample_id)
        
        results.append(result)

    return results


def main() -> None:
    args = parse_args()
    root = project_root()
    task_list_path = (root / args.task_list).resolve() if not Path(args.task_list).is_absolute() else Path(args.task_list)
    raw_output_path = (root / args.output_raw).resolve() if not Path(args.output_raw).is_absolute() else Path(args.output_raw)
    processed_output_path = (root / args.output_processed).resolve() if not Path(args.output_processed).is_absolute() else Path(args.output_processed)

    tasks = load_jsonl(task_list_path)
    results = batch_generate(tasks, use_api=args.use_api, mock_ratio=args.mock_ratio, seed=args.seed)
    
    # 寫出原始 teacher 軌跡
    write_jsonl(raw_output_path, results)
    
    # 直接轉換成 SFT 格式
    sft_records = [
        {
            "sample_id": r.get("sample_id"),
            "messages": [
                {
                    "role": "system",
                    "content": "你是專注於 coding 任務的助理，請直接給出正確、簡潔且可執行的最終答案。",
                },
                {
                    "role": "user",
                    "content": r.get("task", {}).get("instruction", ""),
                },
                {
                    "role": "assistant",
                    "content": r.get("final_output", {}).get("answer", ""),
                },
            ],
            "metadata": {
                "domain": "coding",
                "source": "batch_teacher",
                "quality_score": r.get("metadata", {}).get("quality_score", 0.9),
            },
        }
        for r in results
    ]
    write_jsonl(processed_output_path, sft_records)
    
    summary = {
        "total_samples": len(results),
        "raw_output": str(raw_output_path),
        "processed_output": str(processed_output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
