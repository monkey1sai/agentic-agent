# Gemma 4 E4B Coding Finetuning

這份流程是針對 coding domain 的最小 Gemma 4 E4B 微調骨架。

## 本機限制

目前這台機器的 Python 環境為 3.13，且未安裝 `torch`、`transformers`、`peft`、`datasets`、`accelerate`、`trl`、`bitsandbytes`。
此外，`bitsandbytes` 與多數 QLoRA 工作流通常建議在 Linux / WSL2 與 Python 3.10 或 3.11 下執行。

## 推薦環境

- Windows 使用者：優先改在 WSL2 Ubuntu
- Python: 3.10 或 3.11
- GPU: 至少 8GB VRAM，可先嘗試極保守 QLoRA；較穩定建議 16GB 以上

## 流程

1. 先生成 agentic teacher data
2. 再轉成 coding domain SFT 資料
3. 啟動 QLoRA 微調
4. 對 benchmark 跑 base / teacher / student 比較

## 指令

### 1. 生成 coding domain SFT 資料

```powershell
python src/domain_dataset_builder.py --input data/processed/test_agentic_train.jsonl --domain coding --output-train data/processed/domain_train.jsonl --output-valid data/processed/domain_valid.jsonl --output-test data/processed/domain_test.jsonl
```

### 2. Dry-run 檢查訓練配置

```powershell
python train/run_qlora.py --config train/training_config.json --dry-run
```

### 3. 啟動訓練

```powershell
python train/run_qlora.py --config train/training_config.json
```

### 4. 評測既有預測

```powershell
python src/eval_runner.py --mode score-only --benchmarks examples/coding_benchmark.example.jsonl --predictions data/eval/student_outputs.jsonl
```

## 重要提醒

這個專案目前已補齊訓練與評測骨架，但在現有本機環境下尚未真正啟動 Gemma 4 E4B 微調，因為缺少相容套件與相容 Python 版本。
