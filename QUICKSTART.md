# 🚀 快速開始指南

## 現況

編碼微調流程已完整建立並驗證，包括：
- ✅ 批量資料生成（teacher mock trajectories）
- ✅ SFT 轉換與訓練集準備
- ✅ 基礎與教師推論入口
- ✅ Benchmark 評測框架
- ⏳ 環境升級到 Python 3.10/3.11 + ML 套件（尚未進行）

## 立即可做的事

### 1. 擴大編碼任務清單

編輯 `examples/coding_tasks_batch.example.jsonl`，每行加一個新任務。模板：

```json
{"task_id":"coding_009","instruction":"你的任務敘述","context":{"language":"Python","snippet":""},"constraints":["限制1","限制2"],"expected_artifact":["產出1"],"domain":"coding"}
```

### 2. 產生 100+ 個訓練樣本

```bash
python src/batch_teacher_generator.py \
  --task-list examples/coding_tasks_batch.example.jsonl \
  --output-processed data/processed/large_coding_sft.jsonl
```

### 3. 驗證訓練設定

```bash
python train/run_qlora.py --config train/training_config.json --dry-run
```

預期輸出應包含 `"python_ready": false`（目前環境）。

## 準備訓練環境

選擇一個方案，按照 `setup/SETUP.md` 的步驟執行。完成後，dry-run 應顯示 `"python_ready": true`。

### 推薦：WSL2 + Ubuntu

```bash
wsl
python3.11 -m venv ~/agentic-ml
source ~/agentic-ml/bin/activate
cd ~/agentic-agent
pip install -r setup/requirements.txt
python train/run_qlora.py --config train/training_config.json --dry-run
```

## 啟動訓練（環境就緒後）

```bash
python train/run_qlora.py --config train/training_config.json
```

**預期時間**：5-30 分鐘（視 GPU 與數據量）

**輸出**：`artifacts/gemma4e4b-coding-qlora/`

## 評測成果

### 方式 A：教師模型推論

```bash
python src/eval_runner.py \
  --mode teacher \
  --benchmarks examples/coding_benchmark.example.jsonl \
  --output reports/teacher_eval.json
```

### 方式 B：學生 vs 教師比較

```bash
python src/student_inference.py \
  --benchmarks examples/coding_benchmark.example.jsonl \
  --student-model artifacts/gemma4e4b-coding-qlora/ \
  --output reports/comparison_results.json
```

## 當前檔案清單

| 檔案 | 用途 | 狀態 |
|------|------|------|
| `setup/requirements.txt` | ML 依賴版本表 | ✅ |
| `setup/SETUP.md` | 環境設定指南 | ✅ |
| `TRAINING_ROADMAP.md` | 完整操作步驟 | ✅ |
| `src/batch_teacher_generator.py` | 批量產生教師資料 | ✅ 驗證 |
| `src/domain_dataset_builder.py` | 轉換為訓練集 | ✅ 驗證 |
| `src/eval_runner.py` | 評測 benchmark | ✅ 驗證 |
| `src/student_inference.py` | 學生推論 + 比較 | ✅ 驗證 |
| `src/metrics.py` | 計分邏輯 | ✅ 驗證 |
| `train/run_qlora.py` | 訓練入口 | ✅ dry-run |
| `train/training_config.json` | 訓練超參數 | ✅ |
| `examples/coding_tasks_batch.example.jsonl` | 8 個範例任務 | ✅ 已驗證 |
| `examples/coding_benchmark.example.jsonl` | Benchmark 範例 | ✅ 已驗證 |
| `data/processed/coding_batch_sft.jsonl` | 已產生的訓練數據 | ✅ 8 樣本 |

## 下一步建議

1. **短期**（今天）
   - 選擇並執行 setup/SETUP.md 中的任一方案
   - 驗證環境：dry-run 應顯示 `"python_ready": true`

2. **中期**（本周）
   - 擴大任務清單到 50-100 個編碼題
   - 執行訓練一遍
   - 記錄初始 baseline（base model 與 teacher 的 benchmark 成績）

3. **長期**（下周）
   - 根據 benchmark 結果調整超參數
   - 進行多輪訓練實驗
   - 分析學生 vs 教師的差距

## 與我再次協作

若需要：
- 調整訓練超參數 → 編輯 `train/training_config.json` 或詢問我
- 新增任務類型 → 擴展 `examples/coding_tasks_batch.example.jsonl`
- 除錯訓練過程 → 我可檢查 logs 或提供排查指引
- 優化模型推論 → 可考慮量化或其他部署技巧

**聯繫方式**：直接告訴我你遇到的問題或想要嘗試的新方向。
