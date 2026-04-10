# 📋 完整訓練路線圖

## 架構概覽

```
User Tasks (coding_tasks_batch.example.jsonl)
    ↓
[Student/Base Model]        [Teacher Model API]
    ↓                            ↓                  
Batch Generate SFT Data ← Mock Teacher Trajectory
    ↓
data/processed/coding_batch_sft.jsonl (8+ samples)
    ↓
[QLoRA Training] → artifacts/gemma4e4b-coding-qlora/
    ↓
Student Model Inference
    ↓
Benchmark Scoring (teacher vs student vs base)
    ↓
reports/comparison_results.json
```

## 完整步驟

### Step 0: 環境準備

選擇以下任一方案：

**A. WSL2 + Ubuntu 22.04 + Python 3.11（推薦）**
```bash
wsl
python3.11 -m venv ~/agentic-ml
source ~/agentic-ml/bin/activate
cd ~/agentic-agent  # 或 ln -s /mnt/c/.llmcode/agentic-agent
pip install -r setup/requirements.txt
```

**B. Windows + Python 3.11（風險較高）**
```powershell
python3.11 -m venv .\.venv-ml
.\.venv-ml\Scripts\Activate.ps1
pip install -r setup/requirements.txt
```

驗證安裝：
```bash
python train/run_qlora.py --config train/training_config.json --dry-run
# 應輸出 "python_ready": true
```

### Step 1: 生成編碼領域教師資料

#### 1a. 準備任務清單

若要產生大規模訓練資料，編輯或擴展 `examples/coding_tasks_batch.example.jsonl`，每行一個 JSON 任務定義：
```json
{"task_id":"coding_XXX","instruction":"...","context":{...},"constraints":[...],"domain":"coding"}
```

#### 1b. 批量產生 teacher 軌跡 + SFT 資料

```bash
# 使用範例（8 個任務，全部用 mock）
python src/batch_teacher_generator.py \
  --task-list examples/coding_tasks_batch.example.jsonl \
  --output-raw data/raw/coding_batch_teacher.jsonl \
  --output-processed data/processed/coding_batch_sft.jsonl

# 或：擴大規模（假設有更多任務檔案）
python src/batch_teacher_generator.py \
  --task-list examples/my_large_tasks.jsonl \
  --mock-ratio 0.7 \
  --output-processed data/processed/large_coding_sft.jsonl
```

**輸出**: 
- `data/raw/coding_batch_teacher.jsonl` - 原始 teacher 軌跡（含 plan/steps）
- `data/processed/coding_batch_sft.jsonl` - 可直接用於訓練的 SFT 格式

### Step 2: 轉換為訓練集

SFT 資料已在 Step 1 產生，但如果要併合多個來源或調整格式，可用：

```bash
python src/domain_dataset_builder.py \
  --input data/processed/coding_batch_sft.jsonl \
  --output-train data/processed/domain_train.jsonl \
  --output-valid data/processed/domain_valid.jsonl \
  --output-test data/processed/domain_test.jsonl \
  --valid-ratio 0.1 \
  --test-ratio 0.1
```

### Step 3: 啟動訓練

編輯 `train/training_config.json` 調整超參數（learning_rate, batch size 等），然後：

```bash
python train/run_qlora.py --config train/training_config.json
```

訓練產出將存於 `artifacts/gemma4e4b-coding-qlora/`

**預期時間**: ~5-30 分鐘（視 GPU 和資料量）

### Step 4: 推論與評測

#### 4a. 準備 benchmark

若無現成 benchmark，可用：
```bash
cp examples/coding_benchmark.example.jsonl data/eval/coding_benchmark_test.jsonl
```

#### 4b. 執行教師推論

```bash
python src/eval_runner.py \
  --mode teacher \
  --benchmarks data/eval/coding_benchmark_test.jsonl \
  --output reports/teacher_eval.json
```

#### 4c. 執行學生推論與比較

```bash
python src/student_inference.py \
  --benchmarks data/eval/coding_benchmark_test.jsonl \
  --student-model artifacts/gemma4e4b-coding-qlora/ \
  --base-model google/gemma-4-E4B-it \
  --output reports/comparison_results.json
```

### Step 5: 分析結果

檢查 `reports/comparison_results.json`：

```json
{
  "benchmark_count": 8,
  "models": {
    "teacher": {"pass_rate": 0.875, "passed": 7, "total": 8},
    "student": {"pass_rate": 0.75, "passed": 6, "total": 8}
  }
}
```

## 檔案結構

```
agentic-agent/
├── setup/
│   ├── requirements.txt          # ML 依賴版本清單
│   ├── SETUP.md                  # 環境設定指南
├── src/
│   ├── batch_teacher_generator.py    # 批量生成 teacher 資料
│   ├── domain_dataset_builder.py     # 轉換為訓練集
│   ├── eval_runner.py                # 評測 benchmark
│   ├── student_inference.py          # 學生推論 + 比較
│   ├── metrics.py
│   ├── config.py
│   ├── gemma_client.py
├── train/
│   ├── run_qlora.py              # 訓練入口
│   ├── training_config.json      # 超參數配置
│   ├── README.md
├── examples/
│   ├── coding_tasks_batch.example.jsonl   # 批量任務範例
│   ├── coding_benchmark.example.jsonl     # Benchmark 範例
├── data/
│   ├── raw/        # 原始 teacher 軌跡
│   ├── processed/  # SFT 訓練資料
│   ├── eval/       # Benchmark 與預測
├── artifacts/      # 訓練產出
├── reports/        # 評測報告
```

## 故障排查

### Dry-run 失敗：dependencies 為 false

環境未正確安裝 ML 套件。確認：
1. 虛擬環境已激活
2. 執行 `pip install -r setup/requirements.txt`
3. 確認 `pip list | grep torch`

### 訓練時 CUDA OOM

減少 `batch_size` 或增加 `gradient_accumulation_steps` 在 `train/training_config.json`

### 推論時找不到學生模型

確認訓練產出路徑正確。若訓練未完成，用教師推論作為基準：
```bash
python src/student_inference.py --benchmarks data/eval/coding_benchmark_test.jsonl
```

## 下一步

- 擴大任務規模（100+ 樣本）以提升訓練效果
- 根據 benchmark 結果調整超參數
- 導出學生模型為 ONNX 或 TF Lite 格式用於邊界部署
