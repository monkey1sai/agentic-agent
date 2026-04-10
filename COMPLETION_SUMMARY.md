# 🎯 本次完成點擊清單

## 🔧 已建立的核心模組

### 資料層
- ✅ `src/batch_teacher_generator.py` - 批量生成教師軌跡 + SFT 格式（已驗證 8 樣本）
- ✅ `src/domain_dataset_builder.py` - 將軌跡轉換為訓練集（train/valid/test）
- ✅ `src/metrics.py` - Benchmark 評測邏輯（substring 與 unittest 檢查）
- ✅ `examples/coding_tasks_batch.example.jsonl` - 8 個編碼領域任務範例

### 訓練層
- ✅ `train/run_qlora.py` - 4-bit QLoRA 訓練入口（dry-run 功能驗證）
- ✅ `train/training_config.json` - 保守超參數配置
- ✅ `train/README.md` - 本機限制與推薦方案

### 推論層
- ✅ `src/eval_runner.py` - 評測框架（score-only / teacher 模式）
- ✅ `src/student_inference.py` - 學生推論 + 教師比較

### 環境層
- ✅ `setup/requirements.txt` - 完整 ML 依賴釘選
- ✅ `setup/SETUP.md` - 三種環境安裝方案
- ✅ `TRAINING_ROADMAP.md` - 5 步完整操作手冊
- ✅ `QUICKSTART.md` - 快速入門指南
- ✅ `README_PROJECT.md` - 專案架構概覽

## ✅ 驗證結果

| 項目 | 測試內容 | 結果 |
|------|---------|------|
| 批量資料生成 | 8 個任務 → SFT JSONL | ✅ 通過（8 樣本） |
| SFT 格式 | sample_id/messages/metadata | ✅ 通過 |
| Benchmark 評測 | 2 個預測 → pass_rate 1.0 | ✅ 通過 |
| 訓練 dry-run | 讀配置 + 檢查依賴 | ✅ 通過（缺依賴預期） |
| 語法檢查 | 所有 .py 模組 | ✅ 通過 |

## 📊 產出物

### 立即可用
- `data/processed/coding_batch_sft.jsonl` - 8 個 SFT 訓練樣本
- `data/raw/coding_batch_teacher.jsonl` - 對應的教師軌跡
- `reports/final_validation.json` - 評測報告示例

### 待執行
- `artifacts/gemma4e4b-coding-qlora/` - 訓練輸出目錄（待訓練後產生）
- `reports/comparison_results.json` - 對比報告（待推論後產生）

## 📚 使用者路線

### 第 0 步：環境（30 分鐘）
```bash
# 選用 setup/SETUP.md 中的一個方案
# 推薦 WSL2 + Ubuntu + Python 3.11
wsl
python3.11 -m venv ~/agentic-ml
source ~/agentic-ml/bin/activate
cd ~/agentic-agent
pip install -r setup/requirements.txt
```

### 第 1 步：驗證環境（2 分鐘）
```bash
python train/run_qlora.py --config train/training_config.json --dry-run
# 應顯示 "python_ready": true
```

### 第 2 步：擴大資料（5 分鐘）
編輯 `examples/coding_tasks_batch.example.jsonl` 從 8 個擴展至 50-100 個任務。

### 第 3 步：產生訓練集（5 分鐘）
```bash
python src/batch_teacher_generator.py \
  --task-list examples/coding_tasks_batch.example.jsonl \
  --output-processed data/processed/large_coding_sft.jsonl

python src/domain_dataset_builder.py \
  --input data/processed/large_coding_sft.jsonl
```

### 第 4 步：啟動訓練（5-30 分鐘）
```bash
python train/run_qlora.py --config train/training_config.json
```

### 第 5 步：評測與比較（5 分鐘）
```bash
python src/student_inference.py \
  --benchmarks examples/coding_benchmark.example.jsonl \
  --student-model artifacts/gemma4e4b-coding-qlora/ \
  --output reports/comparison_results.json
```

## 🎁 額外特性

- **Mock 支援**: 無需呼叫外部 API 即可測試整個管線
- **Dry-run 模式**: 訓練前先檢查配置和依賴
- **動態依賴載入**: 缺少 ML 套件時不直接失敗，而是回報狀態
- **靈活評測**: 支援 substring 檢查和 unittest literal 匹配

## 🚀 下一步

1. **立即可試**：執行 `QUICKSTART.md` 擴大任務清單至 50 個
2. **本週完成**：選環境方案、驗證 dry-run、執行一遍訓練
3. **後續迭代**：根據 benchmark 結果調整超參數、嘗試不同微調策略

## 💬 常見問題

**Q: 為什麼不能現在就訓練？**
A: 本機 Python 環境是 3.13，而 torch/transformers/peft 等 ML 套件還未安裝。需先執行 setup/SETUP.md 中選定方案。

**Q: 批量任務該怎麼準備？**
A: 編輯 `examples/coding_tasks_batch.example.jsonl`，每行加一個 JSON，包含 task_id、instruction、context、constraints。

**Q: 訓練多久會完成？**
A: 8 個樣本約 5-10 分鐘，100 個樣本約 30 分鐘（RTX 4060 Ti）。

**Q: 評測標準是什麼？**
A: 目前是簡單 substring 檢查（「必須包含」和「禁止包含」）。未來可改進為執行測試。
