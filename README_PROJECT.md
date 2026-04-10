# 專案概覽

## 目標

在編碼領域使用教師模型生成的軌跡，對 `google/gemma-4-E4B-it` 進行 QLoRA 微調，比較微調後的學生模型與教師/基礎模型的性能。

## 方案

```
User Tasks (coding)
         ↓
   Batch Teacher Generator (mock trajectories)
         ↓
  SFT Training Data (chat format)
         ↓
  [Domain Dataset Builder] ← [Training Config]
         ↓                        ↓
  Train/Valid/Test Sets    QLoRA Fine-tuning
         ↓
  [Student Model]
         ↓
  Inference + Benchmark Scoring
         ↓
  Base vs Teacher vs Student Comparison
```

## 核心模組

### 資料層
- **batch_teacher_generator.py**: 從任務清單產生教師軌跡（含 mock 支援）
- **domain_dataset_builder.py**: 將軌跡轉換為 SFT 訓練格式
- **metrics.py**: 簡單 substring/literal 檢查的 benchmark 評測

### 訓練層
- **run_qlora.py**: 使用 PEFT + Trainer 的 QLoRA 訓練入口
- **training_config.json**: 保守超參數（r=8, alpha=16, per_device_batch=1 + grad_accum=16）

### 推論層
- **eval_runner.py**: 教師/基礎模型推論與評測
- **student_inference.py**: 學生模型推論 + 教師比較

### 環境層
- **setup/SETUP.md**: WSL2 / Windows / Docker 三種安裝方案
- **setup/requirements.txt**: 完整 torch/transformers/peft/etc 版本釘選
- **TRAINING_ROADMAP.md**: 分步驟完整操作指南

## 當前進度

| 階段 | 狀態 | 備註 |
|------|------|------|
| 資料管線 | ✅ 完成 + 驗證 | 可批量產生編碼任務 SFT 資料 |
| 訓練骨架 | ✅ 完成 + dry-run | 支援 4-bit QLoRA，環境未就緒 |
| 評測框架 | ✅ 完成 + 驗證 | 可評測 base/teacher/student |
| 環境規格 | ✅ 完成 | 三種選項 + requirements.txt |
| 實際訓練 | ⏳ 待環境 | Python 3.13 + 缺依賴是主要阻塞 |

## 關鍵表現指标

訓練成功後將評測：
- **Teacher Pass Rate**: 教師模型在 benchmark 上的準確率
- **Base Pass Rate**: 基礎模型（未微調）的準確率
- **Student Pass Rate**: 微調後學生模型的準確率
- **Improvement**: Student vs Base 的相對進步

## 使用者入門

1. 閱讀 `QUICKSTART.md`（3 分鐘）
2. 選擇環境方案並執行 setup（15-30 分鐘）
3. 驗證 dry-run （2 分鐘）
4. 啟動訓練（5-30 分鐘）
5. 執行評測與比較（等待推論完成）

## 已知限制

- **本機環境**: Python 3.13 + 無 ML 套件，需升級
- **VRAM**: RTX 4060 Ti 8GB，建議初期 < 100 訓練樣本
- **時間**: 8 個樣本訓練約 5-10 分鐘，100 個樣本約 30 分鐘
- **Benchmark** 評測為簡單 substring 匹配，非完整功能驗證

## 後續優化方向

- [ ] 擴大訓練資料集（目前 8 個示例）
- [ ] 改進 benchmark 評測（加入執行測試）
- [ ] 支援多領域訓練（非僅編碼）
- [ ] 導出為邊際部署格式（ONNX / TensorRT）
