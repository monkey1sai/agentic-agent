# Todo

- [x] 盤點現有資料管線
- [x] 設計 coding 微調資料格式
- [x] 建立 Gemma 訓練骨架
- [x] 建立推論與比較評測
- [x] 準備相容訓練環境（指南完成，實裝待用戶執行）
- [ ] 啟動實際 Gemma QLoRA 微調（環境就緒後）
- [ ] 比較 base / teacher / student 成果（環境就緒後）

## Review

### 已完成
- domain dataset builder 已驗證可由 agentic JSONL 產出 coding SFT 資料。
- eval runner 已驗證可對 benchmark 與 predictions 進行 scoring （pass_rate 1.0）。
- QLoRA 訓練入口 dry-run 已驗證可讀取設定並回報依賴狀態。
- batch_teacher_generator 已驗證可從任務清單產生 8 個有效的 SFT 樣本。
- student_inference 已實裝支援教師/學生推論與比較。
- 環境設置指南已提供三種方案 (WSL2、Windows 原生、Docker)。

### 尚未啟動
- 實際訓練：需先執行 setup/SETUP.md。
- 比較評測：訓練完成後執行 src/student_inference.py。

### 關鍵文檔
- QUICKSTART.md - 立即開始
- README_PROJECT.md - 整體方案
- TRAINING_ROADMAP.md - 分步驟手冊
