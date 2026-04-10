# 評估與限制

## 先量什麼

### 1. 當前任務品質

新任務不能只看 train loss。要看最終下游指標，例如：

- classification accuracy
- Rouge-L
- CLIP score
- JSON 結構正確率

### 2. 舊任務 retention

至少記錄：

- historical best
- current score
- forgetting = historical best - current score

### 3. 壓縮收益

對比：

- 多個獨立 LoRA 的總大小
- Share 的 principal factors + coefficients 總大小
- temporary expansion 峰值大小

### 4. 子空間品質

可量：

- reconstruction error
- explained variance
- factor similarity，例如 CKA

## 論文中可直接借用的結論

- Share 可接近 joint / non-CL LoRA 的表現
- 在 NLU 設定中，論文報告最高約 100× 參數降低與 281× 記憶體節省
- 在大規模 LoRA 壓縮場景，單一 Share 可壓縮數百個 adapters
- backward transfer 有時會出現，主因是 merge 後舊任務係數被重新投影到更好的共享子空間

## 限制

### 1. 單一 continual stream 內需同 backbone 類型

論文明確限制目前不能把不同預訓練架構混進同一個 continual task stream。

### 2. 純 adapter-only 情境受來源品質限制

如果沒有資料，只能吃到低品質 adapter，那 Share 的上限會被初始資訊品質卡住。

### 3. 不是跨 task 任意泛化魔法

如果任務根本不共享低秩子空間，Share 的優勢會下降。論文理論與實驗都依賴 tasks 至少有某種共享結構。

### 4. 非論文重現時要小心實作偏差

最容易偏離論文精神的地方：

- 把 basis 也一路 full-train
- 沒有做解析式重投影
- 合併時直接平均而不是重新做 SVD

## 實務判斷規則

考慮採用 Share，如果：

- 你有持續流入的 adapter 或資料
- 任務數成長快於可保存 adapter 的預算
- 想保留 replay-free 或近 strict continual learning 條件

考慮改回標準 LoRA，如果：

- 任務很少
- 每任務都可以單獨部署
- 你最在乎單任務峰值表現，而非長期壓縮和累積學習