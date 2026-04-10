# Share 超參數指南

這份文件只回答一件事：`k`、`p`、`phi` 怎麼選。

## 1. `k` Principal Factors

`k` 是共享子空間的維度。

### 論文建議

- 用 explained variance threshold 選擇
- 60% explained variance 已經常有效
- 任務增加後，所需 `k` 只會緩慢成長，並趨於收斂

### 工程起手式

- 小模型、小資料：先試 `k = min(r, 8~16)`
- LLM / 多任務設定：可從 `k = 32` 開始
- 若重建誤差高、舊任務 retention 差：增加 `k`

### 失敗訊號

- projection error 高
- 舊任務性能持續下滑
- 新任務學習需要明顯更大 `phi`

## 2. `p` Pseudo-rank

`p` 是每個任務係數矩陣的維度。

### 論文結論

- `p = 1` 已相當有效
- 從 `p = r/3` 開始是保守做法
- `p` 增加會帶來收益，但邊際效益下降

### 工程起手式

- 超低記憶體：`p = 1`
- 想保守些：`p = max(1, r // 3)`
- 若你觀察到任務多樣性高且 underfit：逐步加到 `2, 4, 8`

### 對記憶體的影響

`p` 直接影響：

- trainable coefficient 數量
- optimizer state
- forward 中的中間矩陣大小

所以在 8GB VRAM 這類環境，先壓 `p` 通常比先壓 `k` 更安全。

## 3. `phi` Temporary Factors

`phi` 是新任務持續適應時的暫時擴張維度。

### 論文建議

- `phi ∈ [1, k/4]`
- 實驗中 `phi = 2` 常常夠用

### 工程起手式

- 小規模原型：`phi = 1` 或 `2`
- 若新任務與既有任務差異大：提高到 `k/4`
- 除非有明確證據，不要讓 `phi` 太接近 `k`

### 失敗訊號

- 新任務學不進去：`phi` 可能太小
- 訓練成本大增：`phi` 太大

## 4. 與標準 LoRA rank `r` 的關係

論文中的常見設定：

- NLU / text-to-image / large-scale LoRA merging：`r=16` 或 `32`
- Share 常搭配：`k=32`, `p=8`, `phi=2~4`
- vision classification：`k=10`, `p=1`, `phi=2`

## 5. 可直接抄的預設

### 超保守省記憶體

```json
{
  "r": 16,
  "k": 8,
  "p": 1,
  "phi": 1
}
```

### 一般 LLM continual adapter 壓縮

```json
{
  "r": 16,
  "k": 16,
  "p": 2,
  "phi": 2
}
```

### 接近論文大模型合併設定

```json
{
  "r": 16,
  "k": 32,
  "p": 8,
  "phi": 2
}
```

## 6. 調參優先順序

若失敗，優先按這個順序調：

1. 先調 `phi`
2. 再調 `p`
3. 最後調 `k`

理由是：

- `phi` 控制新方向探索能力
- `p` 控制任務係數表達能力
- `k` 影響整個共享子空間大小，成本變化最大