---
name: share-lora-subspaces
description: >
  Use when users request "Share LoRA", "shared LoRA subspace", "continual LoRA",
  "replay-free continual finetuning", "merge LoRA adapters", "compress many LoRAs",
  "Share 論文", "共享 LoRA 子空間", "持續學習 LoRA", "LoRA 合併", "低記憶體微調".
metadata:
  version: 1.0.0
  category: ai-engineering
  model: claude-sonnet-4-20250514
  domains: [lora, continual-learning, peft, model-merging, memory-efficiency]
---

# Share LoRA Subspaces

把論文 Shared LoRA Subspaces for almost Strict Continual Learning 轉成可操作技能。

這個技能適合在你要做 replay-free continual finetuning、把多個 LoRA 壓成單一共享子空間、或在低記憶體環境下擴充任務數時使用。

## 這個技能解決什麼

一般 LoRA 會為每個任務保存一組 adapter。任務一多，儲存與切換成本會線性上升，也難以做跨任務知識共享。

Share 的核心做法是：

1. 從既有 LoRA 的 A/B 矩陣中抽出共享主因子子空間
2. 新任務來時，只學少量臨時因子與係數，而不是再長一套獨立 LoRA
3. 用 SVD 與投影把新知識併回單一共享子空間
4. 對舊任務係數做解析重投影，降低遺忘並保留 forward/backward transfer

## Core Workflow

1. 判斷輸入型態
   是 data-only、adapter-only，還是 hybrid stream。
2. 建立 foundational subspace
   從現有 LoRA adapters 的 A/B 因子做 centered SVD，取得 principal factors。
3. 持續適應新任務
   只對少量 temporary factors 與 coefficient 進行訓練。
4. 合併並重新投影
   將舊任務重建出的 adapter 與新學到的暫時因子一起做 SVD，更新共享子空間。
5. 決定是否進入 Share-full
   若可接受較弱的 continual 假設，可對重新計算後的係數再做少量 finetune。
6. 驗證與部署
   評估 retention、forgetting、adapter compression ratio 與 serving path。

## 何時用 Share 而不是標準 LoRA

用 Share：
- 任務會持續增加
- 不想保存每個 task 一套 adapter
- 想從新任務反過來改善舊任務表示
- 記憶體與儲存是主要瓶頸
- 你同時收到資料流與既有 LoRA adapters

不要優先用 Share：
- 只有單一任務一次性微調
- 任務彼此幾乎無共享子空間
- 不同 backbone 類型想混在同一個 continual stream
- 只有低品質 adapters，且完全沒有對應資料

## 操作協議

### 1. 任務建模

先把 continual setup 明確寫成下面格式：

```text
Base model: <model>
Backbone type: <same-backbone-required>
Incoming stream type: data-only | adapter-only | hybrid
Task sequence: [t1, t2, ...]
LoRA rank r: <int>
Memory target: <MB/GB>
Retention target: <metric>
```

### 2. 初始化共享子空間

對每一層分開做：

```text
收集 B1...Bt, A1...At
堆疊成 B_t = [B1, B2, ..., Bt], A_t = [A1, A2, ..., At]
中心化後做 SVD
保留 top-k 主因子 beta_t, alpha_t
凍結主因子，只訓練 epsilon_beta, epsilon_alpha
```

### 3. Continual Adaptation

新任務到來時：

```text
從現有 top-phi 因子初始化 temporary factors
初始化 temporary coefficients
只更新 temporary factors 與 coefficients
避免直接破壞 foundational subspace
```

### 4. Merging and Reprojection

合併時：

```text
先用舊主因子與係數重建舊任務近似 adapter
把重建出的舊 adapter 與新任務 temporary adapter 一起堆疊
再做一次 top-k SVD
得到新的 shared basis
用 pseudoinverse 把所有任務重新投影到新 basis 上
```

### 5. Hyperparameter 起手式

論文結論可直接當預設：

- `k`: 以 explained variance 閾值選，60% 起跳常常就夠
- `phi`: 建議在 `1` 到 `k/4` 間
- `p`: `1` 常有效；保守可從 `r/3` 起步

### 6. 評估

至少量四種東西：

- 當前任務表現
- 舊任務 retention / forgetting
- compression ratio
- temporary expansion 的訓練成本

## 重要技術點

### Foundational Subspace

論文假設相近任務的 LoRA adapters 共享一個低秩子空間。這不是直接共享整組 adapter，而是共享 principal directions。

### Frozen Principal Factors + Trainable Coefficients

Share 保留 frozen basis，只學小型係數。這是它相對 LoRA 省參數與省 optimizer state 的主因。

### Temporary Expansion

新任務不是永久新增一個 adapter，而是短暫擴張少量因子 `phi`，學完再吸收回共享子空間。

### Analytical Merge

合併主要依賴 SVD 與 pseudoinverse，是資料與梯度需求都較低的知識整合步驟。這點是 Share 很有特色的工程優勢。

### Share-full

如果 strict continual 假設可稍微放鬆，合併後可再微調係數。論文中這通常帶來更接近 joint / non-CL LoRA 的表現。

## 實作決策清單

在實作或審查 Share 設計時，逐項確認：

- [ ] 同一個 continual stream 是否只使用單一 backbone 類型
- [ ] 是否已定義 stream 類型是 data-only、adapter-only 或 hybrid
- [ ] 是否對每一層獨立做 basis 抽取與更新
- [ ] `k` 是否依 explained variance 或 eigenvalue 閾值選擇
- [ ] `phi` 是否小於 `k`
- [ ] `p` 是否先從小值起步
- [ ] 合併時是否先重建舊任務 adapter 再做 SVD
- [ ] 是否用 projection / pseudoinverse 重新計算舊任務係數
- [ ] 是否明確區分 strict Share 與 Share-full
- [ ] 是否評估 adapter-only 情境下的品質瓶頸

## 常見失敗模式

1. 把 Share 當成「共享同一個 LoRA 權重」
   這不對。Share 共享的是主因子子空間，不是單一 task adapter。

2. `k` 選太小
   會導致主子空間表達力不足，projection error 上升。

3. `phi` 太大
   會讓 temporary expansion 失去省記憶體優勢，也更容易干擾主子空間。

4. 只在 adapter-only 情境使用低品質來源
   論文明講這會形成資訊瓶頸，沒有資料就很難補救。

5. 混用不同 backbone
   論文限制之一就是單一 continual stream 內必須同 backbone 類型。

## 你應該怎麼用這個技能

如果需求是研究或設計：
讀 references/method.md 與 references/hyperparameters.md。

如果需求是原型實作：
先讀 references/implementation-template.md。

如果需求是做方案評估：
先讀 references/evaluation-and-limits.md。

## Output Checklist

- [ ] 明確寫出 Share 是否適合目前任務序列
- [ ] 給出 `r, k, p, phi` 的初始建議
- [ ] 解釋初始化、持續適應、合併三步驟
- [ ] 指出記憶體節省來自哪裡
- [ ] 明確列出限制與失效條件