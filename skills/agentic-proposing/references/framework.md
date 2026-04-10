# Agentic Proposing 框架總覽

## 核心觀點

論文的核心不是「生成題目」，而是「把題目合成變成 agent 決策問題」。

它把 problem synthesis 視為：

- 有技能庫可調用
- 有 latent validity / difficulty 狀態
- 有 tool-use 與 internal reflection
- 有 verifier 與 prober 回授

因此不是 open-loop text generation，而是 closed-loop compositional logic engineering。

## POMDP 定義

論文把 synthesis 任務形式化為 POMDP：

```text
(S, A, O, P, R, gamma)
```

- `S`: 題目的潛在邏輯完整性與難度
- `A`: agent 可採取的動作
- `O`: 可觀察資訊
- `P`: 狀態轉移
- `R`: 回饋函數
- `gamma`: discount

為什麼是 POMDP 而不是一般 MDP：

因為題目是否可解、是否自洽，不能只從表面輸出立即得知，agent 必須主動 probe。

## Observation 設計

```text
o_t = <K_t, h_t, sigma_t>
```

- `K_t`: 當前啟用技能子集
- `h_t`: 先前對話與工具結果
- `sigma_t`: cognitive stage

`sigma_t` 很關鍵，它不是死板 state machine，而是 functional context。agent 可以因為發現錯誤而回退到 refine。

## Action Space 設計

### 1. Cognitive Actions

- 自然語言思考與規劃
- `tau_think` 做 internal reflection、邏輯稽核

### 2. Interactive Tools

- `tau_exec`: 執行沙盒程式
- `tau_edit`: 修剪技能集合，刪掉不適合當前題目構造的 skill

### 3. Terminal Submission

- `tau_submit(q)`: 提交最終題目

## Draft → Check → Refine → Finalize

這個流程是整篇論文最值得工程化的部分：

1. Draft
   根據 skill composition 先構造題目草稿。
2. Check
   用 `think` 與 `exec` 檢查邏輯一致性、可解性、格式性。
3. Refine
   若發現技能衝突、不可解、難度失準，就回頭修正，必要時 `edit` 技能集。
4. Finalize
   題目通過 verifier 與基本可解檢查後再提交。

## 技能組合的意義

論文把技能組合 formalize 成：

```text
k = <iota, mu, delta, tau>
q ~ pi_theta(. | o_t, Phi(k_1, ..., k_n), K_t)
```

重點是：

- skill 不是普通 tag
- skill composition 會被映射成具體的 instruction constraints
- agent 的工作是選 skill、組 skill、修 skill，而不是一次寫出完整題目

## 為什麼這套方法有效

1. skill library 提供 compositional prior
2. internal reflection 避免單次生成的隨機錯誤
3. tool-use 提供外部、可操作的驗證能力
4. dynamic pruning 讓 agent 能在生成途中修掉錯誤生成路徑
5. verifier + prober 把「有效」與「有難度」拆成兩個不同訊號

## 最關鍵的設計原則

如果你要重建這篇論文的精神，請保留這五點：

1. proposer 必須是 agent，不是模板 prompt
2. skill 必須是結構化模組，不是普通知識點列表
3. 生成流程必須閉環，不是 one-shot
4. validity 與 difficulty 必須分開評估
5. proposer training 與 solver training 必須拆開