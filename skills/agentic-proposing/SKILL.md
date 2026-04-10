---
name: agentic-proposing
description: >
  Use when users request "Agentic Proposing", "compositional skill synthesis",
  "agentic data synthesis", "problem synthesis agent", "MGPO", "agentic SFT",
  "技能合成", "題目合成 agent", "合成高品質訓練資料", "推理技能庫", "可驗證資料生成".
metadata:
  version: 1.0.0
  category: ai-engineering
  model: claude-sonnet-4-20250514
  domains: [agentic-synthesis, reasoning-data, mgpo, skill-library, verifier, curriculum]
---

# Agentic Proposing

把論文 Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis 轉成可操作技能。

這個技能適用於你要建立「會自己組合技能、反思、用工具驗證、再產生高品質可驗證訓練資料」的 proposing agent，而不是單純做 one-shot prompt synthesis。

## 這個技能解決什麼

傳統資料合成方法常卡在兩種失敗模式：

1. 約束太強，題目結構穩但難度上不去
2. 約束太弱，難度變高但容易出現不一致、不可解、不可驗證樣本

Agentic Proposing 的核心做法是把問題合成視為 goal-driven sequential decision process：

1. 先建立可組合的 atomic skill library
2. 讓 proposer agent 在 Draft → Check → Refine → Finalize 的迴圈中動態組合技能
3. 透過 internal reflection、tool-use、dynamic skill pruning 修正邏輯
4. 用 verifier 與 prober 控制有效性與難度
5. 先做 agentic SFT，再用 MGPO 強化 proposer policy

## Core Workflow

1. 定義合成任務為 POMDP
   把問題有效性與難度視為潛在狀態，讓 agent 透過反思與工具探測。
2. 建立技能庫
   從 corpus 或 Q&A 中抽取 atomic skills，過濾後形成 `K_self`。
3. 做 agentic SFT
   用 teacher policy 產生帶有 reflection、tool call、skill pruning 的 expert trajectories，過濾後做 behavioral cloning。
4. 做 agentic RL
   用 MGPO 在 trajectory-level 與 stage-level 同時做 credit assignment。
5. 部署 proposer
   在 closed-loop synthesis 中持續生成可驗證且貼近 reasoning frontier 的資料。
6. 用 proposer 產生 solver training data
   注意 solver 端只吃最終 `{question, answer}`，不吃 proposer 中間軌跡。

## 方法全景

### 1. POMDP 問題定義

論文把問題合成定義成 POMDP，因為：

- 題目的真正可解性是 latent state
- 單靠表面文字歷史無法確知題目是否嚴謹
- agent 必須用 `think`、`exec`、`edit` 等行動降低不確定性

### 2. Observation

每步觀察包含：

```text
o_t = <K_t, h_t, sigma_t>
```

- `K_t`: 目前啟用的技能子集
- `h_t`: 對話歷史與工具輸出
- `sigma_t`: cognitive stage indicator，例如 drafting、checking、refining

### 3. Action Space

動作分三類：

- Cognitive actions `U`: 自然語言輸出與 internal reflection `tau_think`
- Interactive tools `T`: `tau_exec` 執行程式 / `tau_edit` 修剪技能集
- Final submission `F`: `tau_submit(q)` 提交最終題目

### 4. Skill Representation

每個 skill 不是自由文字，而是結構化 tuple：

```text
k = <iota, mu, delta, tau>
```

- `iota`: reasoning intent
- `mu`: construction method
- `delta`: difficulty effect
- `tau`: tool-use hint

### 5. 三階段訓練流程

論文真正的微調路線是三段式：

1. Skill Acquisition and Library Formalization
2. Agentic Supervised Fine-tuning
3. Agentic Post-training via MGPO

## 何時該用這個技能

用 Agentic Proposing：

- 你要的是高品質、可驗證的 reasoning data
- 你想讓 agent 主動組合技能，而不是只做模板改寫
- 你需要 internal reflection、tool-use、pruning 形成 closed loop
- 你要把 proposer 與 solver 分成兩個不同角色

不要優先用這套：

- 你只需要簡單 instruction augmentation
- 沒有 verifier、grader、sandbox 等外部基礎設施
- 你的場景不需要控制難度或可解性
- 你沒有能力維護 skill library 與 external prober

## 操作協議

### 1. 先定義 proposer 系統角色

至少把系統分成這幾個實體：

```text
Teacher model
Proposer policy
Verifier ensemble
Difficulty prober
Rule-based grader / executor
Skill library filesystem
```

### 2. 先建 skill library，再建 proposer

不要反過來。因為論文的 proposer 不是從空白 prompt 直接生長，而是站在 `K_self` 上決策與組合。

### 3. 先做 agentic SFT，再做 MGPO

MGPO 不是冷啟動演算法。論文先用高品質 `D_SFT` 把 proposer policy 拉到可用區間，再用 RL 精煉。

### 4. Solver 與 Proposer 資料分離

這點很重要：

- proposer 訓練吃完整 agentic trajectories
- solver 訓練只吃 final `{question, answer}`

### 5. 驗證與難度估計要外部化

論文不是讓 proposer 自己宣稱「這題很好」，而是外接：

- verifier ensemble 判斷 validity
- prober 用 Pass@k 估 difficulty

## 你應該先讀哪份 reference

如果你要理解整體框架：
讀 `references/framework.md`

如果你要實作微調與訓練：
讀 `references/finetuning-pipeline.md`

如果你要理解 MGPO：
讀 `references/mgpo.md`

如果你要建 skill library 與技能包：
讀 `references/skill-acquisition.md`

如果你要做 verifier/prober 與 acceptance rule：
讀 `references/verification-and-probing.md`

## Output Checklist

- [ ] 清楚區分 proposer 與 solver 的訓練資料
- [ ] 描述三階段訓練流程：skill acquisition → agentic SFT → MGPO
- [ ] 說明 Draft/Check/Refine/Finalize 閉環
- [ ] 包含 `think`, `exec`, `edit`, `submit` 的角色
- [ ] 說明 verifier、prober、curriculum 的關係
- [ ] 指出資料品質來自 closed-loop 驗證，而不是單次生成