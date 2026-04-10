# 微調方法、步驟與流程

這份文件直接回答「論文怎麼微調 proposer」。

## 總流程

論文的 proposer 微調是三段式：

1. Skill Acquisition and Library Formalization
2. Agentic Supervised Fine-tuning
3. Agentic Reinforcement Learning via MGPO

## Stage 1: Skill Acquisition

### 目的

建立 `K_self`，也就是 proposer 之後會用來組合題目的 atomic skills。

### 輸入

- mixed-source corpus
- teacher model

### 輸出

- 被過濾後的高品質技能集合

### 關鍵點

- 不是單純抽 keyword
- 要抽成結構化 agent skill
- 要用 quality threshold 過濾

## Stage 2: Agentic SFT

### 目的

讓 proposer 先學會像 expert agent 一樣運作，而不是直接靠 RL 硬學。

### 教師資料格式

每條軌跡都是：

```text
tau = {(o_t, a_t)}_{t=1..T_tau}
```

其中包含：

- internal reflections
- tool calls
- skill pruning
- final submit

### 過濾機制

只保留 final problem 被 verifier 判定有效的軌跡：

```text
I_valid(q) = I[V(q) = 1]
```

形成 `D_SFT` 後，用 behavioral cloning：

```text
L_SFT(theta) = - 1/|D_SFT| sum_{tau in D_SFT} sum_t log pi_theta(a_t | o_t)
```

### 工程意義

這代表 proposer 的 SFT 標的不是最終題目文字，而是整條 agentic trajectory。

## Stage 3: Agentic RL with MGPO

### 目的

把 proposer 從「會模仿」推到「會生成更難、仍可驗證的題目」。

### 為什麼不用標準 GRPO

論文指出長 synthesis chain 的 reward 很稀疏。只靠 trajectory-level reward，對 `think`、`exec`、`edit` 這些中間行為很難做 credit assignment。

### 論文做法

MGPO 同時用：

- trajectory-level advantage
- stage-level advantage

並把兩者融合：

```text
A_fused = A_E(tau_i) + omega * A_S(a_t)
```

### Stage-level 的價值

它讓 Draft、Check、Refine 這些不同 cognitive stages 的良好局部行為能得到 credit，而不是全靠最後 submit 才知道對不對。

## Curriculum 機制

論文不是固定抽 skill 類別，而是追著 proposer 的弱點走。

### mastery-state

對每個 category 維護 proficiency 向量 `m`：

```text
m_c(t+1) = (1 - alpha) m_c(t) + alpha * success_rate_c(t)
```

### sampling rule

下一輪 skill category 抽樣機率：

```text
p(c) ∝ 1 / (m_c + epsilon)
```

也就是越不熟的類別，越常被抽到。

## Layered Reward

### terminal reward

```text
r_T = V(q) * (R_base + I[rho(q) > 0] * lambda * (1 - rho(q)))
```

- `V(q)`: verifier validity
- `rho(q)`: prober 估計的 Pass@k

這設計很重要：

- 無效題目拿不到 reward
- 難度 bonus 只給可解的題目

### intermediate process reward

對成功的工具執行與高品質 reflection 給較稠密的過程獎勵。

## Solver 端訓練

論文有一個很容易被忽略但很重要的設計：

### proposer 與 solver 分工

- proposer 端訓練：吃完整 agentic process
- solver 端訓練：只吃 final `{question, answer}`

論文明確說 downstream solver 不吃 proposer 中間 trace。

這樣做的好處：

1. 減少 solver 對 proposer style 的耦合
2. 避免 trace formatting 成為混淆變數
3. 直接比較資料品質，而不是比較軌跡包裝方式

## 你重建論文時的最低可行版本

### MVP 版

1. skill acquisition
2. teacher 產生 agentic trajectories
3. verifier 過濾
4. agentic SFT
5. 不先上 MGPO，先做 proposer imitation

### 論文核心版

1. skill acquisition
2. agentic SFT
3. curriculum-based skill sampling
4. verifier + prober
5. layered reward
6. MGPO

### 完整論文版

再加上：

- verifier ensemble + second audit
- curriculum of probers
- dynamic skill pruning 統計分析
- downstream solver evaluation across math/code/science