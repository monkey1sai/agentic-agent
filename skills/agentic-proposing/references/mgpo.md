# MGPO 解釋

MGPO 是論文專門為 proposing agent 設計的 RL 方法，全名是 Multi-Granularity Policy Optimization。

## 為什麼需要 MGPO

標準 trajectory-level RL 在長鏈 synthesis 問題上有兩個問題：

1. reward 太稀疏
2. 中間正確行為拿不到足夠 credit

例如：

- 一次有效的 `think`
- 一次關鍵的 `exec`
- 一次成功刪掉衝突 skill 的 `edit`

如果只看最後 `submit` 的 reward，這些行為很容易被淹沒。

## KL-constrained 目標

論文先從帶 KL 約束的 reward maximization 出發：

```text
max_pi E[R(o, a)] - beta * D_KL[pi_theta(.|o) || pi_ref(.|o)]
```

這表示新 policy 需要：

- 追求高 reward
- 但不能離 reference policy 太遠

## 隱式獎勵與 zero-sum weighting

論文定義 implicit reward：

```text
R_theta(a|o) = beta * log(pi_theta(a|o) / pi_ref(a|o))
```

再用 centered advantage 與 centered implicit reward 構造 sample weight：

```text
w(o, a) = (A(o, a) - mean(A)) - (R_theta(o, a) - mean(R_theta))
```

這樣做的好處是讓 partition function 被抵消，得到可實作的訓練權重。

## 兩層 advantage

### 1. Trajectory-level Advantage

```text
A_E(tau_i) = (r_T(i) - mean(r_T)) / sigma_rT
```

對整條軌跡的最終品質打分。

### 2. Stage-level Advantage

```text
A_S(a_t) = (r_proc_t - mean(r_proc)) / sigma_rproc
```

在同一 cognitive stage subgroup 內標準化，讓局部行為被公平比較。

## Fused Advantage

兩者融合：

```text
A_fused = A_E + omega * A_S
```

論文 sensitivity study 顯示 `omega = 0.5` 最平衡。

## Asymmetric Gate

論文沒有直接把 `w` 拿來更新，而是用 hyperbolic secant gate 稳定 off-policy/noisy updates：

```text
w' = sech^2((tau_it / 2) * (r_i,t(theta) - 1)) * w
```

且：

- 正權重用 `tau_pos`
- 負權重用 `tau_neg`
- `tau_neg > tau_pos`

這代表對負向、不穩定更新更保守。

## 最終目標

```text
L_MGPO(theta) = - 1/N sum_{i,t,j} w'_{i,t} log pi_theta(x_{i,t,j} | o_t^(i), a_{t,<j}^{(i)})
```

本質上是 token-normalized weighted log-likelihood，但權重來自 multi-granularity RL credit assignment。

## 這套方法為什麼對 proposer 特別有效

因為 proposer 是多步驟 agent，不是單步 answerer。

需要 credit 的行為包含：

- skill selection
- skill pruning
- checking
- tool execution
- refine 的回退決策

MGPO 剛好就是為這種情境做的。

## 論文中的實證訊號

- Full MGPO 比標準 trajectory-only GRPO 明顯好
- 移除 stage-level advantage 會掉很多
- 移除或簡化 gate 也會掉 proposing accuracy
- proposer quality 可達約 93.4% verifier-accepted problem accuracy

## 重建時最容易做錯的地方

1. 只保留 final reward
2. 沒有按 cognitive stage 做 subgroup 標準化
3. 把 prober 難度訊號直接當 correctness 用
4. 沒有 KL anchor 到 SFT reference policy