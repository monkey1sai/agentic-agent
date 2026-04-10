# Share 方法拆解

這份文件把論文方法章節轉成工程語言。

## 問題定義

給定固定 base model `W0`，任務序列依序到來。每個時間點你會拿到：

1. 任務資料 `S_t`
2. 或現成 LoRA adapter `ΔW_t = B_t A_t`
3. 或兩者混合

限制是：

- 不能重放舊資料
- 不想線性增加 task-specific adapters
- 希望保留舊知識並吸收新知識

## 核心表示法

標準 LoRA：

```text
ΔW = B A
B ∈ R^(n×r)
A ∈ R^(r×d)
```

Share 改寫成：

```text
ΔW_t ≈ (beta_t epsilon_beta_t) (alpha_t epsilon_alpha_t)^T
beta_t ∈ R^(n×k)
alpha_t ∈ R^(d×k)
epsilon_beta_t, epsilon_alpha_t ∈ R^(k×p)
```

直覺是把原本 task-specific 的 rank-`r` adapter，轉成：

- 跨任務共享的 principal basis
- 每任務很小的 coefficient

## Step 1: Initialization

### 目的

從已有的 LoRA adapters 抽出 foundational shared subspace。

### 做法

對每一層分別做：

1. 蒐集該層所有歷史 `B_i`，堆疊成 `B_t = [B_1, ..., B_t]`
2. 蒐集該層所有歷史 `A_i`，堆疊成 `A_t = [A_1, ..., A_t]`
3. 對 `B_t`、`A_t` 各自做 mean-centering
4. 對中心化結果做 SVD
5. 保留 top-`k` 奇異向量作為 principal factors `beta_t` 與 `alpha_t`
6. 凍結 principal factors，只學任務係數 `epsilon`

### 關鍵性質

- 這步主要是 data-free、gradient-free
- 如果沒有既有 LoRA，可先用單一初始任務訓一個 LoRA 來冷啟動

## Step 2: Continual Adaptation

### 目的

吸收新任務知識，但先不破壞既有 foundational subspace。

### 做法

建立 temporary factors：

```text
beta_t→t+1 = beta_t[:phi]
alpha_t→t+1 = alpha_t[:phi]
epsilon_temp ~ N(0, sigma^2)
```

然後只針對：

- temporary factors
- temporary coefficients

進行訓練。

### 重點

- `phi << k`
- 這是一個小而可控的暫時擴張
- 其目的是識別新任務需要的額外方向

## Step 3: Merging and Reprojection

### 目的

把舊知識與新知識統一收納到單一共享子空間。

### 做法

1. 用舊 principal basis 與舊係數重建每個舊任務的近似 adapter
2. 將所有重建後的舊 adapter 與新 temporary adapter 堆疊
3. 重新做 top-`k` SVD，取得新的 principal basis
4. 用 pseudoinverse / projection 計算所有任務在新 basis 下的係數

對 `B` 而言：

```text
B_hat^(t+1) = [B^1, ..., B^t, beta_t→t+1 epsilon_beta_t+1]
U_k Sigma_k V_k^T = SVD(B_hat^(t+1))[:k]
beta_(t+1) = U_k
epsilon_beta_i = ((beta_(t+1)^T beta_(t+1))^-1) beta_(t+1)^T B^i
```

若 basis 已正交，係數可簡化成：

```text
epsilon_beta_i = beta_(t+1)^T B^i
```

`A` 的更新完全對稱。

## Share-full

Share-full 是合併後再對係數做額外微調的版本。

適用時機：

- 你可以接受較弱的 continual 假設
- 願意為了多一點性能，做一點額外梯度更新

## 與標準 LoRA 的本質差異

標準 LoRA：

- 每個 task 自己一套 adapter
- 任務數越多，儲存越大
- 沒有顯式機制整合共享子空間

Share：

- 所有 task 共享 principal basis
- 每個 task 只保留小係數
- 新任務可透過共享子空間帶來 forward/backward transfer

## 與 O-LoRA / prompt-based continual learning 的差異

- O-LoRA 偏向用正交子空間避免干擾，但知識共享較弱
- prompt-based 方法通常仍需為每任務保存額外狀態
- Share 同時追求 replay-free、共享子空間與低儲存成本