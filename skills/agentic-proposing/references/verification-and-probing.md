# Verifier、Prober 與 Acceptance Rule

## 兩個不同角色

論文把品質控制拆成兩條線：

1. Verifier：檢查題目是否有效、可解、邏輯自洽
2. Prober：估計題目難度是否剛好在 reasoning frontier 上

兩者不可混為一談。

## Verifier Ensemble

論文用三模型 verifier ensemble，加上一個 second-audit reviewer。

### verifier 看到的內容

- proposer 的完整生成 trace
- final problem statement

### verifier 任務

1. 嘗試解題
2. 給出 validity vote `v_i ∈ {0,1}`
3. 提供簡短理由

### acceptance rule

樣本被接受，必須同時滿足：

1. 至少兩個 verifier 投 valid
2. rationale 沒指出 ambiguity / underspecification / unsolvability
3. final answers 一致，或 second audit 可合理裁決差異

## 為什麼 verifier 要看完整 trace

因為 proposer 的中間 skill composition、tool output、refinement 可以暴露潛在邏輯破綻。只看 final statement 可能漏掉生成時已知的問題。

## Prober

prober 不是 verifier。它只做 difficulty estimation。

### 定義

論文用：

```text
rho(q) = Pass@k
```

其中 `k = 16`。

### prober prompt

- 只看 final problem statement
- 明確要求 step-by-step thinking
- 固定 decoding 設定

### 為什麼只看 final problem

因為要量的是 solver 面對該題目的真實成功率，而不是 proposer 的思路輔助。

## curriculum of probers

論文不是永遠用同一個 prober，而是用不同強度模型池：

- Qwen3 1.7B 到 30B
- gpt-oss-20b

並維護 mastery-state EMA。

### 切換規則

每 500 題更新一次 accuracy EMA。

當穩定化後的 accuracy 掉到 `< 30%`，就切到更強的 prober。

這樣可確保 proposer 持續生成落在模型 frontier 附近的題目。

## terminal reward 如何結合 verifier/prober

```text
r_T = V(q) * (R_base + I[rho(q) > 0] * lambda * (1 - rho(q)))
```

解讀：

- verifier 不通過：直接零分
- verifier 通過但題太簡單：difficulty bonus 小
- verifier 通過且可解但仍困難：difficulty bonus 高

## 這套設計帶來的工程收益

1. validity 與 difficulty 被拆開，避免 reward 混亂
2. proposer 不會靠生成怪題來拿到高難度分數
3. 只有可解題才有資格拿難度 bonus
4. proposer 會被推向「難但可解」而不是「亂但看起來難」

## 若要做輕量版重建

### 最低配置

- 1 個強 verifier
- 1 個較小 prober
- rule-based grader

### 論文接近版

- 3 verifier ensemble
- second audit
- prober pool
- EMA mastery-state

## 失敗模式

1. 只有 verifier，沒有 prober
   會得到很多有效但太簡單的題目。

2. 只有 prober，沒有 verifier
   容易把不可解題誤當高難度題。

3. verifier 不看 trace
   可能漏掉 proposer 已暴露的邏輯錯誤。