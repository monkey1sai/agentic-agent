# Skill Acquisition 與技能包設計

## Stage 1 的角色

論文第一階段不是直接產生題目，而是先建立 `K_self`，也就是 autonomous skill library。

這是後續 proposer 的 prior knowledge。

## 技能表示

每個 skill 被定義成：

```text
k = <iota, mu, delta, tau>
```

- `iota`: reasoning intent
- `mu`: construction method
- `delta`: difficulty effect
- `tau`: tool-use hint

### 工程解讀

你可以把它對應成：

```json
{
  "intent": "這個技能要啟動哪種推理骨架",
  "method": "怎麼構造題目或解題條件",
  "difficulty_effect": "會把題目變難在哪裡",
  "tool_hint": "要搭配哪些檢查工具"
}
```

## Skill Acquisition 流程

### 1. 蒐集來源 corpus

`D_corpus` 可以是：

- 高難度題庫 Q&A
- 教材與理論文本
- 研究論文
- 已知可驗證題目集

### 2. Teacher 萃取候選技能

teacher policy 先產生 `K_cand`。

### 3. 技能打分與 rejection sampling

論文使用 quality score `r(k)` 與 threshold `tau_r` 做過濾：

```text
p_skill(k) = I[r(k) >= tau_r] * pi_teacher(k | D_corpus) / normalization
```

### 4. 用 skill acquisition loss 建立 proposer 的技能知識

```text
L_skill-acq(theta) = - E_{k ~ p_skill}[log pi_theta(k | D_corpus)]
```

## Dynamic Skill Pruning

這是論文很有價值的設計。

### 機制

在 drafting 階段，agent 可以先 `think` 檢查目前 skill set 是否矛盾或偏題。如果發現某 skill 可能帶來邏輯錯誤，就直接 `edit` 把它從 `K_t` 刪掉。

### 作用

- 把錯誤阻止在生成源頭
- 降低多技能組合時的邏輯衝突
- 提高最終 verifier 通過率

論文附錄中報告：

- 動態 pruning 約出現在 14.5% 軌跡
- 有 pruning 時問題有效率明顯高於固定 skill set

## 論文中的技能包觀念

論文 Section 11 把 skill 做成 filesystem package，而不是把所有內容塞在單一 prompt。

### Level 1: Discovery Metadata

用 YAML frontmatter 暴露技能的發現資訊。

### Level 2: Procedural Instructions

在 `SKILL.md` body 中放 recognition、workflow、effect。

### Level 3: External Utility

搭配真實 script 進行外部驗證，例如數學 evaluator。

## Progressive Disclosure

這篇論文跟 skill-based agent 設計最契合的一點，就是 progressive loading：

1. Draft 階段先只讀 metadata
2. 真需要時再讀 `SKILL.md` 詳細流程
3. 最後在 verify 階段執行外部 script

這比把全部知識一次塞進 prompt 更可擴充。

## 建 skill library 的實務規則

1. 每個 skill 必須可組合
2. 每個 skill 要說清楚會增加哪種難度
3. 每個 skill 要知道何時需要工具驗證
4. 每個 skill 最好能單獨被 verifier 或 evaluator 測試
5. skill 名稱與 metadata 要讓 proposer 容易搜尋

## 建議的本地 skill package 格式

```text
skills/
  math-complex-generalized-cauchy/
    SKILL.md
    scripts/
      cauchy_evaluator.py
```

其中 `SKILL.md` 應該至少有：

- 結構辨識
- 核心操作流程
- 對最終題目或解法的 effect
- 工具提示