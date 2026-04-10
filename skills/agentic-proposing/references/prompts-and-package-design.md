# Prompt 模板與 Skill Package 設計

## 論文在 prompt 層做了什麼

附錄 Section 10 提供了兩類 Skill Acquisition prompt：

1. Structured Q&A Extraction Prompt
2. Unstructured Corpus Synthesis Prompt

它們的目的不是直接生成題目，而是把知識來源 reverse-engineer 成可被 proposer 使用的 agent skills。

## 這代表什麼

論文的 prompt 工程不是單純寫一段 mega prompt，而是：

- 先把知識變成模組化 skill packages
- 再讓 proposer 依需求按需載入

## 技能包的三層設計

論文 Case Study 很明確地展示 skill package 應該長怎樣。

### Level 1: Discovery Metadata

放在 `SKILL.md` frontmatter。

用途：

- 讓 proposer 在 Draft 階段快速判斷這個 skill 能不能用
- 控制 on-demand loading

### Level 2: Procedural Instructions

放在 `SKILL.md` 內文。

至少要有：

- Structure / Recognition
- Action / Workflow
- Effect / Target Outcome
- Tool hints

### Level 3: External Utility

放在 script，例如 evaluator。

用途：

- 在 Check / Refine 階段做外部驗證
- 不是只靠 LLM 自我感覺良好

## 論文示範的 agent interaction logic

1. Draft 階段只先讀 metadata
2. 確認題目需要某 skill 後，再讀完整 SKILL.md
3. 在驗證階段執行外部 script
4. 若驗證失敗，進入 Refine，再由 `think` 重新規劃

## 套用到你的工作區

如果你要照這篇論文做 skill-driven proposer，建議 skill package 目錄長這樣：

```text
skills/
  math-xxx/
    SKILL.md
    scripts/
      evaluator.py
  code-xxx/
    SKILL.md
    tests/
      verifier.py
```

## 建議的 frontmatter 欄位

```yaml
---
name: math-complex-generalized-cauchy
description: >
  Use when users request contour integral synthesis, generalized Cauchy formula,
  or proof tasks involving high-order poles.
metadata:
  domains: [math, complex-analysis]
  difficulty_effect: high
  tool_hint: symbolic-check
---
```

## 建議的 SKILL.md body 結構

```markdown
## Structure
辨識題型、適用前提、不可違反條件

## Action
核心操作流程

## Effect
這個 skill 會如何改變題目或解法

## Tooling
哪些工具應該被呼叫來驗證

## Failure Cases
哪些情況下這個 skill 會導致題目失效
```

## 對 proposer 的價值

這種 package 化設計讓 proposer：

1. 能做 progressive disclosure，節省 context
2. 能在 Draft/Refine 階段動態選 skill
3. 能把 verifier 與 tool-use 綁到 skill 本身
4. 能把 complex knowledge 轉成可執行 capability