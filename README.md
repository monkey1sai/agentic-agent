# agentic-agent

以 coding domain 為主的 agentic training pipeline，目標是把教師模型輸出轉成可訓練資料，並在 `google/gemma-4-E4B-it` 上執行保守型 QLoRA 微調與評測。

## 目前包含

- 教師輸出到 agentic JSONL 的資料管線
- coding domain 的批次 mock teacher 資料生成
- train/valid/test SFT 資料切分
- Gemma 4 E4B 的 8GB VRAM 友善 QLoRA 訓練骨架
- base / teacher / student 的簡易 benchmark 評測腳本
- WSL2 / Windows / Docker 的環境文件

## 快速入口

- 專案概覽: README_PROJECT.md
- 快速開始: QUICKSTART.md
- 訓練路線圖: TRAINING_ROADMAP.md
- 環境設定: setup/SETUP.md
- 正式訓練配置: train/training_config.json
- smoke 訓練配置: train/training_config_smoke.json

## 建議流程

1. 先看 QUICKSTART.md 與 setup/SETUP.md 完成環境。
2. 用 train/training_config_smoke.json 先跑 smoke training。
3. 確認流程正常後，再用 train/training_config.json 做正式訓練。
4. 用 src/student_inference.py 和 src/eval_runner.py 比較 teacher / student 結果。

## 資料位置

- 原始 teacher 軌跡: data/raw/
- processed 訓練資料: data/processed/
- benchmark 與推論樣例: examples/ 與 data/eval/

## 目前狀態

- smoke training 已成功驗證
- 正式訓練配置已針對 8GB VRAM 收斂
- API 範例文件已改為佔位符，不應提交真實金鑰或內網位址

## 下一版

`v0.1.1` 的預計改動清單見 tasks/v0.1.1-plan.md。