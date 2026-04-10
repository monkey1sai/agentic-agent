# 訓練環境快速設定指南

## 問題現況

目前 Windows 本機的 Python 3.13 環境缺少 ML 必要套件，無法運行 Gemma QLoRA 訓練。

## 推薦方案

### 選項 A：WSL2 + Ubuntu（最穩定）

**前提**：已安裝 WSL2，可用 Ubuntu 20.04/22.04

```bash
# 1. 進入 WSL 終端
wsl

# 2. 安裝 Python 3.11 與系統依賴
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip-python3.11 git

# 3. 建立虛擬環境
python3.11 -m venv ~/agentic-ml
source ~/agentic-ml/bin/activate

# 4. 複製專案到 WSL
# 例：cp -r /mnt/c/.llmcode/agentic-agent ~/agentic-agent
# 或改用 ln -s /mnt/c/.llmcode/agentic-agent ~/agentic-agent

# 5. 安裝依賴
cd ~/agentic-agent
pip install -r setup/requirements.txt

# 6. 驗證安裝
python train/run_qlora.py --config train/training_config.json --dry-run
```

### 選項 B：Windows 原生 + 獨立 Python 3.11 虛擬環境（次佳）

**注意**：Windows 上不推薦 bitsandbytes 與某些依賴，但可先測試。

```powershell
# 1. 下載 Python 3.11 portable 或用 Windows 原生安裝
python3.11 --version

# 2. 建立虛擬環境
python3.11 -m venv .\.venv-ml
.\.venv-ml\Scripts\Activate.ps1

# 3. 升級 pip
python -m pip install --upgrade pip

# 4. 安裝依賴（可能需要編譯，視 VS Build Tools 而定）
pip install torch transformers datasets peft trl accelerate

# （可跳過 bitsandbytes 在 Windows 上不穩定）

# 5. 驗證
python train/run_qlora.py --config train/training_config.json --dry-run
```

### 選項 C：Docker 容器（完全隔離）

如果有 Docker 環境，可使用官方 PyTorch 基礎映像：

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y git
WORKDIR /workspace
COPY . .
RUN pip install -r setup/requirements.txt
CMD ["/bin/bash"]
```

## 驗證安裝

所有方案完成後，都應執行：

```bash
python train/run_qlora.py --config train/training_config.json --dry-run
```

若 `"python_ready": true`，表示環境已就緒。

## 磁盤與顯存需求

- **磁盤**：至少 10 GB（model weights + checkpoints）
- **VRAM**：RTX 4060 Ti 8GB，建議初期數據集大小 < 100 samples
- **RAM**：至少 32 GB 系統 RAM（預設 gradient_accumulation_steps=16）

## 故障排查

### ImportError: No module named 'torch'

確認虛擬環境已激活：

```bash
# Linux/WSL
source ~/agentic-ml/bin/activate

# Windows
.\.venv-ml\Scripts\Activate.ps1
```

### bitsandbytes 無法編譯（Windows）

跳過 bitsandbytes，改用標準精度訓練（會消耗更多 VRAM）。編輯 `train/training_config.json` 移除 `"load_in_4bit": true`。

### CUDA 驅動版本不符

確認 NVIDIA 驅動版本 >= 550，與 PyTorch CUDA 12.1 兼容。

```bash
nvidia-smi
```

## 下一步

1. 選擇上述任一方案並安裝
2. 執行 dry-run 驗證
3. 使用 `src/main_generate_dataset.py` 生成大規模 coding 資料
4. 啟動訓練：`python train/run_qlora.py --config train/training_config.json`
