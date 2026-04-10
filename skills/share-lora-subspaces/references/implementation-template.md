# Share 實作模板

這不是完整論文重現，而是工程原型骨架。

## 核心資料結構

```python
from dataclasses import dataclass
import torch


@dataclass
class ShareLayerState:
    alpha: torch.Tensor
    beta: torch.Tensor
    eps_alpha: dict[str, torch.Tensor]
    eps_beta: dict[str, torch.Tensor]


@dataclass
class ShareConfig:
    lora_rank: int
    k: int
    p: int
    phi: int
    explained_variance_threshold: float = 0.6
```
```

## 初始化

```python
def init_share_from_loras(a_factors: list[torch.Tensor], b_factors: list[torch.Tensor], k: int):
    stacked_a = torch.cat(a_factors, dim=0)
    stacked_b = torch.cat(b_factors, dim=0)

    centered_a = stacked_a - stacked_a.mean(dim=0, keepdim=True)
    centered_b = stacked_b - stacked_b.mean(dim=0, keepdim=True)

    _, _, va = torch.linalg.svd(centered_a, full_matrices=False)
    ub, _, _ = torch.linalg.svd(centered_b, full_matrices=False)

    alpha = va[:k].transpose(0, 1).contiguous()
    beta = ub[:, :k].contiguous()
    return alpha, beta
```

## 用共享子空間重建 adapter

```python
def reconstruct_adapter(alpha, beta, eps_alpha, eps_beta):
    left = beta @ eps_beta
    right = alpha @ eps_alpha
    return left @ right.T
```

## Temporary Expansion

```python
def init_temporary_factors(alpha, beta, phi, p, std=0.02):
    temp_alpha = alpha[:, :phi].clone()
    temp_beta = beta[:, :phi].clone()
    eps_alpha = torch.randn(phi, p) * std
    eps_beta = torch.randn(phi, p) * std
    return temp_alpha, temp_beta, eps_alpha, eps_beta
```

## Merge and Reproject

```python
def merge_share_basis(old_reconstructed: list[torch.Tensor], new_component: torch.Tensor, k: int):
    merged = torch.cat(old_reconstructed + [new_component], dim=1)
    u, s, vh = torch.linalg.svd(merged, full_matrices=False)
    basis = u[:, :k]
    coeffs = torch.diag(s[:k]) @ vh[:k, :]
    return basis, coeffs
```

## 係數重投影

```python
def project_coefficients(basis: torch.Tensor, target: torch.Tensor):
    if torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], device=basis.device), atol=1e-5):
        return basis.T @ target
    pinv = torch.linalg.pinv(basis)
    return pinv @ target
```

## 單層 Share 更新程序

```python
def update_share_layer(layer_state, incoming_data_or_adapter, config):
    temp_alpha, temp_beta, temp_eps_alpha, temp_eps_beta = init_temporary_factors(
        layer_state.alpha,
        layer_state.beta,
        config.phi,
        config.p,
    )

    # 這裡接你的訓練 loop，只更新 temporary factors 與 temp coefficients
    trained_component = (temp_beta @ temp_eps_beta) @ (temp_alpha @ temp_eps_alpha).T

    old_components = []
    for task_id in layer_state.eps_alpha:
        old_components.append(
            reconstruct_adapter(
                layer_state.alpha,
                layer_state.beta,
                layer_state.eps_alpha[task_id],
                layer_state.eps_beta[task_id],
            )
        )

    new_beta, beta_coeff_matrix = merge_share_basis(old_components, trained_component, config.k)
    layer_state.beta = new_beta
    return layer_state
```

## 原型實作注意事項

1. 對每一層獨立維護 `alpha/beta`
2. `A` 與 `B` 通常要分開做 SVD 與投影
3. 真正接 Hugging Face PEFT 時，需先明確定義每層 adapter factor 的提取格式
4. 若要在 8GB VRAM 上運作，先做 adapter-space 原型，再碰端到端訓練
5. Share 的合併步驟很適合放在 CPU 執行，減少 VRAM 壓力