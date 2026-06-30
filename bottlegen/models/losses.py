"""Training losses for the Simple model.

``sobolev_loss`` is a per-degree weighted MSE on the x0 (clean-state) prediction:
each packed element is weighted by ``(1+ℓ(ℓ+1))^β`` and optionally whitened by
``1/perdeg_std_ℓ²`` (see :func:`bottlegen.models.pack.sobolev_packed`).
"""

from __future__ import annotations

import numpy as np
import torch

from .pack import sobolev_packed

__all__ = ["sobolev_loss", "make_weight"]


def make_weight(L: int, beta: float = 1.0, perdeg_std=None, device=None,
                dtype=torch.float32) -> torch.Tensor:
    """Packed loss-weight tensor on ``device``."""
    w = sobolev_packed(L, beta, perdeg_std)
    w = w / w.mean()                       # keep the loss O(1)
    return torch.as_tensor(w, dtype=dtype, device=device)


def sobolev_loss(pred: torch.Tensor, target: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE ``mean_b Σ_i w_i (pred − target)²_i / Σ_i w_i``."""
    se = (pred - target) ** 2
    return (se * weight.reshape(1, -1)).sum(-1).mean() / weight.sum()
