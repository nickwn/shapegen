"""Reusable cold-diffusion training loop for the Simple model.

Kept in the package (not the repo-root ``train/`` script) so tests and the CLI
share one implementation. Batches are drawn from an in-memory stack of complex
σ̂ with optional free Wigner-D rotation augmentation, then real-packed.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import random_so3, wigner_rotate_sigma
from ..flow import state
from .cold_diffusion import EMA, ColdDiffusion
from .losses import make_weight, sobolev_loss
from .pack import PackSpec
from .sh_dit import SHDiT

__all__ = ["make_model", "pack_batch", "fit_simple"]


def make_model(L, model_cfg=None, device=None):
    cfg = dict(d=384, depth=12, heads=6, t_embed=256)
    if model_cfg:
        cfg.update({k: model_cfg[k] for k in cfg if k in model_cfg})
    model = SHDiT(L, **cfg)
    if device is not None:
        model = model.to(device)
    return model


def pack_batch(sigma_stack, idx, L, rotate=False, rng=None):
    """Real-pack a batch of complex σ̂ (optionally Wigner-rotated) -> ``[B, dim]``."""
    rng = np.random.default_rng() if rng is None else rng
    rows = []
    for i in idx:
        s = state.gauge_fix(np.asarray(sigma_stack[i], dtype=complex))
        if rotate:
            s = wigner_rotate_sigma(s, random_so3(rng))
        rows.append(state.real_pack(s))
    return np.stack(rows).astype(np.float32)


def fit_simple(sigma_stack, L, t_max, *, model=None, model_cfg=None,
               steps=2000, batch=64, lr=3e-4, weight_decay=1e-4,
               ema_decay=0.999, sobolev_beta=1.0, whiten=True,
               rotate_aug=False, warmup=100, seed=0, device=None,
               log_every=0):
    """Train the Simple cold-diffusion DiT. Returns ``(model, ema, history)``."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if model is None:
        model = make_model(L, model_cfg, device)
    cdiff = ColdDiffusion(L, t_max, device=device)
    stats = state.perdeg_std(np.stack([np.asarray(s) for s in sigma_stack]))
    weight = make_weight(L, sobolev_beta, stats if whiten else None,
                         device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * p))         # cosine to 0

    ema = EMA(model, ema_decay)
    n = len(sigma_stack)
    history = []
    model.train()
    for step in range(steps):
        idx = rng.integers(0, n, size=min(batch, n) if n < batch else batch)
        x0 = torch.as_tensor(
            pack_batch(sigma_stack, idx, L, rotate_aug, rng), device=device)
        for g in opt.param_groups:
            g["lr"] = lr * lr_at(step)
        loss = cdiff.loss_step(model, x0, weight, sobolev_loss, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        history.append(float(loss.detach().cpu()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:5d}  loss {history[-1]:.3e}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}")
    return model, ema, history
