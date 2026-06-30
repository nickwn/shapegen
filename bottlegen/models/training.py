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

__all__ = ["make_model", "pack_batch", "fit_simple", "fit_vae",
           "vae_latents", "fit_rect_flow"]


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


def fit_vae(meshes, sigma_targets, L, *, k=64, model=None, vae_cfg=None,
            steps=500, batch=8, lr=1e-3, weight_decay=1e-4, beta=1e-4,
            beta_warmup=200, sobolev_beta=1.0, whiten=False, seed=0,
            device=None, log_every=0):
    """Train the set-latent VAE to autoencode meshes → σ̂. Returns
    ``(model, ops, history)`` where ``ops`` are the cached torch operators and
    ``history`` is a list of ``(loss, recon, kl)``."""
    from ..geom.diffusion_ops import mesh_operators
    from .diffusionnet import ops_to_torch
    from .set_vae import SetVAE, vae_loss

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ops = [ops_to_torch(mesh_operators(m.vertices, m.faces, k), device)
           for m in meshes]
    tgt_complex = [state.gauge_fix(np.asarray(s, complex)) for s in sigma_targets]
    targets = torch.as_tensor(
        np.stack([state.real_pack(s) for s in tgt_complex]).astype(np.float32),
        device=device)
    stats = state.perdeg_std(np.stack(tgt_complex))
    weight = make_weight(L, sobolev_beta, stats if whiten else None,
                         device=device)
    if model is None:
        model = SetVAE(L=L, **(vae_cfg or {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    n = len(meshes)
    history = []
    model.train()
    for step in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        ops_b = [ops[i] for i in idx]
        b = beta * min(1.0, step / max(1, beta_warmup))      # KL anneal
        pred, mu, lv = model(ops_b, sample=True)
        loss, recon, kl = vae_loss(pred, targets[idx], mu, lv, weight, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append((loss.item(), recon.item(), kl.item()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:5d}  loss {loss.item():.3e}  "
                  f"recon {recon.item():.3e}  kl {kl.item():.3e}  beta {b:.1e}")
    return model, ops, history


@torch.no_grad()
def vae_latents(vae, ops_list, device=None):
    """Frozen-encoder latents (deterministic μ) for a list of mesh ops:
    ``[N, M, zdim]``."""
    vae.eval()
    mus = [vae.encode_one(o)[0] for o in ops_list]
    return torch.stack(mus).to(device) if device else torch.stack(mus)


def fit_rect_flow(z1_data, *, mode="vanilla", decoder=None, kappa_packed=None,
                  model=None, rf_cfg=None, steps=2000, batch=64, lr=1e-3,
                  weight_decay=1e-4, ema_decay=0.999, normalize_physics=True,
                  seed=0, device=None, log_every=0):
    """Train a :class:`LatentDiT` rectified flow over frozen-VAE latents
    ``z1_data`` ``[N, M, zdim]``.

    ``mode='vanilla'`` matches ``z1−z0``; ``mode='physics'`` matches the
    decoder-VJP reverse-blur velocity (needs ``decoder`` and ``kappa_packed``).
    Returns ``(model, ema, history)``.
    """
    from .cold_diffusion import EMA
    from .rect_flow import LatentDiT, decoder_vjp_velocity, vanilla_target

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    z1_data = z1_data.to(device)
    N, M, zdim = z1_data.shape
    if model is None:
        model = LatentDiT(M, zdim, **(rf_cfg or {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    ema = EMA(model, ema_decay)
    history = []
    model.train()
    for step in range(steps):
        idx = rng.integers(0, N, size=min(batch, N))
        z1 = z1_data[idx]
        zt, t, target = vanilla_target(z1, rng)
        if mode == "physics":
            target = decoder_vjp_velocity(decoder, zt, kappa_packed)
            if normalize_physics:
                target = target / (target.flatten(1).norm(dim=1)
                                   .reshape(-1, 1, 1) + 1e-8)
        v = model(zt, t)
        loss = ((v - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        history.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:5d}  loss {history[-1]:.3e}")
    return model, ema, history
