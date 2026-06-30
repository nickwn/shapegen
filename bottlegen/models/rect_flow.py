"""Rectified flow over the frozen VAE latent set.

A DiT over the ``M`` latent tokens predicts a velocity field; the ODE
``dz/dt = v(z, t)`` transports a Gaussian prior (``t=0``) to the data latent
distribution (``t=1``), integrated with Heun steps. Generation: sample
``z0 ∼ N(0, I)``, integrate to ``z1``, decode ``vae.dec(z1) → σ̂``, robust-decode
to a mesh.

Two velocity targets:

* **vanilla** rectified flow — straight-line interpolation
  ``z_t = (1−t)z0 + t·z1`` with target ``z1 − z0``.
* **physics-informed** — pull the analytic σ̂-space reverse-blur velocity
  ``+κ_ℓ σ̂`` back into latent space through the **differentiable VAE decoder**
  (a vector–Jacobian product), so the flow's velocity is biased toward the
  reverse Willmore-flow direction.

The plan's full physics target pushes the reverse velocity through the *encoder*
(``encode_jvp``), which would differentiate the σ̂→mesh→latent path; the Dirac
decode in that path is a NumPy/SciPy eigensolve and is not autograd-able, so the
faithful version is deferred. The decoder-VJP target below captures the same
intent (reverse-Willmore direction) using the differentiable decoder we have.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AdaLNBlock, timestep_embedding

__all__ = ["LatentDiT", "heun_sample", "vanilla_target",
           "decoder_vjp_velocity"]


class LatentDiT(nn.Module):
    """Time-conditioned DiT over ``M`` latent tokens → per-token velocity."""

    def __init__(self, M, zdim, d=256, depth=6, heads=8, t_embed=128):
        super().__init__()
        self.M, self.zdim, self.t_embed = M, zdim, t_embed
        self.tok = nn.Linear(zdim, d)
        self.pos = nn.Parameter(torch.randn(M, d) * 0.02)
        self.tproj = nn.Sequential(nn.Linear(t_embed, d), nn.SiLU(),
                                   nn.Linear(d, d))
        self.blocks = nn.ModuleList([AdaLNBlock(d, heads) for _ in range(depth)])
        self.untok = nn.Linear(d, zdim)
        nn.init.zeros_(self.untok.weight)
        nn.init.zeros_(self.untok.bias)

    def forward(self, z, t):                                # z [B,M,zdim], t [B]
        c = self.tproj(timestep_embedding(t, self.t_embed))
        h = self.tok(z) + self.pos
        for blk in self.blocks:
            h = blk(h, c)
        return self.untok(h)                                # [B, M, zdim]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def heun_sample(model, z0, n_steps=64):
    """Integrate ``dz/dt = v(z, t)`` from ``t=0`` (``z0``) to ``t=1`` with Heun
    (improved Euler) steps. Returns ``z1``."""
    z = z0
    ts = torch.linspace(0, 1, n_steps + 1, device=z0.device)
    B = z0.shape[0]
    for k in range(n_steps):
        t0 = ts[k].expand(B)
        t1 = ts[k + 1].expand(B)
        dt = (ts[k + 1] - ts[k])
        v0 = model(z, t0)
        v1 = model(z + dt * v0, t1)
        z = z + dt * 0.5 * (v0 + v1)
    return z


def vanilla_target(z1, rng=None):
    """Vanilla RF sample: ``(zt, t, target=z1−z0)`` with ``z0 ∼ N(0,I)``."""
    g = None
    if rng is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(rng.integers(0, 2 ** 31)))
    z0 = torch.randn(z1.shape, generator=g).to(z1.device)
    t = torch.rand(z1.shape[0], generator=g).to(z1.device)
    tb = t.reshape(-1, 1, 1)
    zt = (1 - tb) * z0 + tb * z1
    return zt, t, (z1 - z0)


def decoder_vjp_velocity(decoder, zt, kappa_packed):
    """Physics-informed latent velocity: the σ̂-space reverse-blur velocity
    ``r = κ ⊙ dec(z)`` pulled back through the decoder Jacobianᵀ,
    ``v = J_dec(z)ᵀ r = ∇_z ⟨dec(z), r⟩``. Differentiable, one backward pass."""
    z = zt.detach().requires_grad_(True)
    sig = decoder(z)                                        # [B, dim]
    r = (kappa_packed.reshape(1, -1) * sig).detach()        # reverse velocity
    v = torch.autograd.grad((sig * r).sum(), z)[0]          # [B, M, zdim]
    return v.detach()
