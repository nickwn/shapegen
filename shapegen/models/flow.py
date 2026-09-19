"""Flow matching (rectified flow) over code vectors.

Learns a **velocity field** that transports Gaussian noise to the data
distribution, then samples by integrating it. Nothing here knows what the codes
mean — it sees an ``(N, K)`` array — so the representation is chosen in
:mod:`shapegen.models.basis` and swapped there.

The method, in one paragraph. Pick a data point ``z1`` and a noise sample
``z0 ~ N(0, I)``, join them with a straight line ``z_t = (1-t) z0 + t z1``, and
note that the velocity along *that* line is the constant ``z1 - z0``. Train by
regression::

    minimise  E_{t, z0, z1}  || v_θ(z_t, t) − (z1 − z0) ||²

Any given ``z_t`` is reachable from many ``(z0, z1)`` pairs, so this target is
extremely noisy — but least-squares regression converges to the *conditional
expectation* of its target, and ``E[z1 − z0 | z_t = z]`` is exactly the marginal
velocity field whose ODE carries ``N(0, I)`` to the data. That is the whole
trick: you obtain a field you cannot write down, from a target you can. The loss
is a plain MSE — no adversarial game, no ELBO, no noise schedule.

Because the training paths are straight, the learned field is nearly straight and
20-50 Euler steps suffice; diffusion models are the same object with a curved
variance-preserving path and typically need far more.

**What this does not do:** it learns whatever distribution the data has. If the
inputs are not orientation-canonicalised, the model learns the orientation
distribution as if it were shape. Whitening does not touch that.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["VelocityField", "train_flow", "sample_flow", "gaussian_baseline"]


class VelocityField(nn.Module):
    """MLP velocity field ``v_θ(z, t)``, ``(B,K) x (B,) -> (B,K)``.

    Time enters as a Fourier feature embedding rather than a raw scalar: a single
    appended ``t`` is one input among K, and the network tends to ignore it,
    which silently collapses the flow to a single average direction.
    """

    def __init__(self, k: int, width: int = 512, depth: int = 4, n_freq: int = 16):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * np.pi)
        layers, d_in = [], k + 2 * n_freq
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU()]
            d_in = width
        layers += [nn.Linear(width, k)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        a = t[:, None] * self.freqs[None, :]
        return self.net(torch.cat([z, a.sin(), a.cos()], dim=1))


def train_flow(Z: np.ndarray, steps: int = 20000, batch: int = 256,
               width: int = 512, depth: int = 4, lr: float = 1e-3,
               sigma: float = 0.0, seed: int = 0, verbose: bool = False,
               device: str = "cpu"):
    """Fit a :class:`VelocityField` to codes ``Z`` ``(N, K)``.

    Returns ``(model, history)``. ``history`` is the running loss, which is
    *not* a quality measure: its floor is set by the irreducible variance of the
    conditional target, not by how well the model fits. Judge samples, not loss.

    An underfit field shows up as **under-dispersed samples** — it has learned the
    average direction toward the data but not the spread, so it collapses toward
    the mean. Check the per-dimension std of the samples, not the loss.

    Cosine decay to zero, because the conditional target stays noisy forever: at a
    fixed learning rate the parameters keep bouncing at a scale set by that noise
    and the field never sharpens.

    ``sigma`` adds fresh Gaussian noise to each drawn ``z1`` — a **variance floor**
    on the target. Use it when the data is near-singular: N shapes span at most
    N-1 directions of a K-dimensional code space, and if the intrinsic
    dimensionality is lower still (ρ̂ of a face set is a ~30-dimensional identity
    manifold inside 169 coefficients) the target has near-zero variance in most
    directions. Flow matching onto such a distribution is **ill-posed**, not merely
    data-starved: the field must collapse mass onto a measure-zero set, so it
    blows up as ``t → 1``. Convolving with ``N(0, σ²)`` makes the target
    non-singular at a known cost — the samples are blurred by exactly σ. Diffusion
    models do this implicitly. ``σ ≈ 0.05`` in whitened units is a reasonable
    starting point; 0 (the default) leaves the target untouched.
    """
    torch.manual_seed(seed)
    Zt = torch.as_tensor(np.asarray(Z, dtype=np.float32), device=device)
    model = VelocityField(Zt.shape[1], width, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    hist = []
    for it in range(steps):
        z1 = Zt[torch.randint(len(Zt), (batch,), device=device)]
        if sigma > 0.0:
            z1 = z1 + sigma * torch.randn_like(z1)   # variance floor, see above
        z0 = torch.randn_like(z1)
        t = torch.rand(batch, device=device)
        zt = (1 - t[:, None]) * z0 + t[:, None] * z1
        loss = ((model(zt, t) - (z1 - z0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if verbose and (it + 1) % max(1, steps // 10) == 0:
            print(f"  step {it + 1:6d}/{steps}  loss {np.mean(hist[-500:]):.4f}",
                  flush=True)
    return model, np.array(hist)


@torch.no_grad()
def sample_flow(model, n: int, k: int, steps: int = 50, t_end: float = 1.0,
                rescale: bool = True, seed: int = 0,
                device: str = "cpu") -> np.ndarray:
    """Integrate ``dz/dt = v_θ(z, t)`` from ``N(0, I)`` at t=0 to data at t=1.

    Plain forward Euler with uniform steps. Straight training paths make the
    learned field nearly straight, so the integration error is small; if samples
    improve materially past ~50 steps, the field is more curved than it should be
    and the model is underfit.

    ``t_end < 1`` stops short of the endpoint, the companion trick to ``train_flow``'s
    ``sigma`` and useful for the same reason: a near-singular target makes the
    field stiff *only* at ``t = 1``, so integrating to ``1 − ε`` avoids the blow-up
    entirely.

    The training path is ``z_t = (1−t)·z0 + t·z1``, so the marginal at ``t_end`` is
    the data scaled by ``t_end`` plus noise scaled by ``1 − t_end``: slightly
    shrunk as well as slightly blurred. ``rescale`` divides by ``t_end`` to undo
    the shrink, leaving only the blur. (Dividing by ``1 − t_end`` instead — the
    natural-looking mistake — inflates the samples by 1/ε, which at ε=0.02 is 50x
    and unmissable if you check the sample std, silent if you do not.)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n, k, generator=g, device=device)
    dt = t_end / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        z = z + model(z, t) * dt
    out = z.cpu().numpy()
    return out / t_end if (rescale and t_end < 1.0) else out


def gaussian_baseline(Z: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Samples from a diagonal Gaussian fitted to ``Z`` — the baseline to beat.

    Kept honest and prominent on purpose. Whitened PCA coordinates are decorrelated
    and unit-variance by construction, so this is very nearly ``N(0, I)``, and it
    is exactly what a classical 3DMM ships. With a few hundred training shapes a
    flow model that does not clearly beat this is not earning its complexity.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(Z.mean(0), Z.std(0), size=(n, Z.shape[1]))
