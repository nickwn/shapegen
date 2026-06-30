"""Cold-diffusion training step, EMA, terminal prior, and the deterministic
restoration sampler — the reverse of the analytic SWSH blur.

Forward (corruption): ``x_t = blur(x_0, t)`` with ``t`` a *physical* time in
``[0, t_max]``. The network conditions on the *normalized* time ``t/t_max`` and
predicts the clean state ``x_0``. Sampling seeds from the terminal prior (the
only source of diversity, since cold diffusion is deterministic) and walks the
blur ladder back down, re-blurring each x0 prediction to the next, less-blurred
level.
"""

from __future__ import annotations

import numpy as np
import torch

from ..flow import schedule
from .pack import PackSpec, blur_packed, kappa_packed

__all__ = ["EMA", "ColdDiffusion", "TerminalPriorPacked"]


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


class TerminalPriorPacked:
    """Diagonal Gaussian over packed states at ``t_max`` — the sampling seed.

    Fit on the *blurred* training states so it captures exactly the low-ℓ modes
    that survive the forward process.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.mean = None
        self.std = None

    def fit(self, x0_packed: np.ndarray, t_max: float, kap_packed: np.ndarray):
        x = np.asarray(x0_packed, dtype=np.float64)
        blurred = x * np.exp(-kap_packed[None, :] * t_max)
        self.mean = blurred.mean(0)
        self.std = blurred.std(0) + 1e-8
        return self

    def sample(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return self.mean[None] + self.std[None] * rng.standard_normal((n, self.dim))


class ColdDiffusion:
    """Bundles the blur schedule for a bandlimit ``L`` on a torch device."""

    def __init__(self, L: int, t_max: float, device=None,
                 dtype=torch.float32):
        self.L = L
        self.t_max = float(t_max)
        self.spec = PackSpec(L)
        self.device = device
        self.dtype = dtype
        self.kappa_t = torch.as_tensor(kappa_packed(L), dtype=dtype,
                                       device=device)

    # --- forward corruption ------------------------------------------------
    def blur(self, x, t_phys):
        return blur_packed(x, t_phys, self.kappa_t)

    def sample_t(self, batch: int, rng=None):
        """Random normalized times ``t/t_max`` ∈ [0,1] -> ``(t_norm, t_phys)``."""
        g = torch.Generator(device="cpu")
        if rng is not None:
            g.manual_seed(int(rng.integers(0, 2 ** 31)))
        tn = torch.rand(batch, generator=g).to(self.device)
        return tn, tn * self.t_max

    # --- training step -----------------------------------------------------
    def loss_step(self, model, x0, weight, loss_fn, rng=None):
        from .losses import sobolev_loss
        tn, tp = self.sample_t(x0.shape[0], rng)
        xt = self.blur(x0, tp)
        pred = model(xt, tn)
        return (loss_fn or sobolev_loss)(pred, x0, weight)

    # --- sampler -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, model, prior: TerminalPriorPacked, n: int, T: int = 64,
               rng=None, noise: float = 0.0):
        """Deterministic cold-diffusion restoration. Returns packed x0 states
        ``[n, dim]`` (numpy)."""
        rng = np.random.default_rng() if rng is None else rng
        x = torch.as_tensor(prior.sample(n, rng), dtype=self.dtype,
                            device=self.device)
        t_grid = np.linspace(self.t_max, 0.0, T)        # most -> least blurred
        for k in range(T - 1):
            tn = torch.full((n,), t_grid[k] / self.t_max, dtype=self.dtype,
                            device=self.device)
            x0 = model(x, tn)
            x = self.blur(x0, float(t_grid[k + 1]))     # re-blur to next level
            if noise > 0:
                x = x + noise * torch.as_tensor(
                    rng.standard_normal(x.shape), dtype=self.dtype,
                    device=self.device)
        # final prediction at t≈0
        tn = torch.zeros(n, dtype=self.dtype, device=self.device)
        x0 = model(x, tn)
        return x0.detach().cpu().numpy()
