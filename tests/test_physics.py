"""Session 12 acceptance tests: physics-prior losses + eval metrics.

Maps to Appendix B row 15:
  15. physics losses reduce R_G, R_C                report

The structure-equation residuals vanish at the round sphere (σ̂=0), are
differentiable, and a Codazzi-penalized fit reduces R_C while keeping the σ̂
reconstruction intact. Eval helpers (novelty/coverage, distribution overlap)
behave on known inputs.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bottlegen.eval import metrics                       # noqa: E402
from bottlegen.flow import state                         # noqa: E402
from bottlegen.models import losses                      # noqa: E402
from bottlegen.models.pack import PackSpec               # noqa: E402
from bottlegen.spectral import swsh                      # noqa: E402

L = 10


def _rand_sigma(seed, amp=0.05):
    rng = np.random.default_rng(seed)
    C = swsh.empty_coeffs(L)
    for l in range(2, L + 1):
        for m in range(-l, l + 1):
            C[l, l + m] = (rng.standard_normal() + 1j * rng.standard_normal()) \
                * amp / l
    return state.gauge_fix(C)


# --------------------------------------------------------------------------- #
# Residuals vanish at the round sphere (σ̂ = 0)                                #
# --------------------------------------------------------------------------- #

def test_residuals_vanish_at_sphere():
    r = losses.physics_residual_norms(swsh.empty_coeffs(L))
    print(f"[sphere residuals] R_C={r['R_C']:.2e} R_G={r['R_G']:.2e}")
    assert r["R_C"] < 1e-9 and r["R_G"] < 1e-9


def test_codazzi_field_op_zero_on_constants():
    # constant H, zero q -> ð̄q − ½e^{2u}ðH = 0
    grid_shape = (swsh.synth(swsh.empty_coeffs(L), 0, L)).shape
    u = np.zeros(L + 1)[:, None] * 0 + np.zeros(grid_shape)
    H = np.ones(grid_shape)
    q = np.zeros(grid_shape, dtype=complex)
    rc = losses.codazzi_residual(u, H, q)
    assert np.abs(rc).max() < 1e-9


def test_residuals_grow_with_sigma():
    r0 = losses.physics_residual_norms(_rand_sigma(0, amp=0.02))
    r1 = losses.physics_residual_norms(_rand_sigma(0, amp=0.08))
    assert r1["R_C"] > r0["R_C"] and r1["R_G"] > r0["R_G"]


# --------------------------------------------------------------------------- #
# Physics loss is differentiable and a gradient step reduces it               #
# --------------------------------------------------------------------------- #

def test_physics_loss_differentiable_and_decreases():
    w = torch.as_tensor(losses.physics_weight_packed(L), dtype=torch.float32)
    z = torch.nn.Parameter(torch.as_tensor(
        state.real_pack(_rand_sigma(1)), dtype=torch.float32)[None])
    opt = torch.optim.Adam([z], lr=1e-2)
    l0 = losses.physics_loss(z, w).item()
    for _ in range(50):
        opt.zero_grad()
        loss = losses.physics_loss(z, w)
        loss.backward()
        opt.step()
    assert losses.physics_loss(z, w).item() < 0.2 * l0


# --------------------------------------------------------------------------- #
# Test 15: physics penalty reduces R_C without wrecking reconstruction         #
# --------------------------------------------------------------------------- #

def test_physics_penalty_reduces_RC_keeps_recon():
    sp = PackSpec(L)
    target_c = _rand_sigma(2, amp=0.06)
    target = torch.as_tensor(state.real_pack(target_c), dtype=torch.float32)[None]
    w = torch.as_tensor(losses.physics_weight_packed(L), dtype=torch.float32)

    z = torch.nn.Parameter(target.clone())
    opt = torch.optim.Adam([z], lr=5e-3)
    lam = 3e-3
    for _ in range(400):
        opt.zero_grad()
        recon = ((z - target) ** 2).mean()
        loss = recon + lam * losses.physics_loss(z, w)
        loss.backward()
        opt.step()

    zc = state.real_unpack(z.detach()[0].numpy(), L)
    RC_opt = losses.physics_residual_norms(zc)["R_C"]
    RC_tgt = losses.physics_residual_norms(target_c)["R_C"]
    recon = float(((z.detach() - target) ** 2).mean())
    print(f"[physics reg] R_C target={RC_tgt:.4f} -> opt={RC_opt:.4f}  "
          f"recon={recon:.2e}")
    assert RC_opt < 0.9 * RC_tgt                  # physics residual reduced
    assert recon < 0.05 * (target ** 2).mean().item()   # reconstruction intact


# --------------------------------------------------------------------------- #
# Eval helpers                                                                #
# --------------------------------------------------------------------------- #

def test_novelty_coverage():
    train = np.random.default_rng(0).standard_normal((6, 20))
    # generated = train + tiny noise -> low novelty, full coverage
    gen = train + 1e-3 * np.random.default_rng(1).standard_normal((6, 20))
    nov, cov = metrics.novelty_coverage(gen, train)
    assert nov < 0.1 and cov == 1.0
    # far-away generated -> high novelty
    nov2, _ = metrics.novelty_coverage(gen + 100, train)
    assert nov2 > 10


def test_distribution_overlap():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    assert metrics.distribution_overlap(a, a.copy()) > 0.95     # identical
    b = rng.normal(8, 1, 500)
    assert metrics.distribution_overlap(a, b) < 0.1             # disjoint
