"""Session 5 acceptance tests: forward blur, reverse velocity, Willmore flow.

Maps to Appendix B row 8:
  8. Willmore W↓, q→0; κ_ℓ fit                     qualitative / linear
"""

import numpy as np
import pytest

from bottlegen.flow import schedule, willmore
from bottlegen.flow.schedule import blur, kappa, reverse_velocity
from bottlegen.geom import conformal as cf
from bottlegen.geom.mesh import Mesh, icosphere
from bottlegen.spectral import swsh

L = 16


def _rand_sigma(seed=0, L=L):
    rng = np.random.default_rng(seed)
    C = swsh.empty_coeffs(L)
    for l in range(2, L + 1):
        for m in range(-l, l + 1):
            C[l, l + m] = (rng.standard_normal() + 1j * rng.standard_normal()) \
                / (l ** 2)            # decaying, in-class-ish spectrum
    return C


# --------------------------------------------------------------------------- #
# Blur damps high-ℓ fastest -> only low-ℓ survives                            #
# --------------------------------------------------------------------------- #

def test_kappa_values():
    kap = kappa(8)
    assert kap[0] == 0 and kap[1] == 0
    assert kap[2] == (1 * 2 * 3 * 4)        # 24
    assert kap[3] == (2 * 3 * 4 * 5)        # 120


def test_blur_leaves_low_l():
    C = _rand_sigma(0)
    kap = kappa(L)
    # blur far enough that only the lowest degrees survive (damp ℓ=6 to ~1e-3)
    t = schedule.tmax_for_band(6, top_band_damp=1e-3)
    Cb = blur(C, t, kap)
    e = np.array([np.sum(np.abs(Cb[l]) ** 2) for l in range(L + 1)])
    assert e[L] < 1e-6 * e[2]                  # top band annihilated
    assert e[2:5].sum() > 0.85 * e.sum()       # energy in the lowest degrees


def test_blur_top_band_damped_to_target():
    kap = kappa(L)
    t = schedule.tmax_for_band(L, top_band_damp=1e-3)
    assert abs(np.exp(-kap[L] * t) - 1e-3) < 1e-9


# --------------------------------------------------------------------------- #
# One-step analytic reverse recovers a less-blurred state to first order      #
# --------------------------------------------------------------------------- #

def test_reverse_velocity_first_order():
    C = _rand_sigma(1)
    kap = kappa(L)
    t, dt = 1e-3, 1e-4
    Ct = blur(C, t, kap)
    Ct_less = blur(C, t - dt, kap)               # ground-truth less-blurred
    step = Ct + dt * reverse_velocity(Ct, t, kap)
    err = np.max(np.abs(step - Ct_less))
    # forward-Euler error is O(dt²)
    assert err < 50 * dt ** 2 * np.max(kap) ** 2 * np.max(np.abs(C))


# --------------------------------------------------------------------------- #
# Willmore monotonicity, q→0, κ_ℓ decay rate                                  #
# --------------------------------------------------------------------------- #

def test_willmore_monotone_and_q_to_zero():
    C = _rand_sigma(2)
    # flow long enough that even ℓ=2 rounds out (q → 0, W → 4π)
    t_grid = np.linspace(0, 0.4, 40)
    traj, W = willmore.willmore_flow(C, t_grid)
    assert (np.diff(W) <= 1e-9).all()                       # monotone ↓
    assert W[-1] >= 4 * np.pi - 1e-9                         # bounded below by 4π
    assert W[0] > W[-1]
    q_excess = np.array([willmore.hopf_norm_sq_spectral(s) for s in traj])
    assert q_excess[-1] < 1e-3 * q_excess[0]                # q → 0


def test_decay_rate_matches_kappa():
    C = _rand_sigma(3)
    kap = kappa(L)
    t_grid = np.linspace(0, 5e-3, 20)
    for l in (2, 3, 4, 5):
        amp = np.array([np.linalg.norm(blur(C, float(t), kap)[l]) for t in t_grid])
        # fitted exponential decay rate of degree ℓ equals κ_ℓ
        rate = -np.polyfit(t_grid, np.log(amp), 1)[0]
        assert abs(rate - kap[l]) < 1e-6 * kap[l]


# --------------------------------------------------------------------------- #
# Two ways to ‖q‖²: spectral vs mesh (W − 4π)                                 #
# --------------------------------------------------------------------------- #

def test_willmore_excess_two_ways():
    # encode a bumpy shape, compare spectral ‖q‖² with mesh W − 4π
    s = icosphere(4)
    V = s.vertices
    f = 1.0 + 0.18 * (V[:, 2] ** 2 - 0.3 * V[:, 0])
    m = Mesh(V * f[:, None], s.faces)
    Lm = 16
    sig = cf.mesh_to_sigma(m, L=Lm)
    excess_spec = willmore.hopf_norm_sq_spectral(sig)
    excess_mesh = willmore.willmore_energy_mesh(m.vertices, m.faces) - 4 * np.pi
    rel = abs(excess_spec - excess_mesh) / max(excess_mesh, 1e-9)
    print(f"[‖q‖² two ways] spectral={excess_spec:.4f} mesh={excess_mesh:.4f} "
          f"rel={rel:.3f}")
    assert rel < 0.30


# --------------------------------------------------------------------------- #
# Terminal prior: fit + sample, Hermitian (real σ field)                       #
# --------------------------------------------------------------------------- #

def test_terminal_prior_fit_sample():
    kap = kappa(L)
    t = schedule.tmax_for_band(8)            # degrees ≥ ~8 damped away
    train = np.stack([blur(schedule._hermitian_symmetrize(_rand_sigma(i)), t, kap)
                      for i in range(16)])
    prior = schedule.TerminalPrior().fit(train)
    smp = prior.sample(8, rng=np.random.default_rng(0))
    assert smp.shape == (8,) + train.shape[1:]
    # the top band is negligible vs the surviving low-ℓ
    assert np.max(np.abs(smp[:, L])) < 1e-3 * np.max(np.abs(smp[:, 2]))
    # Hermitian: the implied σ field is real -> synth has ~zero imaginary part
    field = swsh.synth(smp[0], 0, L)
    assert np.max(np.abs(field.imag)) < 1e-9


def test_terminal_prior_deterministic_under_seed():
    kap = kappa(L)
    t = schedule.tmax_for_band(L)
    train = np.stack([blur(_rand_sigma(i), t, kap) for i in range(8)])
    prior = schedule.TerminalPrior().fit(train)
    a = prior.sample(4, rng=np.random.default_rng(42))
    b = prior.sample(4, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)
