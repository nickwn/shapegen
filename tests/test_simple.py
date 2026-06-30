"""Session 7 acceptance tests: the Simple cold-diffusion DiT (MVP).

Maps to Appendix B row 10:
  10. Simple overfit 1 & 10                          loss → 0

Plus: forward shapes, packed blur == analytic blur, terminal prior, and samples
that decode to finite-W genus-0 meshes. Runs on CPU with a tiny model for speed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bottlegen.flow import schedule, state              # noqa: E402
from bottlegen.flow.schedule import _hermitian_symmetrize  # noqa: E402
from bottlegen.geom import dirac                          # noqa: E402
from bottlegen.geom.mesh import icosphere                 # noqa: E402
from bottlegen.geom.operators import vertex_areas         # noqa: E402
from bottlegen.geom.curvature import mean_curvature       # noqa: E402
from bottlegen.models import cold_diffusion as cd          # noqa: E402
from bottlegen.models import pack, training                # noqa: E402
from bottlegen.models.sh_dit import SHDiT                  # noqa: E402
from bottlegen.spectral import swsh                        # noqa: E402

L = 8
TINY = dict(d=48, depth=2, heads=4, t_embed=64)


def _inclass_sigma(seed, amp=0.03, Lm=L):
    """A small, decaying, Hermitian (real-field) in-class σ̂."""
    rng = np.random.default_rng(seed)
    C = swsh.empty_coeffs(Lm)
    for l in range(2, Lm + 1):
        for m in range(-l, l + 1):
            C[l, l + m] = (rng.standard_normal() + 1j * rng.standard_normal()) \
                * amp / l ** 2
    return state.gauge_fix(_hermitian_symmetrize(C))


# --------------------------------------------------------------------------- #
# Forward shapes + param count                                                #
# --------------------------------------------------------------------------- #

def test_forward_shapes():
    sp = pack.PackSpec(L)
    model = SHDiT(L, **TINY)
    B = 5
    x = torch.randn(B, sp.dim)
    t = torch.rand(B)
    y = model(x, t)
    assert y.shape == (B, sp.dim)
    assert model.num_params() > 0


def test_full_size_param_count_in_range():
    # the documented ~22 M default config
    model = SHDiT(16, d=384, depth=12, heads=6)
    n = model.num_params()
    print(f"[params] default SHDiT L=16: {n/1e6:.1f}M")
    assert 10e6 < n < 40e6


# --------------------------------------------------------------------------- #
# Packed blur matches the analytic complex blur                               #
# --------------------------------------------------------------------------- #

def test_packed_blur_matches_analytic():
    C = _inclass_sigma(0)
    kap = schedule.kappa(L)
    kap_t = torch.as_tensor(pack.kappa_packed(L))
    t = 1.3e-3
    x = torch.as_tensor(state.real_pack(C))[None]
    xb = pack.blur_packed(x, torch.tensor([t]), kap_t)[0].numpy()
    Cb_packed = state.real_unpack(xb, L)
    Cb_analytic = schedule.blur(C, t, kap)
    assert np.max(np.abs(Cb_packed - Cb_analytic)) < 1e-6


def test_packed_to_sigma_roundtrip():
    C = _inclass_sigma(1)
    v = state.real_pack(C)
    C2 = pack.packed_to_sigma(v, L)
    assert np.max(np.abs(C2 - C)) < 1e-6


# --------------------------------------------------------------------------- #
# Test 10: overfit 1 and 10                                                   #
# --------------------------------------------------------------------------- #

def test_overfit_one():
    t_max = schedule.tmax_for_band(L)
    data = [_inclass_sigma(7)]
    _, _, hist = training.fit_simple(
        data, L, t_max, model_cfg=TINY, steps=400, batch=1, lr=3e-3,
        whiten=True, rotate_aug=False, warmup=20, seed=0)
    init = np.mean(hist[:10])
    final = np.mean(hist[-20:])
    print(f"[overfit 1] {init:.3e} -> {final:.3e}")
    assert final < 0.05 * init


def test_overfit_ten():
    t_max = schedule.tmax_for_band(L)
    data = [_inclass_sigma(i) for i in range(10)]
    _, _, hist = training.fit_simple(
        data, L, t_max, model_cfg=TINY, steps=800, batch=10, lr=3e-3,
        whiten=True, rotate_aug=False, warmup=40, seed=0)
    init = np.mean(hist[:10])
    final = np.mean(hist[-20:])
    print(f"[overfit 10] {init:.3e} -> {final:.3e}")
    assert final < 0.2 * init


# --------------------------------------------------------------------------- #
# Terminal prior (packed) fit/sample                                          #
# --------------------------------------------------------------------------- #

def test_terminal_prior_packed():
    sp = pack.PackSpec(L)
    data = np.stack([state.real_pack(_inclass_sigma(i)) for i in range(12)])
    prior = cd.TerminalPriorPacked(sp.dim).fit(
        data, schedule.tmax_for_band(8), pack.kappa_packed(L))
    smp = prior.sample(5, rng=np.random.default_rng(0))
    assert smp.shape == (5, sp.dim)


# --------------------------------------------------------------------------- #
# Sample -> decode to a finite-W, genus-0 mesh                                #
# --------------------------------------------------------------------------- #

def test_sample_decodes_to_finite_w_genus0():
    t_max = schedule.tmax_for_band(L)
    data = [_inclass_sigma(i) for i in range(8)]
    model, ema, _ = training.fit_simple(
        data, L, t_max, model_cfg=TINY, steps=300, batch=8, lr=3e-3,
        whiten=True, seed=0)
    ema.copy_to(model)
    model.eval()
    cdiff = cd.ColdDiffusion(L, t_max)
    prior = cd.TerminalPriorPacked(pack.PackSpec(L).dim).fit(
        np.stack([state.real_pack(d) for d in data]), t_max,
        pack.kappa_packed(L))
    states = cdiff.sample(model, prior, n=2, T=16,
                          rng=np.random.default_rng(1))
    base = icosphere(3)
    for vec in states:
        sigma = pack.packed_to_sigma(vec, L)
        V, F = dirac.reconstruct(base.vertices, base.faces, sigma)
        assert np.isfinite(V).all()
        H, _, _ = mean_curvature(V, F)
        W = float((H ** 2 * vertex_areas(V, F)).sum())
        print(f"[sample decode] W={W:.2f}  genus=0  finite={np.isfinite(W)}")
        assert np.isfinite(W) and W > 0
