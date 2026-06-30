"""Session 10 acceptance tests: rectified flow over the VAE latent set.

Maps to Appendix B row 13:
  13. physics velocity beats vanilla                metric

Plus: vanilla RF overfits a tiny latent set; the Heun ODE sampler recovers the
data manifold; samples decode to valid genus-0 bottles (trust-region decode);
the physics-informed velocity target is differentiable and better predicted by
the physics-trained flow. CPU + tiny models.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bottlegen.decode.reconstruct import decode                # noqa: E402
from bottlegen.eval import metrics                             # noqa: E402
from bottlegen.geom.mesh import icosphere                      # noqa: E402
from bottlegen.models import training                          # noqa: E402
from bottlegen.models.pack import kappa_packed, packed_to_sigma  # noqa: E402
from bottlegen.models.rect_flow import (LatentDiT,             # noqa: E402
                                        decoder_vjp_velocity,
                                        heun_sample)
from bottlegen.models.set_vae import SigmaDecoder              # noqa: E402

M, ZDIM, L = 6, 6, 8
RF_CFG = dict(d=96, depth=3, heads=4, t_embed=64)


def _latents(n, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(scale * rng.standard_normal((n, M, ZDIM)),
                           dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Model + sampler sanity                                                       #
# --------------------------------------------------------------------------- #

def test_latent_dit_shapes_and_zero_init():
    dit = LatentDiT(M, ZDIM, **RF_CFG)
    z = torch.randn(4, M, ZDIM)
    v = dit(z, torch.rand(4))
    assert v.shape == (4, M, ZDIM)
    # zero-initialized output -> initial velocity is exactly zero
    assert v.abs().max() < 1e-8
    # so the Heun sampler is the identity at init
    z0 = torch.randn(3, M, ZDIM)
    assert torch.allclose(heun_sample(dit, z0, n_steps=8), z0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Test: vanilla RF overfits a single latent / recovers a tiny manifold        #
# --------------------------------------------------------------------------- #

def test_vanilla_overfit_single_latent():
    # Rectified flow transports the prior toward the data. For a single target,
    # the marginal velocity is singular at t=1 (point mass), so the flow gets
    # close, not exact — measured against the prior→target distance.
    z1 = _latents(1, seed=1, scale=0.6)
    model, ema, hist = training.fit_rect_flow(
        z1.repeat(16, 1, 1), mode="vanilla", rf_cfg=RF_CFG, steps=1200,
        batch=16, lr=2e-3, seed=0)
    ema.copy_to(model)
    model.eval()
    z0 = torch.randn(16, M, ZDIM)
    samples = heun_sample(model, z0, n_steps=100)
    sample_err = (samples - z1).flatten(1).norm(dim=1).mean().item()
    prior_err = (z0 - z1).flatten(1).norm(dim=1).mean().item()
    print(f"[rf overfit-1] sample_err {sample_err:.3f} vs prior {prior_err:.3f}")
    assert sample_err < 0.4 * prior_err                      # transports >60%


def test_vanilla_recovers_manifold():
    data = _latents(8, seed=2, scale=0.6)
    model, ema, hist = training.fit_rect_flow(
        data, mode="vanilla", rf_cfg=RF_CFG, steps=1800, batch=8, lr=2e-3,
        seed=0)
    assert np.mean(hist[-50:]) < 0.7 * np.mean(hist[:50])     # loss decreases
    ema.copy_to(model)
    model.eval()
    z0 = torch.randn(32, M, ZDIM)
    samples = heun_sample(model, z0, n_steps=100)
    d_sample = torch.cdist(samples.flatten(1), data.flatten(1)).min(1).values
    d_prior = torch.cdist(z0.flatten(1), data.flatten(1)).min(1).values
    print(f"[rf manifold] sample→data {d_sample.mean():.3f} vs "
          f"prior→data {d_prior.mean():.3f}")
    assert d_sample.mean() < 0.65 * d_prior.mean()           # pulled to manifold


# --------------------------------------------------------------------------- #
# Physics-informed velocity: differentiable, and better predicted by physics   #
# --------------------------------------------------------------------------- #

def test_decoder_vjp_velocity_finite_nonzero():
    dec = SigmaDecoder(ZDIM, M, L, d=64, heads=4, depth=1)
    kap = torch.as_tensor(kappa_packed(L), dtype=torch.float32)
    zt = torch.randn(5, M, ZDIM)
    v = decoder_vjp_velocity(dec, zt, kap)
    assert v.shape == (5, M, ZDIM)
    assert torch.isfinite(v).all() and v.abs().max() > 0


def test_physics_velocity_beats_vanilla_on_heldout():
    torch.manual_seed(0)
    dec = SigmaDecoder(ZDIM, M, L, d=64, heads=4, depth=1).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    kap = torch.as_tensor(kappa_packed(L), dtype=torch.float32)
    train_z = _latents(16, seed=3)
    test_z = _latents(8, seed=99)

    common = dict(rf_cfg=RF_CFG, steps=700, batch=16, lr=2e-3, seed=0)
    m_phys, ema_p, _ = training.fit_rect_flow(
        train_z, mode="physics", decoder=dec, kappa_packed=kap, **common)
    m_van, ema_v, _ = training.fit_rect_flow(train_z, mode="vanilla", **common)
    ema_p.copy_to(m_phys); ema_v.copy_to(m_van)
    m_phys.eval(); m_van.eval()

    # held-out physics targets at interpolated states
    rng = np.random.default_rng(7)
    z0 = torch.randn(test_z.shape)
    t = torch.rand(test_z.shape[0])
    zt = (1 - t.reshape(-1, 1, 1)) * z0 + t.reshape(-1, 1, 1) * test_z
    tgt = decoder_vjp_velocity(dec, zt, kap)
    tgt = tgt / (tgt.flatten(1).norm(dim=1).reshape(-1, 1, 1) + 1e-8)

    def cos(model):
        with torch.no_grad():
            v = model(zt, t)
        v = v / (v.flatten(1).norm(dim=1).reshape(-1, 1, 1) + 1e-8)
        return (v * tgt).flatten(1).sum(1).mean().item()

    cp, cv = cos(m_phys), cos(m_van)
    print(f"[physics vs vanilla] cosine to physics target: "
          f"physics={cp:.3f} vanilla={cv:.3f}")
    assert cp > cv + 0.1


# --------------------------------------------------------------------------- #
# Samples decode to valid genus-0 bottles (trust-region decode)               #
# --------------------------------------------------------------------------- #

def test_samples_decode_to_valid_bottles():
    torch.manual_seed(0)
    dec = SigmaDecoder(ZDIM, M, L, d=64, heads=4, depth=1).eval()
    # small in-class latents (so decoded σ̂ is modest)
    data = _latents(8, seed=5, scale=0.5)
    model, ema, _ = training.fit_rect_flow(
        data, mode="vanilla", rf_cfg=RF_CFG, steps=400, batch=8, lr=2e-3,
        seed=0)
    ema.copy_to(model); model.eval()
    z = heun_sample(model, torch.randn(3, M, ZDIM) * 0.5, n_steps=32)
    with torch.no_grad():
        sigmas = dec(z).cpu().numpy()
    base = icosphere(3)
    for vec in sigmas:
        sig = packed_to_sigma(vec, L)
        V, F = decode(base.vertices, base.faces, sig, w_cap=120)
        assert np.isfinite(V).all()
        assert metrics.genus(V, F) == 0
        assert metrics.willmore_energy(V, F) > 0
