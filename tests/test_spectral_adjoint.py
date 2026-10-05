"""The adjoint encoder through the mesh-free decoder
(:mod:`shapegen.geom.spectral_adjoint`): its forward pass is the shipped
decoder, and every stage of its gradient agrees with finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.geom import dirac_spectral as ds, spectral_adjoint as sa
from shapegen.geom.encode import n_coeffs, real_sh_basis
from shapegen.geom.mesh import icosphere
from shapegen.quad import lift as ql

from tests.quad_fixtures import bundle_style_coeffs

L = 8
LS = ds.spinor_band(L, ds.potential_band_extra(L))


@pytest.fixture(scope="module")
def setup():
    m = icosphere(3)
    P = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    lift = sa.SpectralLift(LS)
    return P, m.faces, rho, lift, sa.SpectralSpinor(L, LS), real_sh_basis(P, lift.Lf)


def test_forward_is_the_shipped_decoder(setup):
    """Same spinor as :func:`~shapegen.geom.dirac_spectral.solve_spinor`, same
    surface as :func:`~shapegen.quad.lift.lift_spectral` -- the encoder
    optimises the decoder that actually ships, not a look-alike. And the lift
    is exact on the round sphere and exactly band-limited at ``2·Ls``."""
    P, _F, rho, lift, spinor, Y = setup
    assert lift.Lf == int(round(2 * LS))
    v0, _ = spinor.forward(np.zeros(n_coeffs(L)))
    assert codec.shape_error(P, Y @ lift.forward(v0)[0]) < 1e-12
    v, sc = spinor.forward(rho)
    assert sc["gamma"] == pytest.approx(ds.solve_spinor(rho, L, Ls=LS).gamma, rel=1e-10)
    res = ds.DiracSpectral(v[:spinor.n], v[spinor.n:], sc["gamma"], np.nan, 2, LS, L, 1.0)
    Xref, _ = ql.lift_spectral(P, rho, L, res=res)
    assert codec.shape_error(Xref, Y @ lift.forward(v)[0]) < 1e-12


def test_lift_adjoint_matches_finite_differences(setup):
    _P, _F, rho, lift, spinor, _Y = setup
    v, _ = spinor.forward(rho)
    fh, cache = lift.forward(v)
    rng = np.random.default_rng(0)
    W = rng.normal(size=fh.shape)
    g = lift.backward(W, cache)
    for _ in range(3):
        dv = rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)
        h = 1e-6
        fd = (np.sum(W * lift.forward(v + h * dv)[0])
              - np.sum(W * lift.forward(v - h * dv)[0])) / (2 * h)
        assert np.real(np.vdot(g, dv)) == pytest.approx(fd, rel=1e-7)


def test_end_to_end_gradient_matches_finite_differences(setup):
    """ρ̂ -> spinor -> lift -> coefficient Procrustes. The spinor stage
    differentiates a degenerate (quaternionic) kernel pair, which is only legal
    because the loss is gauge-invariant; this is the test that says the
    reduced-resolvent formula and its LU reuse are right."""
    _P, _F, rho, lift, spinor, _Y = setup
    v, sc = spinor.forward(rho)
    fh, lc = lift.forward(v)
    rng = np.random.default_rng(1)
    T = fh + 0.02 * rng.normal(size=fh.shape) / (1 + np.arange(len(fh)))[:, None]
    _J, gf = sa.coef_procrustes(T, fh)
    g = spinor.backward(lift.backward(gf, lc), sc)

    def J(r):
        return sa.coef_procrustes(T, lift.forward(spinor.forward(r)[0])[0])[0]

    for _ in range(3):
        d = rng.normal(size=rho.shape)
        h = 1e-5
        fd = (J(rho + h * d) - J(rho - h * d)) / (2 * h)
        assert g @ d == pytest.approx(fd, rel=1e-5)


def test_encoder_recovers_a_known_code(setup):
    """Target = the decode of a known code; start 20% away from it. The fit
    must close most of that gap -- a wrong sign or a missing adjoint term
    stalls at the start."""
    P, F, rho, lift, spinor, Y = setup
    target = Y @ lift.forward(spinor.forward(rho)[0])[0]
    c0 = rho * (1 + 0.2 * np.random.default_rng(2).normal(size=rho.shape))
    out = sa.encode_spectral(target, P, F, L, c0, maxiter=60, lift=lift,
                             spinor=spinor, Ls=LS)
    assert out["err"] < 0.1 * out["err0"], out


def test_codec_spectral_round_trip():
    """``codec.encode`` (spectral, the default) is decoded by ``codec.decode``
    to exactly the error it reports, beats the round sphere it starts from,
    and returns a closed (exactly realisable) code."""
    from shapegen.data.dataset import demo_bottle_meshes

    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    r = codec.encode(mesh, L, maxiter=40)
    X = codec.decode(r.coeffs, r.sphere)
    assert codec.shape_error(mesh.vertices, X) == pytest.approx(r.err, rel=1e-6)
    assert r.err < 0.5 * r.err0, r
    assert r.gamma < 1e-10, r                       # l = 0 closed
