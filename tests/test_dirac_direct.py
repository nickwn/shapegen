"""The direct Dirac encoder (:mod:`shapegen.geom.dirac_direct`): the spinor read
off a surface is the decoder's, a code is recovered exactly from its own
spinor, and the codec method beats the curvature measurement."""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.geom import dirac_direct as dd, dirac_spectral as ds, spectral_adjoint as sa
from shapegen.geom.encode import n_coeffs

from tests.quad_fixtures import bundle_style_coeffs

L = 8
LS = ds.spinor_band(L, ds.potential_band_extra(L))


@pytest.fixture(scope="module")
def closed():
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    rho = ds.close_potential(rho, ds.solve_spinor(rho, L, Ls=LS))
    res = ds.solve_spinor(rho, L, Ls=LS)
    return rho, np.r_[res.psi1, res.psi2], sa.SpectralLift(LS)


def test_code_is_recovered_from_its_own_spinor(closed):
    """``G(ρ̂) v = 0`` exactly at the code, and the least squares is
    well-posed, so the decoder's spinor gives the code back."""
    rho, v, _lift = closed
    out = dd.rho_for_spinor(v, L, LS, reweight=0)
    assert np.linalg.norm(out - rho) < 1e-8 * np.linalg.norm(rho)
    out = dd.rho_for_spinor(v, L, LS, reweight=1)
    assert np.linalg.norm(out - rho) < 1e-8 * np.linalg.norm(rho)


def test_mesh_spinor_converges_on_a_decoded_surface(closed):
    """λ read off a mesh of a decoded surface (per-face similarities, projected
    by quadrature) converges to the decoder's spinor at second order in the mesh
    size, and the code recovered from it follows."""
    from shapegen.geom.encode import real_sh_basis
    from shapegen.geom.mesh import icosphere

    rho, v, lift = closed
    fh = lift.forward(v)[0]
    errs = []
    for k in (3, 4):
        m = icosphere(k)
        S = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
        vT = dd.mesh_spinor(S, real_sh_basis(S, lift.Lf) @ fh, m.faces, LS)
        vT = vT * np.sign(np.real(np.vdot(v, vT)))           # global sign is a gauge
        errs.append(np.linalg.norm(vT - v) / np.linalg.norm(v))
    assert errs[1] < 5e-3 and errs[1] < 0.35 * errs[0], errs     # 0.0089 -> 0.0021
    r = dd.rho_for_spinor(vT, L, LS)
    assert np.linalg.norm(r[1:] - rho[1:]) < 1e-2 * np.linalg.norm(rho[1:])


def test_codec_dirac_beats_direct():
    from shapegen.data.dataset import demo_bottle_meshes

    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    sph = codec.base_sphere(mesh)
    r = codec.encode(mesh, L, sphere=sph, method="dirac")
    assert codec.shape_error(mesh.vertices, codec.decode(r.coeffs, sph)) == pytest.approx(r.err, rel=1e-6)
    assert r.gamma < 1e-10
    assert r.err < codec.encode(mesh, L, sphere=sph, method="direct").err, r
