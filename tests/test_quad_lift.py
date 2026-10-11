"""The spinor-to-quaternion dictionary and the mesh-free lift
built on it (:mod:`shapegen.quad.lift`).

The dictionary is the gate — and the round sphere alone cannot pin it (see
``test_kernel_gauge_is_a_right_constant``). ``geom/dirac_spectral`` returns ``λ`` as spin-±½
halves; every one of these tests exists because building an immersion on an
unverified spinor-to-quaternion map is how a wrong convention becomes a silent
geometric error.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.geom import dirac, dirac_spectral as ds
from shapegen.geom.encode import n_coeffs, real_sh_basis
from shapegen.geom.mesh import icosphere
from shapegen.quad import lift as ql
from shapegen.spectral.quaternion import qinv, qmul, qnorm

from tests.quad_fixtures import bundle_style_coeffs

L = 8


def _sphere(subdiv=3):
    m = icosphere(subdiv)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    return V, m.faces


def _scatter(q):
    """Median distance from the mean on sign-aligned unit quaternions.
    A random field scores ~0.91; a constant field scores 0."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    s = np.sign((q * q[0]).sum(1))
    s[s == 0] = 1.0
    qs = q * s[:, None]
    mu = qs.mean(0)
    mu /= np.linalg.norm(mu)
    return float(np.median(np.linalg.norm(qs - mu, axis=1)))


# --------------------------------------------------------------------------- #
# The dictionary                                                              #
# --------------------------------------------------------------------------- #

def test_frame_quaternion_carries_i_to_the_normal():
    """``R = q_z(φ) q_y(θ − π/2)`` is unit and rotates the quaternion **i**
    axis — the one ``ℍ = ℂ ⊕ ℂj`` quantises spin about — onto the outward
    normal, which is what makes left multiplication by it a frame."""
    from shapegen.spectral.quaternion import from_vec3, qconj, to_vec3
    rng = np.random.default_rng(0)
    u = rng.normal(size=(500, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    R = ql.frame_quaternion(th, ph)
    assert R.shape == (500, 4)
    assert np.allclose(qnorm(R), 1.0, atol=1e-13)
    i = from_vec3(np.tile([1.0, 0.0, 0.0], (500, 1)))
    assert np.abs(to_vec3(qmul(qmul(R, i), qconj(R))) - u).max() < 1e-12


def test_dictionary_is_exact_on_the_round_sphere():
    """At γ = 0 the spinor is constant up to the right gauge, and agrees with
    the mesh solver's ``λ ≡ 1``. Necessary, but **not sufficient**: any
    ``A(x)·raw·B(x)`` with ``A raw₀ B = 1`` passes here, which is how a wrong
    dictionary survived until 2026-09-26. The two tests below are the ones
    that can tell."""
    V, faces = _sphere(3)
    lam_mesh, _ = dirac.spin_transform(V, faces, np.zeros(len(V)))
    res = ds.solve_spinor(np.zeros(n_coeffs(L)), L)
    assert res.gamma < 1e-12
    assert res.multiplicity == 2
    lam = ql.spinor_quaternion(res, V)
    assert qnorm(lam).std() / qnorm(lam).mean() < 1e-12
    assert _scatter(qmul(qinv(lam), lam_mesh)) < 1e-10


def test_kernel_gauge_is_a_right_constant():
    """The spectral kernel is two complex dimensions — one quaternionic one —
    and under the dictionary it must act on ``λ`` as ``λ → λq``: a *right*
    constant, i.e. a rigid motion of the output. A *left* constant rotates
    the base sphere under a fixed ``λ`` and is a different surface.

    The pre-2026-09-26 map (``F`` on the right, no conjugation) failed exactly
    this, mirrored: left 1e-14, right 0.185."""
    import scipy.linalg as sla
    V, _faces = _sphere(3)
    rho_hat = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    Ls = ds.spinor_band(L)
    G = ds.dirac_matrix(ds.rho_on_grid(rho_hat, L, Ls), Ls, L)
    n = G.shape[0] // 2
    _w, vecs = sla.eigh(G.conj().T @ G, subset_by_index=[0, 1])
    l1, l2 = (ql.spinor_quaternion(
        ds.DiracSpectral(v[:n], v[n:], 0.0, 0.0, 2, Ls, L, 1.0), V)
        for v in vecs.T)
    assert _scatter(qmul(qinv(l1), l2)) < 1e-10          # right: constant
    assert _scatter(qmul(l2, qinv(l1))) > 0.05           # left: not


def test_dictionary_agrees_with_the_mesh_solver_off_the_sphere():
    """Away from γ = 0, ``λ_spec⁻¹ λ_mesh`` is constant to the mesh solver's
    own discretisation error — it falls with subdivision — at production-like
    amplitude. Measured (L=8, ``ℓ^-2``): 2.3e-4 → 1.9e-3 over amp 0.25 → 2 at
    subdivision 3, and 1.6e-3 at amp 2 on subdivision 4.

    The old map scored 0.014 → 0.080 here. It was guarded by a test that
    compared ``λ_mesh λ_spec⁻¹`` — the **left** side, the wrong gauge — where
    old and new score the same ~0.10, and that residual was read as the two
    solvers legitimately disagreeing."""
    rho_hat = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    res = ds.solve_spinor(rho_hat, L)
    out = []
    for subdiv in (3, 4):
        V, faces = _sphere(subdiv)
        lam_mesh, _ = dirac.spin_transform(V, faces, real_sh_basis(V, L) @ rho_hat)
        out.append(_scatter(qmul(qinv(ql.spinor_quaternion(res, V)), lam_mesh)))
    assert out[0] < 5e-3, out
    assert out[1] < out[0], out


def test_spinor_quaternion_is_single_valued():
    """``half_eval_basis`` is antiperiodic in φ and so is ``F``; the product
    is not. Checked across the φ = ±π cut and at both poles, where φ is
    undefined."""
    res = ds.solve_spinor(np.zeros(n_coeffs(L)), L)
    th = np.linspace(0.05, np.pi - 0.05, 64)
    eps = 1e-9

    def at(phi):
        return ql.spinor_quaternion(res, np.stack(
            [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], 1))

    assert np.abs(at(np.pi - eps) - at(-np.pi + eps)).max() < 1e-10

    for z in (+1.0, -1.0):
        t = 1e-7
        ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        d = np.stack([np.sin(t) * np.cos(ang), np.sin(t) * np.sin(ang),
                      np.full_like(ang, z * np.cos(t))], 1)
        q = ql.spinor_quaternion(res, d)
        assert np.abs(q - q.mean(0)).max() < 1e-9


# --------------------------------------------------------------------------- #
# The lift                                                                    #
# --------------------------------------------------------------------------- #


def test_lift_spectral_reconstructs_the_round_sphere():
    V, _faces = _sphere(4)
    Xs, info = ql.lift_spectral(V, np.zeros(n_coeffs(L)), L)
    assert info["gamma"] < 1e-12
    assert codec.shape_error(V, Xs) < 1e-12


@pytest.mark.parametrize("amp", [0.1, 2.0])
def test_mesh_decoder_converges_to_lift_spectral(amp):
    """``lift_spectral`` is the continuum answer the triangle-mesh spin
    transform (:mod:`shapegen.geom.dirac`, an independent solver) approximates: their
    disagreement falls O(h²) with subdivision. Measured 3.8e-4 -> 9.7e-5 ->
    2.4e-5 at amp 0.1 and 7.9e-3 -> 2.1e-3 -> 5.2e-4 at amp 2 (γ = 1.2e-2),
    subdivisions 2/3/4 -- a factor of ~3.9 each step."""
    base = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * amp
    errs = []
    for subdiv in (2, 3, 4):
        V, faces = _sphere(subdiv)
        Xs, _ = ql.lift_spectral(V, base, L)
        lam, _ = dirac.spin_transform(V, faces, real_sh_basis(V, L) @ base)
        errs.append(codec.shape_error(dirac.integrate_edges(V, faces, lam), Xs))
    assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0, errs
    assert errs[2] < 1e-3, errs
