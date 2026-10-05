"""Gates for the stable Wigner-d recurrence,
half-integer degrees, and the spectral quaternionic Dirac solve.

The three are one chain: the Dirac system is written in spin ±1/2 fields,
which need half-integer ℓ and m, which need a Wigner small-d that survives
degree 32. The last of those was a decision to take before any of the
spectral work, and it was still open.

The blend control ``t`` and the Hopf differential it is built on are
deliberately not covered -- that is the curvature-alignment work, and
it is out of scope here.
"""

import numpy as np
import pytest

from shapegen.geom import dirac, dirac_spectral as ds
from shapegen.geom.encode import n_coeffs, real_sh_basis
from shapegen.geom.mesh import icosphere
from shapegen.quad import field as qf
from shapegen.spectral import swsh as sw
from tests.quad_fixtures import bundle_style_coeffs

THETA = np.linspace(0.05, np.pi - 0.05, 9)


# --------------------------------------------------------------------------- #
# P8.1 -- the transform has to reach band 32                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L", [16, 24, 32])
@pytest.mark.parametrize("s", [0, 2, 4])
def test_gate_p8_1_round_trip_at_band_32(L, s):
    """The Wigner-recurrence gate: round trip to 1e-12 at band 32, both ways.

    The sum formula this replaced managed 5.3e-8 there -- four decades short
    of the gate, at the band the whole project targets.
    """
    rng = np.random.default_rng(0)
    C = sw.empty_coeffs(L)
    for l in range(abs(s), L + 1):
        C[l, 0:2 * l + 1] = (rng.normal(size=2 * l + 1)
                             + 1j * rng.normal(size=2 * l + 1))
    err = np.abs(sw.analy(sw.synth(C, s, L), s, L) - C).max() / np.abs(C).max()
    assert err < 1e-12


@pytest.mark.parametrize("l", [8, 16, 24, 32, 48])
def test_wigner_d_is_unitary_at_every_degree(l):
    """``Σ_b (d^ℓ_{a,b})² = 1`` exactly, for any correct implementation. This
    is what fails first, and silently, when the sum formula loses precision:
    6.2e-15 at ℓ=8 but 1.1e-3 by ℓ=48 (and an OverflowError at 64)."""
    for a in (0, 3):
        tot = np.zeros_like(THETA)
        for b in range(-l, l + 1):
            _ls, d = sw.wigner_d_ladder(l, a, b, THETA)
            tot += d[-1] ** 2
        assert np.abs(tot - 1.0).max() < 1e-12


def test_recurrence_reproduces_the_sum_formula_where_the_sum_is_trustworthy():
    """Below ℓ≈12 the sum formula is still accurate, so it is a valid oracle
    there -- and the whole point of the replacement is that the two agree
    where both work, then only one keeps working."""
    from math import factorial, sqrt

    def sum_formula(l, a, b, th):
        c, s = np.cos(th / 2), np.sin(th / 2)
        pref = sqrt(factorial(l + a) * factorial(l - a)
                    * factorial(l + b) * factorial(l - b))
        out = np.zeros_like(th)
        for k in range(max(0, b - a), min(l + b, l - a) + 1):
            den = (factorial(l + b - k) * factorial(k)
                   * factorial(a - b + k) * factorial(l - a - k))
            out += ((-1) ** (a - b + k)) * (pref / den) \
                * c ** (2 * l + b - a - 2 * k) * s ** (a - b + 2 * k)
        return out

    worst = 0.0
    for a in (-2, 0, 1, 3):
        for b in (-4, 0, 2):
            ls, D = sw.wigner_d_ladder(12, a, b, THETA)
            for i, l in enumerate(ls):
                worst = max(worst, np.abs(D[i] - sum_formula(int(l), a, b,
                                                             THETA)).max())
    assert worst < 1e-12


def test_legendre_is_the_a_equals_b_equals_zero_case():
    """``d^ℓ_{0,0} = P_ℓ(cos θ)`` -- the cheapest check that the recurrence is
    transcribed correctly, since it must degenerate to Legendre's."""
    from numpy.polynomial import legendre
    ls, D = sw.wigner_d_ladder(20, 0, 0, THETA)
    for i, l in enumerate(ls):
        c = np.zeros(int(l) + 1)
        c[-1] = 1.0
        assert np.abs(D[i] - legendre.legval(np.cos(THETA), c)).max() < 1e-12


# --------------------------------------------------------------------------- #
# P8.2 -- half-integer degrees, which have no external oracle                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L", [5.5, 11.5, 31.5])
@pytest.mark.parametrize("s", [0.5, -0.5])
def test_half_integer_round_trip(L, s):
    """``half_analy ∘ half_synth = id``. At L=31.5 there are 1056 modes per
    spin -- 2112 across both, exactly the predicted unknown count for
    the Dirac solve at L=32."""
    n = len(sw.half_modes(s, L))
    rng = np.random.default_rng(0)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    err = np.abs(sw.half_analy(sw.half_synth(v, s, L), s, L) - v).max()
    assert err / np.abs(v).max() < 1e-12
    if L == 31.5:
        assert n == 1056


@pytest.mark.parametrize("s,l,m", [(0.5, 0.5, 0.5), (0.5, 1.5, -0.5),
                                   (-0.5, 2.5, 1.5), (0.5, 5.5, 3.5)])
def test_gate_p8_2_orthonormality(s, l, m):
    """The half-integer gate, at a half-integer ``(s, ℓ, m)``.

    **There is no oracle for this half.** ``spherical.Wigner(2.5)``
    constructs without complaint and then dies inside its numba kernel on
    ``sYlm(0.5, R)``, so these internal identities are the only protection
    the half-integer path has -- do not weaken them, and do not assume that
    passing at integer spin says anything here.
    """
    L = 7.5
    modes = sw.half_modes(s, L)
    i = modes.index((l, m))
    B = sw.half_basis(s, L)
    _th, _ph, w = sw.half_grid(L)
    W = np.broadcast_to(w, (len(_th), len(_ph))).ravel()
    gram = B.conj() @ (W[:, None] * B.T)
    assert abs(gram[i, i] - 1.0) < 1e-8
    off = np.abs(np.delete(gram[i], i))
    assert off.max() < 1e-8


@pytest.mark.parametrize("L", [5.5, 11.5])
def test_gate_p8_2_edth_eigenvalue_ratio_is_constant(L):
    """The second half-integer gate: the ð ladder factor depends only on ℓ,
    never on m. At spin ∓1/2 it is exactly ``ℓ + 1/2``, which is the identity
    the Dirac operator's diagonal blocks are built from."""
    for s in (0.5, -0.5):
        fe = sw.half_edth_factor(s, L)
        fb = sw.half_edthbar_factor(s, L)
        for i, (l, _m) in enumerate(sw.half_modes(s, L)):
            assert fe[i] == pytest.approx(np.sqrt((l - s) * (l + s + 1)), abs=1e-9)
            assert fb[i] == pytest.approx(-np.sqrt((l + s) * (l - s + 1)), abs=1e-9)
    # the pair the Dirac system uses: both magnitudes are l + 1/2
    for i, (l, _m) in enumerate(sw.half_modes(-0.5, L)):
        assert sw.half_edth_factor(-0.5, L)[i] == pytest.approx(l + 0.5, abs=1e-9)
        assert sw.half_edthbar_factor(0.5, L)[i] == pytest.approx(-(l + 0.5), abs=1e-9)


# --------------------------------------------------------------------------- #
# P9 -- the Dirac solve                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L", [4, 8, 12])
def test_gate_p9_a_round_sphere(L):
    """**The round-sphere gate, not to proceed past.** ``ρ_repo ≡ 0``
    must give ``γ = 0`` and a constant spinor.

    This is also what pins the ρ convention: the round sphere is
    ``ρ_bundle = 1``, not 0, and the kernel condition on a constant ρ is
    ``(ℓ+1/2)² = ρ²`` -- which at ρ=1 selects ℓ=1/2 and nothing else. If the
    ``+1`` were missing or applied twice, this test would find no kernel at
    all.
    """
    res = ds.solve_spinor(np.zeros(n_coeffs(L)), L, with_gap=True)
    assert res.gamma < 1e-10
    assert res.multiplicity == 2          # 2, not the mesh path's 4
    assert res.gap > 0.5                  # the kernel is cleanly separated

    modes = sw.half_modes(0.5, res.Ls)
    lv = np.array([l for l, _m in modes])
    power = np.abs(res.psi1) ** 2 + np.abs(res.psi2) ** 2
    assert power[lv == 0.5].sum() / power.sum() > 1.0 - 1e-12


@pytest.mark.parametrize("amp", [0.0, 1.0, 8.0])
def test_lu_solve_matches_the_svd(amp):
    """The default ``method="lu"`` (one LU + block inverse iteration) against
    the dense SVD: same ``gamma``, same gap, same kernel *pair* (compared as a
    subspace -- the pair is only defined up to the quaternionic gauge), at
    the round sphere, fixture amplitude and production-like amplitude. And a
    warm start from a nearby solve reaches the same answer."""
    L = 10
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * amp
    Ls = ds.spinor_band(L, ds.potential_band_extra(L))
    ref = ds.solve_spinor(rho, L, Ls=Ls, method="svd")
    res = ds.solve_spinor(rho, L, Ls=Ls, with_gap=True)
    assert res.gamma == pytest.approx(ref.gamma, rel=1e-10, abs=1e-13)
    assert res.multiplicity == ref.multiplicity == 2
    assert res.gap == pytest.approx(ref.gap, rel=1e-4)

    # both vectors of the reference pair must lie in the LU solve's pair
    G = ds.dirac_matrix(ds.rho_on_grid(rho, L, Ls), Ls, L)
    _u, _s, vh = np.linalg.svd(G)
    pair = vh[-2:].conj().T
    v = np.r_[res.psi1, res.psi2]
    assert np.linalg.norm(v - pair @ (pair.conj().T @ v)) < 1e-8

    warm = ds.solve_spinor(rho * 1.01, L, Ls=Ls, v0=res)
    cold = ds.solve_spinor(rho * 1.01, L, Ls=Ls)
    assert warm.gamma == pytest.approx(cold.gamma, rel=1e-10, abs=1e-13)
    assert np.isnan(cold.gap)             # not computed unless asked for


def test_schur_solves_match_the_dense_operator():
    """:class:`~shapegen.geom.dirac_spectral.DiracOperator` solves ``G`` and
    ``G*`` through an ``n × n`` Schur complement; both must invert the dense
    ``G``, and ``apply``/``apply_h`` must be it and its adjoint."""
    L = 8
    Ls = ds.spinor_band(L, ds.potential_band_extra(L))
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    op = ds.dirac_operator(ds.rho_on_grid(rho, L, Ls), Ls, L)
    G = op.dense()
    rng = np.random.default_rng(0)
    b = rng.normal(size=(2 * op.n, 3)) + 1j * rng.normal(size=(2 * op.n, 3))
    assert np.abs(op.apply(b) - G @ b).max() < 1e-12
    assert np.abs(op.apply_h(b) - G.conj().T @ b).max() < 1e-12
    assert np.abs(G @ op.solve(b) - b).max() < 1e-10
    assert np.abs(G.conj().T @ op.solve_h(b[:, 0]) - b[:, 0]).max() < 1e-10


def test_close_potential_zeroes_gamma_without_moving_the_surface():
    """``sigma3 G`` is Hermitian and ``l = 0`` of ``rho`` shifts it by a
    constant, so :func:`~shapegen.geom.dirac_spectral.close_potential` makes
    ``gamma`` vanish while the spinor (and so the surface) stays put."""
    from shapegen.geom.spectral_adjoint import SpectralLift, coef_procrustes

    L = 8
    Ls = ds.spinor_band(L, ds.potential_band_extra(L))
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    op = ds.dirac_operator(ds.rho_on_grid(rho, L, Ls), Ls, L)
    Gp = np.r_[np.ones(op.n), -np.ones(op.n)][:, None] * op.dense()
    assert np.abs(Gp - Gp.conj().T).max() < 1e-12
    res = ds.solve_spinor(rho, L, Ls=Ls)
    assert abs(ds.signed_gamma(res)) == pytest.approx(res.gamma, rel=1e-8)
    closed = ds.close_potential(rho, res)
    assert np.array_equal(closed[1:], rho[1:])
    res2 = ds.solve_spinor(closed, L, Ls=Ls)
    assert res2.gamma < 1e-12
    lift = SpectralLift(Ls)
    f1 = lift.forward(np.r_[res.psi1, res.psi2])[0]
    f2 = lift.forward(np.r_[res2.psi1, res2.psi2])[0]
    assert coef_procrustes(f1, f2)[0] < 1e-18


def test_separable_synthesis_is_the_basis():
    """:class:`~shapegen.geom.dirac_spectral.SpinorTransform` evaluates the
    same spin-±1/2 functions as :func:`~shapegen.spectral.swsh.half_eval_basis`,
    and ``adjoint`` is its adjoint."""
    Ls = 7.5
    th = np.arccos(np.polynomial.legendre.leggauss(12)[0])
    tf = ds.SpinorTransform(Ls, th, 20)
    TH, PH = np.meshgrid(th, 2 * np.pi * np.arange(20) / 20, indexing="ij")
    rng = np.random.default_rng(0)
    for s in (0.5, -0.5):
        x = rng.normal(size=tf.n) + 1j * rng.normal(size=tf.n)
        B = sw.half_eval_basis(s, Ls, TH.ravel(), PH.ravel())
        assert np.abs(tf.synth(s, x).ravel() - B @ x).max() < 1e-12
        g = rng.normal(size=TH.shape) + 1j * rng.normal(size=TH.shape)
        assert np.vdot(g, tf.synth(s, x)) == pytest.approx(np.vdot(tf.adjoint(s, g), x))


def test_half_eval_adjoint_is_the_basis_adjoint():
    rng = np.random.default_rng(1)
    th, ph = np.arccos(rng.uniform(-1, 1, 300)), rng.uniform(-np.pi, np.pi, 300)
    g = rng.normal(size=300) + 1j * rng.normal(size=300)
    for s in (0.5, -0.5):
        B = sw.half_eval_basis(s, 6.5, th, ph)
        assert np.abs(sw.half_eval_adjoint(s, 6.5, th, ph, g) - B.conj().T @ g).max() < 1e-12


@pytest.mark.parametrize("L", [6, 10])
def test_gate_p9_b_coupling_is_hermitian(L):
    """Coupling gate: catches conjugation errors in the ρ coupling, which
    would otherwise show up only as a subtly wrong spinor."""
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]
    Ls = ds.spinor_band(L)
    grid = ds.rho_on_grid(rho, L, Ls)
    for s in (0.5, -0.5):
        M = ds.coupling_matrix(grid, s, Ls, L)
        assert np.abs(M - M.conj().T).max() < 1e-10


def _coupling_on(rho_hat, L, Ls, s, th, ph, w):
    """``B* diag(rho w) B^T`` on an arbitrary product grid."""
    from shapegen.quad.field import real_to_complex
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    C = real_to_complex(np.asarray(rho_hat, float), L)
    Y = sw.eval_basis(0, L, TH.ravel(), PH.ravel())
    rho = (Y @ np.array([C[l, l + m] for l, m in sw.mode_list(0, L)])).real + 1.0
    B = sw.half_eval_basis(s, Ls, TH.ravel(), PH.ravel())
    W = np.broadcast_to(w, TH.shape).ravel()
    return B.conj().T @ ((rho * W)[:, None] * B)


def test_coupling_quadrature_is_exact():
    """``conj(Y_l') rho Y_l`` has degree ``2 Ls + L``; ``half_grid(Ls)`` is
    exact only to ``2 Ls``. :func:`coupling_grid` must agree with a grid
    padded far beyond need, and ``half_grid`` must not -- otherwise this test
    is not seeing the aliasing it guards."""
    L = 8
    Ls = ds.spinor_band(L, 2)
    rho = bundle_style_coeffs(L=32, power=-1.0, seed=3)[:n_coeffs(L)] * 3.0
    x, wg = np.polynomial.legendre.leggauss(3 * L + 30)
    nph = 6 * L + 60
    ref_grid = (np.arccos(x), 2 * np.pi * np.arange(nph) / nph,
                wg[:, None] * (2 * np.pi / nph))
    for s in (0.5, -0.5):
        ref = _coupling_on(rho, L, Ls, s, *ref_grid)
        M = ds.coupling_matrix(ds.rho_on_grid(rho, L, Ls), s, Ls, L)
        assert np.abs(M - ref).max() < 1e-11 * np.abs(ref).max()
        old = _coupling_on(rho, L, Ls, s, *sw.half_grid(Ls))
        assert np.abs(old - ref).max() > 1e-3 * np.abs(ref).max()


def test_gate_p9_c_branch_points():
    """Branch-point gate: ``min|λ|`` approaching zero means the spinor vanishes
    somewhere and no surface exists for that ρ. It must fall as the curvature
    is driven harder, and stay at 1 on the round sphere, where λ is
    constant."""
    L = 10
    base = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]
    assert ds.min_spinor_modulus(ds.solve_spinor(np.zeros(n_coeffs(L)), L)) \
        == pytest.approx(1.0, abs=1e-9)
    prev = 1.0
    for amp in (1.0, 3.0, 6.0):
        m = ds.min_spinor_modulus(ds.solve_spinor(base * amp, L))
        assert m < prev
        prev = m
    assert prev < 0.9


def test_gate_p9_d_converges_in_band():
    """Band-convergence gate: γ must settle as the band grows, for a ρ that is itself
    band-limited well below it -- otherwise the solve is resolving its own
    truncation rather than the problem."""
    src = bundle_style_coeffs(L=32, power=-2.0, seed=1)
    gammas = []
    for L in (8, 10, 12, 14):
        rho = np.zeros(n_coeffs(L))
        k = n_coeffs(6)
        rho[:k] = src[:k]                       # band-6 content, solved at band L
        gammas.append(ds.solve_spinor(rho, L).gamma)
    rel = [abs(gammas[i + 1] - gammas[i]) / gammas[-1]
           for i in range(len(gammas) - 1)]
    assert rel[-1] < rel[0]
    assert rel[-1] < 1e-3


def test_gate_p9_e_conformal_factor_matches_the_mesh_solver():
    """**Agreement with the mesh solver**, on the row that does not need a frame.

    ``ψ1``/``ψ2`` are spin ±1/2, so they are not functions on the sphere and
    cannot be compared to a per-vertex quaternion field without first
    matching frames. ``|λ|² = |ψ1|² + |ψ2|²`` *is* a scalar, and comparing it
    against :func:`shapegen.geom.dirac.spin_transform`'s own ``|λ|²`` is a
    genuine two-discretisation check of one continuum object.

    **This is what pins ``c``.** ``ρ_bundle = c·ρ_repo + 1`` was open between
    ``c = 1`` and ``c = 2``, to be pinned empirically, never assumed. Running the spectral side at ``c = 1`` and taking the
    amplitude ratio against the mesh reads it off directly: **1.0003**, flat
    across a 10x amplitude sweep. So ``c = 1``.
    """
    from scipy.spatial import cKDTree

    L = 12
    m = icosphere(5)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    Yv = real_sh_basis(V, L)
    gth, gph, _w = sw.half_grid(ds.spinor_band(L))
    GT, GP = np.meshgrid(gth, gph, indexing="ij")
    gx = np.stack([np.sin(GT) * np.cos(GP), np.sin(GT) * np.sin(GP),
                   np.cos(GT)], -1).reshape(-1, 3)
    _d, nn = cKDTree(gx).query(V)

    for amp in (0.02, 0.05, 0.10, 0.20):
        rho = bundle_style_coeffs(L=L, power=-2.0, seed=1)[:n_coeffs(L)] * amp
        lam, _g = dirac.spin_transform(V, m.faces, Yv @ rho)   # repo convention
        cf_m = (lam ** 2).sum(1)
        cf_m = cf_m / cf_m.mean()

        res = ds.solve_spinor(rho, L, c=1.0)
        cf_s = ds.conformal_factor(res).ravel()[nn]
        cf_s = cf_s / cf_s.mean()

        ratio = np.std(cf_s - 1.0) / np.std(cf_m - 1.0)
        assert ratio == pytest.approx(1.0, abs=0.01)


def test_gamma_mesh_is_linear_in_gamma_spec_not_quadratic():
    """**A correction to an earlier prediction**, which expected
    ``γ_mesh ≈ γ_spec²`` "up to the mass normalisation" on the grounds that
    the mesh operator is a Gram operator.

    Measured, the relation is **linear**: ``γ_mesh/γ_spec`` is constant to
    3% across an 8x amplitude sweep, while ``γ_mesh/γ_spec²`` swings by 60x
    over the same range. The constant is mesh-dependent (0.0128 at subdivision
    4, 0.0033 at 5 -- it scales like 1/N, which is the mass normalisation
    the prediction hedged about), so only its constancy at fixed resolution is
    asserted here.
    """
    L = 12
    m = icosphere(4)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    Yv = real_sh_basis(V, L)
    lin, quad = [], []
    for amp in (0.05, 0.1, 0.2, 0.4):
        rho = bundle_style_coeffs(L=L, power=-2.0, seed=1)[:n_coeffs(L)] * amp
        _lam, gm = dirac.spin_transform(V, m.faces, Yv @ rho)
        gs = ds.solve_spinor(rho, L).gamma
        lin.append(gm / gs)
        quad.append(gm / gs ** 2)
    assert max(lin) / min(lin) < 1.1          # constant
    assert max(quad) / min(quad) > 10.0       # very much not constant


# --------------------------------------------------------------------------- #
# The wiring                                                                  #
# --------------------------------------------------------------------------- #

def test_dirac_potential_keeps_the_topology():
    """Whatever it does to cone *positions*, the exact potential must not
    change the topology: Poincare-Hopf still forces 8 cones summing to +2."""
    L, S = 12, 4
    rho = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]
    phi = qf.curvature_potential(rho, L)
    Cmat, modes = qf.assemble_C(qf.connection_field(phi), S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    cones = qf.cones_from_field(psiG, S, L, over=4.0)
    assert len(cones) == 8
    assert sum(k for _p, k in cones) == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Measured dead end: decoding on the quad mesh instead of an icosphere        #
# --------------------------------------------------------------------------- #

def test_dirac_conditions_worse_on_a_conformal_quad_mesh():
    """**Why the lift still decodes on an icosphere and interpolates.**

    Decoding directly on the quad mesh's own triangulation is tempting: 2.3x
    faster (27.2s against 62.6s at 20k faces) and it removes the
    interpolation, the base-mesh sizing, and a whole coupling between mesh
    density and lift resolution. It also disagrees with the icosphere route by
    a median 16% of the radius, and that disagreement is the quad mesh being
    wrong, not the icosphere.

    Raw ``gamma`` cannot show this -- it scales like 1/N, so two meshes of
    different size are not comparable on it. The resolution-normalised measure
    is the mesh-agreement gate's third row: ``rho_from_lambda`` must return the
    ``rho`` that went in. Measured on a real face at L=32::

        icosphere(5)          10242 v   rel err 0.0314   tri-area spread 1.2
        icosphere(6)          40962 v   rel err 0.0100   tri-area spread 1.2
        quad mesh, triangulated 20037 v rel err 0.0515   tri-area spread 3.3

    The quad mesh carries twice ico5's vertices and conditions *worse* than
    it. A conformal mesh cannot be uniform -- that is the same inherent area
    spread conformal maps always have -- so this is a property of the object, not a
    bug to fix, and it is the reason the decode stays on a uniform sphere.

    Run small here (band 8, icosphere 3/4) so it is a regression guard rather
    than a re-measurement; the ordering is what matters.
    """
    from shapegen.geom.mesh import triangle_areas

    L = 8
    rho_hat = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]

    def roundtrip(V, F):
        rv = real_sh_basis(V, L) @ rho_hat
        lam, _g = dirac.spin_transform(V, F, rv)
        rec = dirac.rho_from_lambda(V, F, lam)
        return np.linalg.norm(rec - rv) / np.linalg.norm(rv)

    ico = icosphere(4)
    Vi = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    err_ico = roundtrip(Vi, ico.faces)

    # a quad mesh of comparable size, triangulated
    from shapegen.quad import cones as qc, grid as qg, periods as qp
    S = 4
    phi = qf.curvature_potential(rho_hat, L)
    Cmat, modes = qf.assemble_C(qf.connection_field(phi), S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    ch = qc.project_cones(qf.cones_from_field(psiG, S, L, over=4.0))
    tgt = qg.cell_for_faces(ch["z"], ch["idx"], len(Vi))
    sol = qp.solve_lattice(ch["z"], ch["idx"],
                           cell_sizes=tuple(tgt * 0.92 ** k for k in range(30)))
    assert sol["cones"] is not None
    g = qg.quad_grid(sol["cones"], ch["idx"], sol["cell"])
    Q = g.points @ ch["Rm"]
    tris = np.array([[f[0], f[k], f[k + 1]]
                     for f in g.faces for k in range(1, len(f) - 1)], dtype=np.int64)
    err_quad = roundtrip(Q, tris)

    # the quad mesh is the less uniform substrate, and it costs accuracy
    Ai, Aq = triangle_areas(Vi, ico.faces), triangle_areas(Q, tris)
    assert (np.percentile(Aq, 90) / np.percentile(Aq, 10)
            > np.percentile(Ai, 90) / np.percentile(Ai, 10))
    assert err_quad > err_ico


# --------------------------------------------------------------------------- #
# Off-grid spinor evaluation (needed by the lift)                             #
# --------------------------------------------------------------------------- #

def test_half_eval_basis_matches_the_grid_basis():
    """:func:`~shapegen.spectral.swsh.half_eval_basis` must agree exactly with
    :func:`~shapegen.spectral.swsh.half_basis` where they overlap -- it is the
    same formula, evaluated at arbitrary points instead of the quadrature
    grid."""
    L = 7.5
    th, ph, _w = sw.half_grid(L)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    for s in (0.5, -0.5):
        B = sw.half_basis(s, L)
        Be = sw.half_eval_basis(s, L, TH.ravel(), PH.ravel())
        assert np.abs(Be.T - B).max() == 0.0
        rng = np.random.default_rng(0)
        v = rng.normal(size=B.shape[0]) + 1j * rng.normal(size=B.shape[0])
        assert np.abs(sw.half_synth(v, s, L).ravel() - Be @ v).max() < 1e-12


def test_spinor_halves_are_antiperiodic_but_their_modulus_is_not():
    """Half-integer ``m`` makes ``psi1``/``psi2`` antiperiodic in ``phi`` --
    they are not functions on the sphere. Anything built from them has to be
    quadratic for the sign to cancel, which ``|lambda|^2`` is, and which is
    why it is the only part of the spinor this module compares across
    solvers."""
    L = 5.5
    th = np.array([1.0, 2.0])
    for s in (0.5, -0.5):
        B0 = sw.half_eval_basis(s, L, th, np.array([0.3, 0.3]))
        B1 = sw.half_eval_basis(s, L, th, np.array([0.3, 0.3]) + 2 * np.pi)
        assert np.abs(B1 + B0).max() < 1e-12          # flips sign
        assert np.abs(np.abs(B1) - np.abs(B0)).max() < 1e-12   # modulus does not
