"""The quaternionic Dirac solve, spectrally.

Solves for the spinor ``lambda`` of the spin transformation of the round
sphere with curvature potential ``rho``. It is the only potential the quad
pipeline uses and the spinor that :mod:`shapegen.geom.spectral_adjoint` and
:func:`shapegen.quad.lift.lift_spectral` integrate into a surface.

**The system.** Splitting ``lambda = a + b j`` gives the Kamberov-Pedit-Pinkall
/ Konopelchenko-Taimanov pair

    ethbar psi1 =  rho psi2         psi1 spin +1/2
    eth    psi2 = -rho psi1         psi2 spin -1/2

``eth``/``ethbar`` are diagonal in the spin-weighted basis
(:func:`shapegen.spectral.swsh.half_basis`); multiplication by ``rho`` is
built **by transform** (synthesise, multiply on :func:`coupling_grid`,
analyse), not from 3j symbols.

**The operator.** The solution set is the kernel of

    G(psi1, psi2) = ( eth psi2 + rho psi1 ,  ethbar psi1 - rho psi2 )

``G`` is not Hermitian (its eigenvalues go imaginary once ``l + 1/2 > rho``),
so the residual ``gamma`` is its smallest **singular** value. The mesh
reference :mod:`shapegen.geom.dirac` reports a squared (Gram) residual, so
``gamma_mesh ~ gamma**2``.

**The rho convention.** ``rho_bundle = c * rho_repo + 1`` with ``c = 1``:
the round sphere is ``rho_bundle = 1``, where the kernel
condition ``(l + 1/2)**2 = rho**2`` selects the constant spinor at
``l = 1/2``. The offset is applied in exactly one place,
:func:`rho_on_grid`.

**Gates** (``tests/test_dirac_spectral.py``): round sphere -> ``gamma``
1.3e-16 with a multiplicity-2 kernel at ``l = 1/2`` (P9.A); coupling
Hermitian to 2.8e-16 (P9.B); :func:`min_spinor_modulus` flags branch points
(P9.C); spectral convergence in band (P9.D); conformal factor agrees with
:mod:`shapegen.geom.dirac` (P9.E).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..spectral import swsh as sw

__all__ = ["DiracSpectral", "spinor_band", "potential_band_extra",
           "coupling_grid", "rho_on_grid", "coupling_matrix", "SpinorTransform",
           "DiracOperator", "dirac_operator", "dirac_matrix", "solve_spinor",
           "lu_kernel_pair", "signed_gamma", "close_potential", "conformal_factor",
           "log_conformal_coeffs",
           "min_spinor_modulus"]


def potential_band_extra(L: int) -> int:
    """Degrees to pad the spinor band beyond ``L`` before taking
    :func:`log_conformal_coeffs` at band ``L``: ``max(4, L // 4)``.

    The padding scales with ``L`` because the Galerkin truncation it
    compensates for is a ``rho`` coupling that moves degree ``l`` to
    ``l +/- L`` (see :func:`spinor_band`). Measured on the exact
    :func:`coupling_grid` at L=32 on two FaceScape heads, the potential's
    relative A-error against the decoded surface is 0.10 / 0.07 / 0.04 /
    0.033 at ``extra`` = 4 / 6 / 8 / 12, where ~0.033 is the referee's own
    floor; 8 is the knee, 12 costs 1.8x the solve time.
    """
    return max(4, L // 4)


def spinor_band(L: int, extra: int = 0) -> float:
    """Half-integer spinor band ``L + extra - 1/2`` for an integer ``rho``
    band ``L`` (``extra = 0``: 1056 modes per spin at L=32).

    **Pass** ``extra = potential_band_extra(L)`` **whenever the spinor feeds
    :func:`log_conformal_coeffs` at band** ``L``. The spinor itself is
    resolved at ``L - 1/2``; what is not is the Galerkin system's top rows,
    whose ``rho``-coupled partners at ``l + L`` are truncated. ``|lambda|^2``
    beats that residue against the dominant ``l = 1/2`` mode and deposits it
    at degree ``L`` -- the top of the potential's band, which ``edth`` weights
    most. On a smooth fixture at L=12/16 one extra degree removes 95% of the
    error and four converge; real faces at L=32 need ``L // 4``. The cost is
    ``(1 + extra/L)^2`` in modes.
    """
    return float(L + extra) - 0.5


def coupling_grid(Ls: float, L: int):
    """``(theta, phi, weights)`` exact for the ``rho`` coupling at spinor band
    ``Ls`` and ``rho`` band ``L``.

    :func:`~shapegen.spectral.swsh.half_grid` is exact for a product of *two*
    spinor modes, degree ``2 Ls``. The coupling integrand
    ``conj(Y_l') rho Y_l`` has degree ``2 Ls + L`` in ``cos theta`` (the
    half-angle factors pair up into integer powers because
    ``|m-s| + |m'-s| + |m-m'|`` is even), so it needs
    ``ceil((2 Ls + L + 1) / 2)`` Gauss-Legendre nodes and more than
    ``2 Ls + L`` azimuthal samples. Trap: ``half_grid`` would alias the top
    ~30 degrees of the coupling at ``L = 32``, ``Ls = 39.5``.
    """
    return _coupling_grid(float(Ls), int(L))


@lru_cache(maxsize=None)
def _coupling_grid(Ls: float, L: int):
    nth = int(np.ceil((2.0 * Ls + L + 1.0) / 2.0))
    nph = int(np.floor(2.0 * Ls)) + L + 2
    x, w = np.polynomial.legendre.leggauss(nth)
    theta = np.arccos(x)
    phi = 2.0 * np.pi * np.arange(nph) / nph
    return theta, phi, w[:, None] * (2.0 * np.pi / nph)


class SpinorTransform:
    """Spin ±1/2 synthesis and its adjoint on a product grid -- ``theta``
    nodes times ``nph`` equispaced ``phi`` -- done separably: a per-``m``
    Wigner-d table in ``theta``, then a trig sum in ``phi``.

    Both the ``rho`` coupling (:func:`coupling_matrix`) and the continuum
    lift (:class:`shapegen.geom.spectral_adjoint.SpectralLift`) use it, each on
    its own grid. Coefficients are in :func:`~shapegen.spectral.swsh.half_modes`
    order and ``synth`` agrees with
    :func:`~shapegen.spectral.swsh.half_eval_basis`. Setup is a few MB, where
    a dense basis on the coupling grid is ~170 MB per spin at L=32.
    """

    def __init__(self, Ls: float, theta: np.ndarray, nph: int):
        from math import sqrt

        self.Ls, self.theta, self.nph = float(Ls), np.asarray(theta), int(nph)
        ph = 2.0 * np.pi * np.arange(nph) / nph
        self._half = {}
        for s in (+0.5, -0.5):
            modes = sw.half_modes(s, Ls)
            row = {lm: i for i, lm in enumerate(modes)}
            ms = np.arange(-Ls, Ls + 0.5, 1.0)
            nL = int(round(Ls + 0.5))
            D = np.zeros((len(ms), len(self.theta), nL))
            idx = np.full((len(ms), nL), len(modes), dtype=np.intp)   # pad -> row n
            for a, m in enumerate(ms):
                ls, d = sw.wigner_d_ladder(Ls, m, -s, self.theta)
                for i, l in enumerate(ls):
                    D[a, :, i] = sqrt((2 * l + 1) / (4 * np.pi)) * d[i]
                    idx[a, i] = row[(l, m)]
            Ph = np.exp(1j * ms[:, None] * ph[None, :])
            self._half[s] = (D, idx, ms, Ph)
        self.n = len(sw.half_modes(0.5, Ls))

    def synth(self, s: float, x: np.ndarray) -> np.ndarray:
        """Spin-``s`` coefficients ``(n,)`` -> field on the grid ``(Nθ, Nφ)``."""
        D, idx, _ms, Ph = self._half[s]
        return np.einsum("mtl,ml->tm", D, np.r_[x, 0.0][idx]) @ Ph

    def adjoint(self, s: float, g: np.ndarray) -> np.ndarray:
        """The adjoint of :meth:`synth` (no quadrature weights). ``g`` is
        ``(Nθ, Nφ)`` or ``(Nθ, Nφ, B)`` for ``B`` fields at once."""
        D, idx, _ms, Ph = self._half[s]
        G = g if g.ndim == 3 else g[..., None]
        X = np.tensordot(Ph.conj(), G, axes=([1], [1]))           # (nM, Nθ, B)
        gve = np.matmul(D.transpose(0, 2, 1), X).reshape(-1, G.shape[-1])
        flat = idx.ravel()
        valid = flat < self.n                                      # every mode once; pads -> n
        out = np.empty((self.n, G.shape[-1]), complex)
        out[flat[valid]] = gve[valid]
        return out if g.ndim == 3 else out[:, 0]

    def coupling(self, s: float, rho_grid: np.ndarray, w: np.ndarray,
                 L: int) -> np.ndarray:
        """``∫ conj(Y_a) rho Y_b`` with weights ``w`` ``(Nθ, 1)``, for ``rho``
        of band ``L``: FFT ``rho`` in ``phi``, then one batched product per
        ``Δm = m_a − m_b`` with ``|Δm| ≤ L`` (the rest vanish)."""
        D, idx, ms, _Ph = self._half[s]
        assert self.nph > 2 * L, "phi grid aliases the coupling"
        R = np.fft.fft(rho_grid, axis=1) * w
        Dt = np.ascontiguousarray(D.transpose(0, 2, 1))
        nM = len(ms)
        M = np.zeros((self.n + 1, self.n + 1), complex)
        for dm in range(-min(L, nM - 1), min(L, nM - 1) + 1):
            a = np.arange(max(0, dm), min(nM, nM + dm))
            M[idx[a][:, :, None], idx[a - dm][:, None, :]] = (Dt[a] * R[:, dm % self.nph]) @ D[a - dm]
        return M[:self.n, :self.n]


@lru_cache(maxsize=4)
def _coupling_transform(Ls: float, L: int) -> SpinorTransform:
    th, ph, _w = coupling_grid(Ls, L)
    return SpinorTransform(Ls, th, len(ph))


@lru_cache(maxsize=4)
def _rho_basis(Ls: float, L: int) -> np.ndarray:
    """Real SH of band ``L`` on :func:`coupling_grid`, ``(Nθ·Nφ, (L+1)²)``."""
    from .encode import real_sh_basis

    th, ph, _w = coupling_grid(Ls, L)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    pts = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                    np.cos(TH)], -1).reshape(-1, 3)
    return real_sh_basis(pts, L)


def rho_on_grid(rho_hat: np.ndarray, L: int, Ls: float, c: float = 1.0):
    """This repository's real-SH ``rho_hat`` -> the bundle's ``rho``, sampled
    on :func:`coupling_grid` ``(Ls, L)``.

    ``rho_bundle = c * rho_repo + 1``. **The one place that
    offset is applied** -- twice, or again inside a re-measurement pass,
    shifts the realised curvature by a constant and the failure reads as a
    scale error in the output rather than as a bug. Its derivative in
    ``rho_hat`` is ``c`` times :func:`_rho_basis`.
    """
    th, ph, _w = coupling_grid(Ls, L)
    field = _rho_basis(float(Ls), int(L)) @ np.asarray(rho_hat, float)
    return c * field.reshape(len(th), len(ph)) + 1.0


def coupling_matrix(rho_grid: np.ndarray, s: float, Ls: float,
                    L: int) -> np.ndarray:
    """Multiplication by ``rho`` as a matrix in the spin-``s`` basis.

    Built by transform -- ``B* diag(rho . w) B^T`` -- separably
    (:meth:`SpinorTransform.coupling`). Hermitian for real ``rho`` (gate
    P9.B, the cheapest catch for a conjugation error). ``rho_grid`` is
    :func:`rho_on_grid`'s output for the same ``(L, Ls)``; the quadrature is
    :func:`coupling_grid`, exact for the triple product.
    """
    _th, _ph, w = coupling_grid(Ls, L)
    return _coupling_transform(float(Ls), int(L)).coupling(s, rho_grid, w, int(L))


class DiracOperator:
    """``G`` above in blocks, factorised on construction:

        G = [[ Mp, E ], [ Ē, −Mm ]],   E = diag(ð), Ē = diag(ð̄)

    ``E``/``Ē`` are ``±(ℓ + 1/2)``, never below 1 in modulus, so ``G`` is
    solved through the ``n × n`` Schur complement ``S = Ē + Mm E⁻¹ Mp``
    rather than a ``2n × 2n`` LU: 3.3x cheaper at L=32 with the same
    solutions (1e-12 relative, the conditioning of a near-singular ``G``).
    ``solve``/``solve_h`` apply ``G⁻¹``/``G⁻*`` to ``(2n,)`` or ``(2n, k)``.
    """

    def __init__(self, Mp: np.ndarray, Mm: np.ndarray, Ls: float):
        import scipy.linalg as sla

        self.A, self.B = Mp, Mm
        self.e = sw.half_edth_factor(-0.5, Ls).astype(complex)       # eth : -1/2 -> +1/2
        self.eb = sw.half_edthbar_factor(+0.5, Ls).astype(complex)   # ethbar: +1/2 -> -1/2
        self.n = Mp.shape[0]
        self._BE = Mm / self.e[None, :]
        self._lu = sla.lu_factor(np.diag(self.eb) + self._BE @ Mp, check_finite=False)

    def _split(self, X):
        d = (slice(None),) + (None,) * (X.ndim - 1)
        return X[:self.n], X[self.n:], d

    def apply(self, X):
        x1, x2, d = self._split(X)
        return np.concatenate([self.A @ x1 + self.e[d] * x2, self.eb[d] * x1 - self.B @ x2])

    def apply_h(self, Y):
        y1, y2, d = self._split(Y)
        return np.concatenate([self.A @ y1 + self.eb.conj()[d] * y2,
                               self.e.conj()[d] * y1 - self.B @ y2])

    def solve(self, b):
        import scipy.linalg as sla

        b1, b2, d = self._split(b)
        x1 = sla.lu_solve(self._lu, b2 + self._BE @ b1, check_finite=False)
        return np.concatenate([x1, (b1 - self.A @ x1) / self.e[d]])

    def solve_h(self, b):
        import scipy.linalg as sla

        b1, b2, d = self._split(b)
        t1 = b2 / self.e.conj()[d]
        t2 = sla.lu_solve(self._lu, b1 - self.A @ t1, trans=2, check_finite=False)
        return np.concatenate([t1 + (self.B @ t2) / self.e.conj()[d], t2])

    def dense(self) -> np.ndarray:
        n = self.n
        G = np.zeros((2 * n, 2 * n), dtype=complex)
        G[:n, :n], G[:n, n:] = self.A, np.diag(self.e)
        G[n:, :n], G[n:, n:] = np.diag(self.eb), -self.B
        return G


def dirac_operator(rho_grid: np.ndarray, Ls: float, L: int) -> DiracOperator:
    """``G`` for ``rho_grid`` (:func:`rho_on_grid`), factorised."""
    return DiracOperator(coupling_matrix(rho_grid, +0.5, Ls, L),
                         coupling_matrix(rho_grid, -0.5, Ls, L), Ls)


def dirac_matrix(rho_grid: np.ndarray, Ls: float, L: int) -> np.ndarray:
    """The operator ``G`` above, as a dense ``(2n, 2n)`` complex matrix with
    the spin ``+1/2`` block first -- for the SVD reference and tests; the solve
    itself uses :class:`DiracOperator`."""
    return dirac_operator(rho_grid, Ls, L).dense()


@dataclass
class DiracSpectral:
    """The solve's result, with the diagnostics its gates are stated in."""
    psi1: np.ndarray        # spin +1/2 coefficients, half_modes order
    psi2: np.ndarray        # spin -1/2 coefficients
    gamma: float            # smallest singular value of G
    gap: float              # the next one -- the kernel is simple iff this is big
    multiplicity: int       # singular values within 1e-8 of gamma
    Ls: float
    L: int
    c: float
    niter: int = 0          # inverse-iteration steps (0 for method="svd")
    # the factorised operator and the kernel pair (2n, 2): what the encoder's
    # backward pass reuses (method="lu" only)
    op: "DiracOperator | None" = field(default=None, repr=False, compare=False)
    pair: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def gamma_squared(self) -> float:
        """Comparable to :mod:`shapegen.geom.dirac`'s Gram residual
        ``gamma_mesh``."""
        return self.gamma ** 2


def solve_spinor(rho_hat: np.ndarray, L: int, c: float = 1.0,
                 Ls: float | None = None, v0=None, method: str = "lu",
                 with_gap: bool = False, tol: float = 1e-11,
                 maxiter: int = 50) -> DiracSpectral:
    """Solve ``(D - rho) lambda = gamma lambda`` for the smallest ``|gamma|``.

    ``rho_hat`` is this repository's real-SH ``rho`` at band ``L``; ``Ls``
    defaults to ``spinor_band(L)`` (pass ``spinor_band(L,
    potential_band_extra(L))`` if the result feeds
    :func:`log_conformal_coeffs`). Returns the two spin-weighted halves and
    ``gamma``. The realised curvature is ``rho - gamma``, so on real face
    codes, where ``gamma`` is not small, it describes the output rather than
    a convergence failure.

    ``method="lu"`` (default): one factorisation of ``G``
    (:class:`DiracOperator`), then block inverse iteration on ``G* G`` with a
    Rayleigh-Ritz step, stopping when the kernel pair's residual
    ``||G*G V - V gamma^2||`` is below ``tol`` (~3 steps cold). 0.4 s at L=32
    against 33 s for the dense SVD, ``gamma`` equal to the SVD's to ~1e-14
    relative. Exactly singular ``G`` (round sphere) is fine. The result keeps
    the operator and the pair (``op``, ``pair``) for the encoder's adjoint.

    ``with_gap``: ``gap`` is ``nan`` unless asked for. The next singular
    value converges after the pair (stopping on the pair alone once reported
    2.0 for a true 1.0), and settling it roughly doubles the cost.

    ``v0``: a previous :class:`DiracSpectral` or ``(2n,)``/``(2n, k)``
    block to start from. Buys nothing measurable; building ``G`` and the LU
    dominate.

    ``method="svd"``: the dense SVD, the reference the tests hold ``"lu"`` to.
    """
    Ls = spinor_band(L) if Ls is None else Ls
    op = dirac_operator(rho_on_grid(rho_hat, L, Ls, c), Ls, L)
    n = op.n
    if method == "svd":
        _U, S, Vh = np.linalg.svd(op.dense())
        v = Vh[-1].conj()
        mult = int((S < S[-1] + 1e-8).sum())
        return DiracSpectral(psi1=v[:n], psi2=v[n:], gamma=float(S[-1]),
                             gap=float(S[-mult - 1]) if mult < len(S) else float("inf"),
                             multiplicity=mult, Ls=Ls, L=L, c=c)
    if method != "lu":
        raise ValueError(f"unknown method {method!r}")
    if isinstance(v0, DiracSpectral):
        v0 = np.r_[v0.psi1, v0.psi2]
    sv, X, it = lu_kernel_pair(op, v0=v0, with_gap=with_gap, tol=tol,
                               maxiter=maxiter)
    v = X[:, 0]
    mult = int((sv < sv[0] + 1e-8).sum())
    return DiracSpectral(psi1=v[:n], psi2=v[n:], gamma=float(sv[0]),
                         gap=(float(sv[mult]) if mult < len(sv) else float("inf"))
                         if with_gap else float("nan"),
                         multiplicity=mult, Ls=Ls, L=L, c=c, niter=it,
                         op=op, pair=X[:, :2])


def signed_gamma(res: DiracSpectral) -> float:
    """``gamma`` with its sign: the eigenvalue ``mu`` nearest zero of the
    Hermitian form ``sigma3 G`` (``sigma3 = diag(I, -I)``), with ``|mu| = gamma``.

    ``sigma3 G = K + rho`` with ``K = [[0, eth], [-ethbar, 0]]`` Hermitian
    (``eth* = -ethbar``), so a constant added to ``rho`` shifts every eigenvalue
    by that constant and leaves the eigenvectors, hence the surface, unchanged.
    That is why ``l = 0`` of ``rho_hat`` is a null direction of the decode, and
    what :func:`close_potential` uses.
    """
    if res.op is None:
        raise ValueError("signed_gamma needs a method='lu' solve (it keeps the operator)")
    v = np.r_[res.psi1, res.psi2]
    n = len(res.psi1)
    Gv = res.op.apply(v)
    return float(np.real(np.vdot(v[:n], Gv[:n]) - np.vdot(v[n:], Gv[n:]))
                 / np.real(np.vdot(v, v)))


def close_potential(rho_hat: np.ndarray, res: DiracSpectral) -> np.ndarray:
    """``rho_hat`` with its ``l = 0`` coefficient set so that ``gamma = 0``
    exactly, i.e. so the code is the curvature of an actual immersion.

    The surface does not change (see :func:`signed_gamma`): only the constant
    part of ``rho`` moves, by ``-mu``. ``res`` is ``solve_spinor(rho_hat, ...)``.
    Measured: ``gamma`` 1.2e-2 -> 9e-16 with the surface equal to 4e-23.
    """
    out = np.array(rho_hat, dtype=float)
    out[0] -= signed_gamma(res) * np.sqrt(4.0 * np.pi) / res.c    # Y_00 = 1/sqrt(4 pi)
    return out


def lu_kernel_pair(op: DiracOperator, v0=None, with_gap: bool = False,
                   tol: float = 1e-11, maxiter: int = 50):
    """The inverse iteration behind :func:`solve_spinor`'s ``method="lu"``:
    ``(sv, X, niter)`` with ``sv`` ascending Ritz singular values and
    ``X[:, :2]`` the kernel pair. ``op`` carries the factorisation, which
    :mod:`shapegen.geom.spectral_adjoint` reuses for the backward pass."""
    n2 = 2 * op.n
    k = 6 if with_gap else 4
    X = np.random.default_rng(0).standard_normal((n2, k)) + 0j
    if v0 is not None:
        v0 = np.asarray(v0).reshape(n2, -1)[:, :k]
        X[:, :v0.shape[1]] = v0
    prev_gap = np.inf
    for it in range(1, maxiter + 1):
        X = op.solve(op.solve_h(X))                           # (G* G)^-1 X
        Q, _ = np.linalg.qr(X)
        _u, sv, wh = np.linalg.svd(op.apply(Q), full_matrices=False)
        X, sv = Q @ wh.conj().T[:, ::-1], sv[::-1]            # ascending
        V = X[:, :2]
        settled = not with_gap or abs(sv[2] - prev_gap) <= 1e-6 * sv[2]
        prev_gap = sv[2]
        if settled and np.linalg.norm(op.apply_h(op.apply(V)) - V * sv[:2] ** 2) < tol:
            break
    else:
        import warnings
        warnings.warn(f"solve_spinor: inverse iteration did not reach tol={tol} "
                      f"in {maxiter} steps", RuntimeWarning, stacklevel=3)
    return sv, X, it


def conformal_factor(res: DiracSpectral, theta=None, phi=None) -> np.ndarray:
    """``|lambda|^2 = |psi1|^2 + |psi2|^2`` -- the conformal factor of the
    immersion, on the solve's own grid or at given ``(theta, phi)``.

    Frame-free: ``psi1``/``psi2`` are spin ±1/2 and pick up phases
    ``e^{∓i alpha/2}`` under a frame rotation, but their moduli do not, so
    ``|lambda|^2`` is a scalar field. That makes it the quantity compared
    with the mesh reference (gate P9.E) and the one
    :mod:`shapegen.quad.field`'s connection is built from.
    """
    f1 = sw.half_synth(res.psi1, +0.5, res.Ls)
    f2 = sw.half_synth(res.psi2, -0.5, res.Ls)
    return np.abs(f1) ** 2 + np.abs(f2) ** 2


def log_conformal_coeffs(res: DiracSpectral, L: int | None = None) -> np.ndarray:
    """``log|lambda|^2`` as spin-0 coefficients in :mod:`swsh`'s dense
    ``C[l, l+m]`` layout -- the exact conformal potential, for
    :func:`shapegen.quad.field.curvature_potential`.

    The ``l = 0`` mode is dropped, because the immersion is only defined up
    to scale and the connection only sees ``eth phi`` anyway.

    **Trap:** ``res`` must be solved at ``spinor_band(L,
    potential_band_extra(L))``, not ``spinor_band(L)``; at ``Ls = L - 1/2``
    the degree-``L`` coefficients come back 6-9x too large (warned below).
    That is truncation, not quadrature -- see :func:`spinor_band`.

    Quadrature is ``half_grid(res.Ls)``, which is ample for a band-``L``
    spin-0 projection once ``Ls > L`` (unchanged against a 6x-padded grid).
    """
    L = int(res.L) if L is None else int(L)
    if res.Ls < L + 0.5:
        import warnings
        warnings.warn(
            f"log_conformal_coeffs at band {L} from a spinor solved at "
            f"Ls={res.Ls}: the top degrees will be truncation residue, not "
            f"signal. Solve with Ls=spinor_band({L}, potential_band_extra({L})).",
            RuntimeWarning, stacklevel=2)
    th, ph, w = sw.half_grid(res.Ls)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    f = np.log(np.maximum(conformal_factor(res), 1e-300))
    Y = sw.eval_basis(0, L, TH.ravel(), PH.ravel())
    # w is a COLUMN (Nθ,1) that broadcasts over φ; it must be broadcast before
    # ravelling, never ravelled on its own
    flat = Y.conj().T @ (f * w).ravel()
    C = sw.empty_coeffs(L)
    for i, (l, m) in enumerate(sw.mode_list(0, L)):
        C[l, l + m] = flat[i]
    C[0, 0] = 0.0
    return C


def min_spinor_modulus(res: DiracSpectral) -> float:
    """**Gate P9.C.** ``min |lambda|`` over the grid, normalised by its mean.

    Approaching zero means the spinor vanishes somewhere, and where
    ``lambda = 0`` the immersion degenerates -- no surface exists for that
    ``rho``.
    """
    cf = conformal_factor(res)
    _th, _ph, w = sw.half_grid(res.Ls)
    W = np.broadcast_to(w, cf.shape)
    return float(np.sqrt(cf.min() / ((cf * W).sum() / W.sum())))
