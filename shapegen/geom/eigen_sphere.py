"""Conformal sphere map from the input mesh's own Laplace eigenbasis.

**The idea.** Dirichlet energy is conformally invariant in 2D, so the input's
cotan stiffness ``L0`` is also the stiffness of its conformal image on the round
sphere; only the *mass* differs. Solve the generalised eigenproblem
``L0 φ = λ N φ`` with ``N`` the round sphere's area pulled back to the input:
its eigenpairs are then spherical harmonics composed with the map, with
``λ = ℓ(ℓ+1)``, and the three ``ℓ = 1`` eigenvectors **are** the sphere
coordinates. The mass is the unknown, so iterate: eigenvectors → map → pulled-back
area → eigenvectors.

**All of it after one mesh step runs in eigenspace.** From the mesh we take K
eigenpairs of ``(L0, M0)`` and the triple products ``C_jkl = ∫ φ_j φ_k φ_l dA``.
Then the mass is ``w = Σ b_l φ_l`` with matrix ``N(b) = Σ_l b_l C[:, :, l]``
(``L0`` stays diagonal); the ``ℓ = 1`` eigenvectors of the small dense problem
``Λ0 x = ν N(b) x`` give the map as coefficients ``Y``; and for a conformal map
the pulled-back area is ``½|∇S|²``, whose coefficients follow from
``∫ ∇φ_j·∇φ_k φ_l dA = ½(λ_j + λ_k − λ_l) C_jkl`` (integration by parts). No
projection onto the sphere is needed: the fixed point lands on it by itself.

**The Möbius gauge is not optional.** Every Möbius image of a conformal map is
conformal too, and each is a fixed point of the iteration above -- a
three-parameter family (boosts). Without a gauge the iteration drifts along it
and never converges (residual stalls at ~1e-4 after 3000 steps, runs ending 35°
apart), and the drift concentrates area where K modes resolve it worse. Each
step therefore takes a first-order boost that puts the input-area centroid of
the map at the origin -- the condition :func:`shapegen.geom.conformal.mobius_normalize`
enforces. With it the fixed point is isolated and Anderson acceleration applies.

**Measured** (two FaceScape heads, 10242 vertices, K=100): converged to 1e-10 in
16 Anderson steps (0.08 s; 54 plain steps); setup 0.5 s (eigensolve 0.4 s,
triple products 0.1 s). :func:`spectrum_error` 0.0016, the best of the
correspondences tried (the mean-curvature-flow map it replaced scored
0.04-0.05). This is the codec's default sphere (:func:`shapegen.codec.base_sphere`).
"""

from __future__ import annotations

import numpy as np

__all__ = ["conformal_sphere", "spectrum_error"]

FOURPI = 4.0 * np.pi


def _gen_eig(L0, m, k):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    lam, phi = spla.eigsh(L0, k=k, M=sp.diags(m), sigma=-1e-3, which="LM")
    o = np.argsort(lam)
    return lam[o], phi[:, o]


def _triple_products(phi, m):
    """``C_jkl = Σ_i m_i φ_ij φ_ik φ_il`` as K matrix products (an ``einsum``
    over all four indices is ~140x slower)."""
    K = phi.shape[1]
    pm = phi * m[:, None]
    C = np.empty((K, K, K))
    for j in range(K):
        C[j] = (pm * phi[:, j:j + 1]).T @ phi
    return C


def conformal_sphere(V: np.ndarray, F: np.ndarray, K: int = 100,
                     tol: float = 1e-10, max_iter: int = 200, m_hist: int = 6,
                     return_info: bool = False):
    """Map ``(V, F)`` conformally onto the unit sphere; returns ``S (n,3)``.

    ``K``: eigenmodes carried. 100 is enough on a face (K=200 moves the map by
    no more than the K=100 map moves between solvers); memory ``2·K³`` doubles.
    ``m_hist``: Anderson history (0 = plain fixed-point iteration). The map is
    Möbius-normalised against the input's areas (``mobius_normalize``'s gauge)
    and oriented like the input; the ``ℓ = 1`` eigenvectors are only defined up
    to an orthogonal 3x3, which can be a reflection, so orientation is checked.
    """
    import scipy.linalg as sla

    from .conformal import mobius_normalize
    from .mesh import Mesh
    from .operators import cotan_laplacian, vertex_areas

    V = np.asarray(V, float)
    F = np.asarray(F)
    if len(V) <= K + 1:                 # eigsh needs k < N
        raise ValueError(
            f"conformal_sphere: {len(V)} vertices is too coarse for K={K} eigenmodes. "
            f"Remesh first (shapegen.geom.mesh.remesh_isotropic; codec.encode does it "
            f"automatically for meshes under its refine threshold).")
    X = V - V.mean(0)
    X = X / np.sqrt(vertex_areas(X, F).sum() / FOURPI)       # area 4π, so A0 = 4π below
    L0, M0 = cotan_laplacian(X, F)
    m0 = M0.diagonal()
    lam, phi = _gen_eig(L0, m0, K)
    lam[0] = 0.0
    phi[:, 0] = np.abs(phi[:, 0])       # eigsh's sign is arbitrary; negative makes N indefinite
    C = _triple_products(phi, m0)
    D = 0.5 * (lam[:, None, None] + lam[None, :, None] - lam[None, None, :]) * C
    Lam = np.diag(lam)
    sqA = np.sqrt(FOURPI)

    def g(b):
        _nu, Z = sla.eigh(Lam, np.einsum("l,jkl->jk", b, C))
        Y = Z[:, 1:4] * np.sqrt(FOURPI / 3.0)                  # ∫x² dA = 4π/3 on the unit sphere
        bn = 0.5 * np.einsum("jc,kc,jkl->l", Y, Y, D)          # coefficients of ½|∇S|²
        bn *= FOURPI / (bn[0] * sqA)
        # Möbius gauge: input-area centroid ∫S dA0 = √A·Y[0] to the origin by a
        # first-order boost a, which moves it by (A·I − YᵀY)a and scales w by (1 − 2a·S)
        a = -np.linalg.solve(FOURPI * np.eye(3) - Y.T @ Y, sqA * Y[0])
        bn = bn - 2.0 * np.einsum("j,kc,jkl->lc", bn, Y, C) @ a
        return bn * FOURPI / (bn[0] * sqA), Y

    b = np.zeros(K)
    b[0] = sqA                                                  # w ≡ 1 (φ0 = 1/√A)
    Gs, Fs, Y, res, fails = [], [], None, np.inf, 0
    for it in range(1, max_iter + 1):
        try:
            gb, Y = g(b)
        except np.linalg.LinAlgError:     # N lost definiteness: back off to the last good image
            fails += 1
            if not Gs:
                raise
            b, Gs, Fs = Gs[-1], [], []
            continue
        f = gb - b
        res = float(np.linalg.norm(f) / np.linalg.norm(b))
        if res < tol:
            break
        Gs.append(gb)
        Fs.append(f)
        Gs, Fs = Gs[-(m_hist + 1):], Fs[-(m_hist + 1):]
        if m_hist == 0 or len(Fs) == 1:
            b = gb
            continue
        dF = np.diff(np.array(Fs), axis=0).T
        dG = np.diff(np.array(Gs), axis=0).T
        gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
        b = gb - dG @ gam
    else:
        import warnings
        warnings.warn(f"conformal_sphere: residual {res:.1e} after {max_iter} steps",
                      RuntimeWarning, stacklevel=2)

    S = phi @ Y
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    nrm = np.cross(S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]])
    if ((nrm * S[F].mean(1)).sum(1) < 0).sum() > len(F) // 2:
        S = S * np.array([1.0, 1.0, -1.0])                      # the eigenbasis came out mirrored
    S = mobius_normalize(Mesh(V, F), S)
    if return_info:
        nrm = np.cross(S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]])
        return S, {"iters": it, "residual": res, "fails": fails, "K": K,
                   "flips": int(((nrm * S[F].mean(1)).sum(1) < 0).sum())}
    return S


def spectrum_error(V: np.ndarray, F: np.ndarray, S: np.ndarray, lmax: int = 5) -> float:
    """How far ``S`` is from conformal: the relative rms deviation of the
    spectrum of ``(L0, M_sphere)`` from ``ℓ(ℓ+1)``, ℓ = 1..lmax. 0 iff the map
    ``V → S`` is exactly conformal (at this discretisation);
    :func:`conformal_sphere` scores 0.0016 on a scanned head."""
    from .operators import cotan_laplacian, vertex_areas

    ell = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(lmax + 1)])
    L0, _ = cotan_laplacian(np.asarray(V, float), F)
    lam, _ = _gen_eig(L0, vertex_areas(S, F), len(ell))
    return float(np.sqrt(np.mean(((lam[1:] - ell[1:]) / ell[1:]) ** 2)))
