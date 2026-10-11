"""The fast polyhedral encode: a discrete conformal sphere map corrected by local
periods, and a piecewise-Möbius spinor. ``codec.encode(method="polyhedral")`` on
meshes above ``poly_conformal.EXACT_MAX_VERTICES``.

The exact map (:mod:`~shapegen.geom.edge_periods`) asks every edge's period
``|∫ K Π_k (w − w_k)^{a_k} dw|`` along its great arc to be its 3D length. It is
an N-body integral per edge, and on a scanned head its Newton solve takes ~20 min.
This module solves a sparse model of the same equations instead and lands
~4e-4 rad (median) from the exact map on that head in ~4 s; the code decodes as
well as the exact one.

1. **Chord model.** ``log ℓ_e = log |X_i − X_j| + (u_i + u_j)/2`` on the mesh's
   own triangulation: discrete conformal equivalence (Bobenko, Pinkall &
   Springborn 2015) *without* intrinsic flips. With flips (the discrete
   uniformization, :func:`~shapegen.geom.uniformize.uniformize`) the chord
   lengths of a triangulation that is not the convex hull of ``X`` are a different
   metric, and the map is 2.8e-3 rad from the exact one; without them 1e-3.
   Two Newton steps on ``(X, u)`` from the (converged) uniformization, each by its
   own sparse LU, backtracked on the rows' sum of squares. (A flip-free
   uniformization, the same equations
   as a convex problem in ``u``, cannot turn a triangle over; with the
   corrections below it makes triangles degenerate instead and gets worse.)
2. **Local periods.** The exact rows are ``log ℓ_e = log c + (φ_i + φ_j)/2 +
   log ρ_e`` with ``φ_i = Σ_{k≠i} a_k log chord(X_i, X_k)`` and ``ρ_e`` the
   period over the chord with each end's ``φ`` divided out. ``φ`` is per vertex,
   which the chord model's ``u`` absorbs; in ``ρ_e`` the far cones cancel to second
   order, so :func:`local_log_ratio` computes it from the two ends' 1-rings (the
   complex period in the arc's midpoint chart; with every cone near it is exact),
   and :func:`far_log_ratio` adds the far cones' smooth second-order share. One
   update and one more Newton step (GMRES on the second LU: the corrected rows
   have the same Jacobian) take the head's map from 1e-3 to 3.7e-4 rad and a smooth
   2562-vertex ellipsoid's from 1e-3 to 4e-5 (the far share alone: 5e-4 → 4e-5);
   more updates gain nothing.
3. **Spinor.** On each face the map from the sphere to the 3D triangle is the
   Möbius transformation through its three corners (:class:`MobiusFaces`):
   discretely conformal triangles are related by exactly that, and neighbouring
   faces' maps agree on their shared edge. λ is its derivative's rotation and
   scale. A constant λ per face (the best-fit similarity of the chord triangle)
   lost 7–9% on the head's face region; the Möbius λ on the exact map matches the
   exact quadrature's code (face 0.0249 vs 0.0250 at L=32).
4. **Quadrature.** Each sphere triangle is split into ``n²`` sub-triangles,
   ``n ≥ Ls · side``, a node at each centroid weighted by its spherical area. One
   node per face cost 2× in decode error: a few hundred large faces (sides to 0.3
   rad) carry the high bands. ``n ≥ Ls · side / 4`` was enough on the head but
   cost 25% on the ellipsoid at L=16.

**Measured** at L=32. FaceScape head (26k vertices): vertex error 0.0684, face
region 0.0260, surface 0.0688; the exact map's encode 0.0684 / 0.0250 / 0.0688.
~3.9 s end to end through the codec, single-threaded (also with BLAS on one
thread): uniformization (with the row setup) 0.8 s, chord solve 0.95 s (two LUs
of the 79k-unknown system, 0.43 s each), periods 0.6 s, correction 0.17 s,
gauge, quadrature and code 1.0 s, scoring 0.2 s (COLAMD LUs; not re-timed since
the nested-dissection order). Smoothed icosphere-6 heads (``rho_ico6``, 40962
vertices): 6–7 s per encode at L = 16 and 32 — uniformization 1.0 s, chord solve
3.6 s (two LUs of the 123k-unknown system, 1.8 s each by :func:`_dissection_columns`;
3.7 s each under COLAMD), periods 0.7 s, code 0.4 s (L=16) / 1.0 s (L=32). They
took ~62 s before uniformize's minimum-degree ordering was relabelled (53 s of it
in SuperLU's ordering on the icosphere numbering). Smooth 2562-vertex ellipsoid:
surface error 9.8e-5 vs 7.8e-5 exact (L=16: 2.32e-4 vs 2.24e-4), 0.9 s vs 10 s;
codes agree to 0.2–0.7% on smooth or finely tessellated meshes (also a cube
tessellated 12×12 per side). A 9-vertex hull flat-subdivided to 1800 vertices:
3.23e-2 vs 3.17e-2.

Coarse polyhedra are the exact path's (``EXACT_MAX_VERTICES``): a face is then a
large part of the sphere and its λ is singular at the corners
(``|ζ − p|^{a/2}``), which a Möbius map is not (a cube, a box, a 9-vertex hull,
a 12-gon prism: 3–12× the exact surface error at L=16).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import poly_conformal as pc

__all__ = ["chord_map", "local_log_ratio", "far_log_ratio", "MobiusFaces", "sphere_nodes", "encode_chord"]

NQ = 3          # Gauss(–Jacobi) nodes per panel of the local periods (8e-6 in log ρ)
SEP = 1.0       # panels: every near cone but the end's own ≥ SEP panel lengths away
GMRES_TOL, GMRES_MAX = 1e-3, 20
FAR_BAND = 16           # the far cones' smooth field: band (and heat taper) of its expansion


# ------------------------------------------------------------------ local periods
def near_sets(E, n):
    """CSR ``(ptr, idx)``: per edge, its two ends and their 1-rings."""
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                      shape=(n, n)).tocsr()
    A = (A + sp.identity(n, format="csr")).tocsr()
    A.data[:] = 1.0
    M = (A[E[:, 0]] + A[E[:, 1]]).tocsr()
    M.sort_indices()
    return M.indptr, M.indices


def local_log_ratio(X, E, a, ptr, idx, nq: int = NQ, sep: float = SEP):
    """``log ρ_e`` (module docstring, step 2) for edges ``E`` of the sphere points
    ``X`` with cone exponents ``a`` and near sets ``(ptr, idx)``, and each edge's
    near cones' total exponent ``A_e``.

    Written in the stereographic chart ``w`` centred at the arc's midpoint, where the
    arc is the real segment ``[−τ, τ]`` (``τ = tan(β/4)``) and the metric is
    ``K |Π_k (w − w_k)^{a_k}| |dw|``: the period is the *complex* integral of the
    near cones' product (its modulus; the arc's image is not straight, and its
    length differs at second order: 1e-3 on a 200-vertex ellipsoid, 3e-2 on a
    9-vertex hull), and the per-vertex parts and constants are those of
    ``log|I_e| − log chord_e − (φ_i + φ_j)/2`` in the same chart. With every cone
    in the near set it is that exactly (:class:`~shapegen.geom.edge_periods.EdgeSystem`'s
    scale), to the quadrature error.

    The segment is split in halves and bisected until every near cone but the
    panel end's own is ``sep`` panel lengths away; a panel at an end uses
    Gauss–Jacobi for that end's ``(w − w_end)^{a}``."""
    i, j = E[:, 0], E[:, 1]
    ne = len(E)
    A_, B_ = X[i], X[j]
    m = pc._normalize(A_ + B_)
    beta = np.arctan2(np.linalg.norm(np.cross(A_, B_), axis=1), np.einsum("ij,ij->i", A_, B_))
    d_ = B_ - A_
    e1 = pc._normalize(d_ - np.einsum("ij,ij->i", d_, m)[:, None] * m)  # along the arc at m
    e2 = np.cross(m, e1)
    tau = np.tan(beta / 4)
    lt = np.log1p(tau ** 2)
    own = np.repeat(np.arange(ne), np.diff(ptr))
    k = idx
    ak = a[k]
    endi, endj = k == i[own], k == j[own]
    Xk = X[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        wk = (np.einsum("ij,ij->i", Xk, e1[own]) + 1j * np.einsum("ij,ij->i", Xk, e2[own])) \
            / (1 + np.einsum("ij,ij->i", Xk, m[own]))
    wk = np.where(endi, -tau[own], np.where(endj, tau[own], wk))
    l1k = np.log1p(np.abs(wk) ** 2)
    # constants and per-vertex parts: K's near factors, φ's near terms (chord = 2|Δw|/√((1+|w|²)(1+|w'|²)))
    Anear = np.bincount(own, ak, ne)
    with np.errstate(divide="ignore"):
        li = np.log(np.abs(-tau[own] - wk))
        lj = np.log(np.abs(tau[own] - wk))
    vi = np.where(endi, 0.0, ak * (np.where(endi, 0.0, li) - 0.5 * l1k))
    vj = np.where(endj, 0.0, ak * (np.where(endj, 0.0, lj) - 0.5 * l1k))
    const = (np.bincount(own, -0.5 * ak * l1k, ne) - 0.5 * (np.bincount(own, vi, ne) + np.bincount(own, vj, ne))
             + (2 + 0.5 * (a[i] + a[j])) * (np.log(2.0) - 0.5 * lt))
    # each live near cone: its position along the segment and distance off it, in t units
    live = ~endi & ~endj & (ak != 0)
    lo = own[live]
    sc = 1.0 / np.maximum(2 * tau[lo], 1e-300)
    sst = 0.5 + wk[live].real * sc
    dd = np.abs(wk[live].imag) * sc
    lptr = np.r_[0, np.cumsum(np.bincount(lo, minlength=ne))]
    r = np.r_[np.arange(ne), np.arange(ne)]
    t0, t1 = np.r_[np.zeros(ne), np.full(ne, 0.5)], np.r_[np.full(ne, 0.5), np.ones(ne)]
    R_, T0, T1 = [], [], []
    for _ in range(40):
        ow, ent = pc._ranges(lptr, r)
        gap = np.maximum(np.maximum(t0[ow] - sst[ent], sst[ent] - t1[ow]), 0.0)
        dist = np.sqrt(gap ** 2 + dd[ent] ** 2)
        dm = np.full(len(r), np.inf)
        if len(dist):
            st = np.flatnonzero(np.r_[True, ow[1:] != ow[:-1]])
            dm[ow[st]] = np.minimum.reduceat(dist, st)
        ok = dm >= sep * (t1 - t0)
        R_.append(r[ok]); T0.append(t0[ok]); T1.append(t1[ok])
        r, t0, t1 = r[~ok], t0[~ok], t1[~ok]
        if not len(r):
            break
        mid = 0.5 * (t0 + t1)
        r, t0, t1 = np.r_[r, r], np.r_[t0, mid], np.r_[mid, t1]
    if len(r):                                                      # a cone on the arc: give up
        R_.append(r); T0.append(t0); T1.append(t1)
    r, t0, t1 = (np.concatenate(v) for v in (R_, T0, T1))
    o = np.argsort(r, kind="stable")
    r, t0, t1 = r[o], t0[o], t1[o]
    left, right = t0 == 0, t1 == 1
    alpha = np.where(left, a[i[r]], np.where(right, a[j[r]], 0.0))
    x, wq = pc._gj_batch(alpha, nq)                                 # ∫ (1 − s)^α g ds on [0, 1]
    h = t1 - t0
    t = np.where(left[:, None], t0[:, None] + h[:, None] * (1 - x), t0[:, None] + h[:, None] * x)
    bh = beta[r] / 2
    w = np.tan((t - 0.5) * bh[:, None])
    dwdt = bh[:, None] / np.cos((t - 0.5) * bh[:, None]) ** 2
    dw_end = bh / np.cos(0.5 * bh) ** 2                             # dw/dt at either end
    ow, ent = pc._ranges(ptr, r)
    kk, wke = idx[ent], wk[ent]
    endk = (kk == i[r][ow]) | (kk == j[r][ow])
    sing = (left[ow] & (kk == i[r][ow])) | (right[ow] & (kk == j[r][ow]))
    # a_k Log(w − w_k), continuous along the segment: the ends are real (constant phase), the
    # others log|w_k| + Log(1 − w/w_k) (1 − w/w_k never crosses the cut); summed per panel
    # (w is real: 1 − w/w_k = 1 − w·conj(w_k)/|w_k|², in real arithmetic; flat vertices skipped)
    lgr, lgi = np.zeros(t.shape), np.zeros(t.shape)
    ne_ = np.flatnonzero(~endk & (a[kk] != 0))
    if len(ne_):
        o_ = ow[ne_]
        inv = 1.0 / wke[ne_]
        wr = w[o_]
        qr, qi = 1.0 - wr * inv.real[:, None], -wr * inv.imag[:, None]
        ak_ = a[kk[ne_]][:, None]
        st = np.flatnonzero(np.r_[True, o_[1:] != o_[:-1]])
        lgr[o_[st]] = np.add.reduceat(ak_ * (0.5 * np.log(qr * qr + qi * qi) - np.log(np.abs(inv))[:, None]), st, axis=0)
        lgi[o_[st]] = np.add.reduceat(ak_ * np.arctan2(qi, qr), st, axis=0)
    en = np.flatnonzero(endk)
    o_ = ow[en]
    with np.errstate(divide="ignore"):
        d_end = np.log(np.abs(w[o_] - wke[en, None].real))
    # the panel's singular end: |w − w_end| / (t-distance · dw/dt|end); the weight carries the rest
    d_end = np.where(sing[en, None], d_end - np.log(h[o_][:, None] * (1 - x[o_]) * dw_end[o_][:, None]), d_end)
    st = np.flatnonzero(np.r_[True, o_[1:] != o_[:-1]])
    lgr[o_[st]] += np.add.reduceat(a[kk[en], None] * d_end, st, axis=0)
    lg = lgr + 1j * lgi
    val = np.sum(np.exp(lg) * dwdt * wq, 1) * h ** (1 + alpha) * dw_end ** alpha
    I = np.abs(np.bincount(r, val.real, ne) + 1j * np.bincount(r, val.imag, ne))
    chord = np.linalg.norm(A_ - B_, axis=1)
    return np.log(I) - np.log(chord) + const, Anear


def far_log_ratio(X, E, a, Anear, B: int = FAR_BAND):
    """The far cones' share of ``log ρ_e`` that :func:`local_log_ratio` drops, to
    second order: Simpson's ``(2/3)[g(0) − (g(−τ) + g(τ))/2]`` of their
    ``g = log |Π_far (w − w_k)^{a_k}|`` along the edge's chart segment. In sphere terms
    ``g = Φ_far + (A_far/2) log(1 + |w|²)`` (``Φ`` the log-chord field
    ``Σ a_k log chord(x, X_k)``, ``A_far = −2 − A_e``). Only its smooth part is kept:
    ``Φ`` is the band-``B``, heat-tapered expansion by the sphere's Green function
    (``½ log(2 − 2t) = const − ½ Σ_{l≥1} (2l+1)/(l(l+1)) P_l(t)``); taking the
    edge's near cones back out of it changed nothing measurable. On a smooth
    surface it is coherent (a 2562-vertex ellipsoid: map error 5.3e-4 → 4e-5 rad,
    the decode then equal to the exact map's); next to a scan's slivers the direct
    second difference is meaningless (values to 3) and gains nothing."""
    from .encode import eval_real_sh, real_sh_basis

    l = np.arange(B + 1, dtype=float)
    gl = np.where(l > 0, -2 * np.pi / np.maximum(l * (l + 1), 1.0) * np.exp(-l * (l + 1) / (B * (B + 1))), 0.0)
    Yv = real_sh_basis(X, B)
    coef = np.repeat(gl, 2 * np.arange(B + 1) + 1) * (Yv.T @ a)
    i, j = E[:, 0], E[:, 1]
    Pv = Yv @ coef
    d2 = eval_real_sh(pc._normalize(X[i] + X[j]), coef, B) - 0.5 * (Pv[i] + Pv[j])
    beta = np.arctan2(np.linalg.norm(np.cross(X[i], X[j]), axis=1), np.einsum("ij,ij->i", X[i], X[j]))
    return (2.0 / 3.0) * (d2 + 0.5 * (2 + Anear) * np.log1p(np.tan(beta / 4) ** 2))


def _periods(X, es):
    """``log ρ_e`` (local + far) for the rows of ``es``."""
    lr, Anear = local_log_ratio(X, es.E, es.a_v, *near_sets(es.E, es.n))
    return lr + far_log_ratio(X, es.E, es.a_v, Anear)


# ------------------------------------------------------------------ the map
def _jacobian(es, X):
    """The chord model's Jacobian bordered by the Möbius directions ((E + 6) × 3n)."""
    return sp.vstack([es._chord_jacobian(X), sp.csr_matrix(es._gauge(X).T)]).tocsc()


def _dissection_columns(V, E, n, leaf: int = 64):
    """A fill-reducing column order of the chord Jacobian: geometric nested dissection
    of the vertices (halve along the longest axis of their 3D extent, the halves'
    separator — the first half's vertices joined to the second — last), each vertex's
    three columns (two tangent, ``u``) kept together. On an icosphere-6 head (123k
    unknowns) SuperLU factors it in 1.9 s against 3.9 s under COLAMD (fill 20M vs
    24M entries)."""
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                      shape=(n, n)).tocsr()
    order, stack = [], [(np.arange(n), 0)]
    # iterative post-order: a part is (vertices, 0) to split, or (separator, 1) to emit
    while stack:
        idx, done = stack.pop()
        if done or len(idx) <= leaf:
            order.append(idx)
            continue
        Q = V[idx]
        ax = np.argmax(Q.max(0) - Q.min(0))
        side = np.full(n, -1)
        side[idx] = (Q[:, ax] > np.median(Q[:, ax])).astype(int)
        sub = A[idx]
        r = np.repeat(idx, np.diff(sub.indptr))
        sep = np.unique(r[(side[r] == 0) & (side[sub.indices] == 1)])
        side[sep] = 2
        lo, hi = idx[side[idx] == 0], idx[side[idx] == 1]
        if not len(lo) or not len(hi):
            order.append(idx)
            continue
        stack += [(sep, 1), (hi, 0), (lo, 0)]                       # popped: lo, hi, then sep
    pv = np.concatenate(order)
    return np.stack([pv, n + pv, 2 * n + pv], 1).ravel()


class _ColumnLU:
    """``solve`` of an LU of ``A[:, cols]``, in ``A``'s unknowns."""

    def __init__(self, lu, cols):
        self.lu, self.cols = lu, cols

    def solve(self, b):
        y = self.lu.solve(b)
        x = np.empty_like(y)
        x[self.cols] = y
        return x


def chord_map(V, F, log=None):
    """Sphere positions ``(n, 3)`` of the vertices of the closed genus-0 polyhedron
    ``(V, F)`` (module docstring, steps 1–2), and an info dict (stage times, the
    rows' final rms/max)."""
    from .edge_periods import EdgeSystem
    from .uniformize import uniformize

    T = {}
    t = time.perf_counter()
    es = EdgeSystem(V, F)                       # rows: near-flat faces flipped intrinsically
    i, j = es.E.T
    # converged: a layout of an unconverged flat metric is inconsistent (tol 1e-3 started the
    # head as well but a 2562-vertex ellipsoid 1e-1 off, from which it did not recover)
    X = es.centre(uniformize(V, es.F, lengths=(es.E, es.logl)))
    T["uniformize"] = time.perf_counter() - t
    lr = np.zeros(len(i))

    def resid(X, u):
        return np.log(np.linalg.norm(X[i] - X[j], axis=1)) + 0.5 * (u[i] + u[j]) + lr - es.logl

    def scales(X):                              # u from the edges' mean
        d = es.logl - lr - np.log(np.linalg.norm(X[i] - X[j], axis=1))
        return np.bincount(es.E.ravel(), np.repeat(d, 2), es.n) / np.bincount(es.E.ravel(), minlength=es.n)

    def step(X, u, lu, krylov):
        """A Newton step: solved by the LU, or by GMRES preconditioned by it. Backtracked on
        the rows' sum of squares: a few scan rows (sliver clusters) cannot be met, and
        backtracking on the largest row halved every step (rows rms 1.5e-2 after 7 steps
        instead of 1e-3 after 2). ``X`` is not re-centred in between."""
        r = resid(X, u)
        b = np.r_[-r, np.zeros(6)]
        if krylov:
            A = _jacobian(es, X)
            y, _ = spla.gmres(A, b, M=spla.LinearOperator(A.shape, lu.solve), rtol=GMRES_TOL,
                              restart=GMRES_MAX, maxiter=1)
        else:
            y = lu.solve(b)
        s = 1.0
        for _ in range(12):
            X1, u1 = es._retract(X, u, s * y)
            r1 = resid(X1, u1)
            if r1 @ r1 < r @ r or s < 1e-3:
                break
            s *= 0.5
        if log:
            log(f"   rows rms {np.sqrt(np.mean(r1 ** 2)):.2e} max {np.abs(r1).max():.1e} (step {s:g})")
        return X1, u1

    cols = _dissection_columns(es.V, es.E, es.n)

    def factor(X):
        return _ColumnLU(spla.splu(_jacobian(es, X)[:, cols].tocsc(), permc_spec="NATURAL"), cols)

    # Two chord steps (each by its own LU), one corrected step by GMRES on the second LU (the
    # corrected rows have the same Jacobian, and X moved little since it was factored).
    t = time.perf_counter()
    X, u = step(X, scales(X), factor(X), False)
    lu2 = factor(X)
    X, u = step(X, u, lu2, False)
    T["chord"] = time.perf_counter() - t
    t = time.perf_counter()
    lr = _periods(X, es)
    T["periods"] = time.perf_counter() - t
    t = time.perf_counter()
    X, u = step(X, scales(X), lu2, True)
    T["correct"] = time.perf_counter() - t
    r = resid(X, u)
    return X, dict(times=T, rms=float(np.sqrt(np.mean(r ** 2))), max=float(np.abs(r).max()))


# ------------------------------------------------------------------ the spinor
class MobiusFaces:
    """Per face, the Möbius transformation from its stereographic chart (centred at
    the face's sphere centroid) to the plane of its 3D triangle, through the three
    corners. ``points`` maps sphere points to 3D, ``spinor`` gives λ."""

    def __init__(self, X, V, F):
        self.V, self.F = V, F
        self.Rf = pc.face_rotations(X[F].mean(1))
        z = pc.chart(self.Rf[:, None], X[F])
        p = V[F]
        self.e1 = pc._normalize(p[:, 1] - p[:, 0])
        e2 = np.cross(pc._normalize(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])), self.e1)
        d = p - p[:, :1]
        w = np.einsum("fki,fi->fk", d, self.e1) + 1j * np.einsum("fki,fi->fk", d, e2)
        # the chart's orientation is the sphere's; match the plane's to it
        oz = np.imag(np.conj(z[:, 1] - z[:, 0]) * (z[:, 2] - z[:, 0]))
        ow = np.imag(np.conj(w[:, 1] - w[:, 0]) * (w[:, 2] - w[:, 0]))
        flip = np.sign(oz) != np.sign(ow)
        w = np.where(flip[:, None], np.conj(w), w)
        self.e2 = np.where(flip[:, None], -e2, e2)

        def to_01inf(q):                        # q1, q2, q3 → 0, 1, ∞
            q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
            return np.stack([np.stack([q2 - q3, -q1 * (q2 - q3)], -1),
                             np.stack([q2 - q1, -q3 * (q2 - q1)], -1)], -2)

        Sw = to_01inf(w)
        adj = np.stack([np.stack([Sw[:, 1, 1], -Sw[:, 0, 1]], -1), np.stack([-Sw[:, 1, 0], Sw[:, 0, 0]], -1)], -2)
        self.M = adj @ to_01inf(z)

    def points(self, x, f):
        """3D points of sphere points ``x`` lying in faces ``f``."""
        M, zeta = self.M[f], pc.chart(self.Rf[f], x)
        w = (M[:, 0, 0] * zeta + M[:, 0, 1]) / (M[:, 1, 0] * zeta + M[:, 1, 1])
        return self.V[self.F[f, 0]] + w.real[:, None] * self.e1[f] + w.imag[:, None] * self.e2[f]

    def spinor(self, x, f):
        """λ (quaternion, sign arbitrary per point) at sphere points ``x`` in faces ``f``:
        ``dX = M_f'(ζ) dζ`` written as a rotation and scale (:meth:`PolyMap.spinor_at`)."""
        from ..spectral.quaternion import qconj
        from .spin import rotation_to_quaternion

        R, M = self.Rf[f], self.M[f]
        y = np.einsum("nij,nj->ni", R, x)
        zeta = pc.chart(R, x)
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        dw = det / (M[:, 1, 0] * zeta + M[:, 1, 1]) ** 2
        t1 = pc._normalize(np.cross(x, np.where(np.abs(x[:, 2:3]) < 0.9, [0, 0, 1.0], [1.0, 0, 0])))
        t2 = np.cross(x, t1)

        def dX(t):
            dy = np.einsum("nij,nj->ni", R, t)
            dz = (dy[:, 0] + 1j * dy[:, 1]) / (1 - y[:, 2]) + (y[:, 0] + 1j * y[:, 1]) * dy[:, 2] / (1 - y[:, 2]) ** 2
            zt = dw * dz
            return zt.real[:, None] * self.e1[f] + zt.imag[:, None] * self.e2[f]

        d1, d2 = dX(t1), dX(t2)
        s = np.linalg.norm(d1, axis=1)
        # a Möbius derivative is a similarity: (d1, d2, n)/s is orthonormal, (t1, t2, x) too
        B = np.stack([d1 / s[:, None], d2 / s[:, None], pc._normalize(np.cross(d1, d2))], 2)
        Rm = B @ np.stack([t1, t2, x], 1)
        return np.sqrt(s)[:, None] * qconj(rotation_to_quaternion(Rm))


def sphere_nodes(X, F, Ls: float, per: float = 1.0):
    """Nodes ``(x, w, f)`` on the sphere triangles ``X[F]``: each split into ``n²``
    sub-triangles (``n = ⌈Ls · longest side / per⌉``), a node at each normalised
    centroid with the sub-triangle's spherical area (they tile the sphere)."""
    T = X[F]
    side = np.arccos(np.clip(np.einsum("fki,fki->fk", T, T[:, [1, 2, 0]]), -1, 1)).max(1)
    nf = np.maximum(1, np.ceil(Ls * side / per)).astype(int)
    xs, ws, fs = [], [], []
    for n in np.unique(nf):
        f = np.flatnonzero(nf == n)
        # the n² sub-triangles' corners as barycentric weights (k, 3 corners, 3 weights)
        tri = []
        for a in range(n):
            for b in range(n - a):
                tri.append(((a, b), (a + 1, b), (a, b + 1)))
                if a + b < n - 1:
                    tri.append(((a + 1, b), (a + 1, b + 1), (a, b + 1)))
        Wb = np.array([[[n - p - q, p, q] for p, q in t] for t in tri], float) / n
        P = pc._normalize(np.einsum("kcv,fvi->fkci", Wb, X[F[f]]))         # (faces, k, 3, 3)
        A, B, C = P[:, :, 0], P[:, :, 1], P[:, :, 2]
        num = np.abs(np.einsum("fki,fki->fk", A, np.cross(B, C)))
        den = 1 + np.einsum("fki,fki->fk", A, B) + np.einsum("fki,fki->fk", B, C) + np.einsum("fki,fki->fk", C, A)
        # (sub-triangle major, as before: node order k, then face)
        ws.append((2 * np.arctan2(num, den)).T.ravel())
        xs.append(pc._normalize(A + B + C).transpose(1, 0, 2).reshape(-1, 3))
        fs.append(np.tile(f, len(tri)))
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(fs)


# ------------------------------------------------------------------ the code
def encode_chord(V, F, L: int, iters: int = 3, over: int = 3, lift=None, Ls: float | None = None,
                 n_check: int = 6000) -> dict:
    """Polyhedron ``(V, F)`` → closed ρ̂ at band ``L`` (module docstring). Returns the
    dict of :func:`~shapegen.geom.poly_conformal.encode_polyhedral`."""
    from ..quad.lift import frame_quaternion
    from ..spectral.quaternion import qconj, qmul
    from . import dirac_spectral as ds
    from .dirac_quotient import _project_faces, close_code
    from .dirac_quotient import rho_quotient
    from ..quad.lift import SpectralLift
    from .spin import align_signs

    V, F = np.asarray(V, float), np.asarray(F)
    X, info = chord_map(V, F)
    t = time.perf_counter()
    a_v = -pc.angle_defects(V, F) / (2 * np.pi)
    X = pc.cone_gauge(V, F, np.where(np.abs(a_v) > 1e-9 / (2 * np.pi), a_v, 0.0), X)
    Ls = ds.spinor_band(L, ds.potential_band_extra(L)) if Ls is None else Ls
    Lm = L + 0.5
    mf = MobiusFaces(X, V, F)
    xs, w, fo = sphere_nodes(X, F, Ls)
    lc = align_signs(mf.spinor(pc._normalize(X[F].mean(1)), np.arange(len(F))), F)
    th, ph = np.arccos(np.clip(xs[:, 2], -1, 1)), np.arctan2(xs[:, 1], xs[:, 0])

    lam = mf.spinor(xs, fo)
    lam *= np.where(np.einsum("ij,ij->i", lam, lc[fo]) < 0, -1.0, 1.0)[:, None]
    raw = qmul(qconj(frame_quaternion(th, ph)), lam)
    v = _project_faces(Lm, th, ph, w * (raw[:, 0] + 1j * raw[:, 1]), w * (raw[:, 2] - 1j * raw[:, 3]))
    rho = rho_quotient(v, L, Ls, Lm, iters=iters, over=over)
    n, k = len(ds.sw.half_modes(0.5, Ls)), len(v) // 2
    rho, res, gamma = close_code(rho, L, Ls, v0=np.r_[v[:k], np.zeros(n - k), v[k:], np.zeros(n - k)])
    lift = SpectralLift(Ls) if lift is None else lift
    # check points as the exact path's: halfway from the chart centre to each corner, on up to n_check / 3 faces
    fs = np.unique(np.linspace(0, len(F) - 1, max(1, min(len(F), n_check // 3))).astype(int))
    zc = (0.5 * pc.chart(mf.Rf[fs][:, None], X[F[fs]])).ravel()
    fc = np.repeat(fs, 3)
    xc = np.einsum("nji,nj->ni", mf.Rf[fc], pc._unstereo(zc))
    info["times"]["code"] = time.perf_counter() - t
    info.update(solver="fast", nodes=len(xs), sphere_weight=float(w.sum()))
    return {"coeffs": rho, "sphere": X, "gamma": gamma,
            "f_hat": lift.forward(np.r_[res.psi1, res.psi2])[0],
            "samples": (xc, np.ones(len(xc)), mf.points(xc, fc)), "map": mf, "info": info}
