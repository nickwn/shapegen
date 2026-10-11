"""The polyhedral encode, ``codec.encode(method="polyhedral")`` on every mesh: a
discrete conformal sphere map corrected by edge periods, and a piecewise-Möbius
spinor.

The exact map (:mod:`~shapegen.geom.edge_periods`) asks every edge's period
``|∫ K Π_k (w − w_k)^{a_k} dw|`` along its great arc to be its 3D length. It is
an N-body integral per edge, and on a scanned head its Newton solve takes ~20 min.
This module solves a sparse model of the same equations instead and lands
~4e-4 rad (median) from the exact map on that head in ~4 s; the code decodes as
well as the exact one.

0. **The polyhedron.** :func:`flat_reduce` joins faces across flat edges and drops
   vertices that are not cones (inside a planar region, or on a straight crease),
   and triangulates each planar region from its points alone, so the code depends
   on the polyhedron and not on its triangulation: a cube's 12-triangle, flipped
   192-triangle and flat-subdivided meshes give the same code (to the bit; the hull
   below and its flat subdivision: 1e-13). A mesh with no flat edge is used as it is.
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
   more local updates gain nothing. On coarse meshes the far cones are not far
   and the far share is off by 1e-3 (rms, in log ρ); so when the far share moves
   the map by more than ``FAR_EXACT`` (median, one solve on the LU) the far
   correction is taken from the exact periods
   (:meth:`~shapegen.geom.edge_periods.EdgeSystem.lengths`, FMM), frozen, the
   local part re-evaluated after the rows converge, and repeated until the map
   stops moving (< 1e-7): 1e-6–1e-7 rad from the exact map on 162–642-vertex
   bottles and ellipsoids (from 2e-4–9e-4). Where that does not settle, or
   settles on a spurious root (unequal ``u``: a spiky 9-vertex hull, a 12-triangle
   cube), the exact Newton–Krylov solve runs from the uniformization
   (:meth:`~shapegen.geom.edge_periods.EdgeSystem.solve`, ≲ 1 s there). The far share moves a 41k-vertex head's map by 1.7e-5 (the
   exact correction would move it 1.3e-5, at 3–6 s), a 2562-vertex bottle's by
   8e-5: both skip it.
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
   cost 25% on the ellipsoid at L=16. The sides are great arcs, not the faces'
   true (curved) edges; :func:`lune_nodes` moves them to the mean of the two
   faces' Möbius circles, to first order (the arcs cost 19–24 % on 162- and
   642-vertex smooth meshes at L=16; +0.3 s / +0.45 s on a 41k-vertex head at
   L = 16 / 32, which it leaves unchanged).

**Measured** at L=32. FaceScape head (26k vertices): vertex error 0.0684, face
region 0.0260, surface 0.0688; the exact map's encode 0.0684 / 0.0250 / 0.0688.
~3.9 s end to end through the codec, single-threaded (also with BLAS on one
thread): uniformization (with the row setup) 0.8 s, chord solve 0.95 s (two LUs
of the 79k-unknown system, 0.43 s each), periods 0.6 s, correction 0.17 s,
gauge, quadrature and code 1.0 s, scoring 0.2 s (COLAMD LUs; not re-timed since
the nested-dissection order). Smoothed icosphere-6 heads (``rho_ico6``, 40962
vertices): 7–8 s per encode at L = 16 and 32 — uniformization 1.0 s, chord solve
3.6 s (two LUs of the 123k-unknown system, 1.8 s each by :func:`_dissection_columns`;
3.7 s each under COLAMD), periods 0.7 s, code 0.7 s (L=16) / 1.4 s (L=32), check
points 0.7 s (:func:`surface_samples`). They
took ~62 s before uniformize's minimum-degree ordering was relabelled (53 s of it
in SuperLU's ordering on the icosphere numbering).

**Against the exact encoder** (a global Newton fit and a closed-form curved-patch
spinor quadrature; removed 2026-10-11 after this comparison), both scored on its true surface points, L=16: cube 0.0107 vs
0.0113, demo bottle (162 / 642 / 2562 vertices) 0.00624 / 0.00297 / 0.00145 vs
0.00626 / 0.00301 / 0.00149, ellipsoid (162 / 642 / 2562) 0.00113 / 0.000315 /
0.000213 vs 0.00111 / 0.000317 / 0.000214; 0.3–2.0 s against 0.15–7.8 s. At L=8 a box
0.0388 vs 0.0402, a 12-gon prism 0.0324 vs 0.0323, and the spiky 9-vertex hull
0.0670 vs 0.0594 (+13 %: its faces are large and its cones strong, the Möbius
spinor's worst case). The codes themselves differ by 1–37 % (the spinor
models differ); the exact encoder is tessellation-invariant to its quadrature
error, this one through step 0. (A once-measured "3–12× the exact error on coarse
polyhedra" was mostly the Möbius check points, now :func:`surface_samples`.)
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
FAR_EXACT = 1e-4        # exact periods once the far share moves the map more than this (median, rad)


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
    """``log ρ_e`` for the rows of ``es``: its local part and the far cones' share."""
    lr, Anear = local_log_ratio(X, es.E, es.a_v, *near_sets(es.E, es.n))
    return lr, far_log_ratio(X, es.E, es.a_v, Anear)


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
    X = X_du = es.centre(uniformize(V, es.F, lengths=(es.E, es.logl)))
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
    loc, far = _periods(X, es)
    lr = loc + far
    T["periods"] = time.perf_counter() - t
    t = time.perf_counter()
    X, u = step(X, scales(X), lu2, True)
    T["correct"] = time.perf_counter() - t
    # How far the far share alone moves the map (one solve on the LU): its approximation
    # error moves it 0.1–0.75× as much (measured), so only then are exact periods worth it
    y = lu2.solve(np.r_[far, np.zeros(6)])
    far_move = float(np.median(np.hypot(y[:es.n], y[es.n:2 * es.n])))
    exact = far_move > FAR_EXACT
    if exact:
        t = time.perf_counter()

        def converge(X, u, lu):
            """Steps until the rows stop improving 4× a step (a fresh LU once GMRES on the
            old one stalls): on an 8-vertex box a correction moves the map by 0.1 rad."""
            fresh = False
            for _ in range(20):
                r0 = resid(X, u)
                X1, u1 = step(X, u, lu, not fresh)
                r1 = resid(X1, u1)
                if r1 @ r1 < r0 @ r0:
                    X, u = X1, u1
                if r1 @ r1 > 0.25 * (r0 @ r0) or r1 @ r1 < 1e-24 * len(r1):
                    if fresh or r1 @ r1 < 1e-24 * len(r1):
                        break
                    lu, fresh = factor(X), True
                else:
                    fresh = False
            return X, u, lu

        lu, rounds = lu2, 0
        move = np.inf
        for rounds in range(1, 6):              # the far correction from the exact periods, frozen
            X0 = X
            D = np.log(es.lengths(X)) - np.log(np.linalg.norm(X[i] - X[j], axis=1)) - sum(_periods(X, es))
            lr = sum(_periods(X, es)) + D
            X, u, lu = converge(X, scales(X), lu)
            lr = sum(_periods(X, es)) + D        # the local part at the new X
            X, u, lu = converge(X, scales(X), lu)
            move = float(np.median(np.linalg.norm(es.centre(X) - es.centre(X0), axis=1)))
            if log:
                log(f"   exact round {rounds}: move {move:.1e}")
            if move < 1e-7:
                break
        # exact rows with u = log c everywhere: unequal u is a spurious root
        lr = np.log(es.lengths(X)) - np.log(np.linalg.norm(X[i] - X[j], axis=1))
        uu = scales(X)
        spread = float(uu.max() - uu.min())
        if log:
            log(f"   exact rounds {rounds}, u spread {spread:.1e}")
        if move >= 1e-7 or spread > 1e-6:
            # the frozen far correction does not settle (a spiky 9-vertex hull: moves of
            # 0.06–0.3 rad per round), or settles on a spurious root (the same hull,
            # numbered otherwise): the exact Newton–Krylov solve, from the uniformization
            X, u, _ = es.solve(X_du)
            T["newton"] = True
        T["exact"] = time.perf_counter() - t
    r = resid(X, u)
    return X, dict(times=T, rms=float(np.sqrt(np.mean(r ** 2))), max=float(np.abs(r).max()),
                   far_move=far_move, exact_periods=exact)


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

    def sphere_points(self, P, f):
        """Sphere points of 3D points ``P`` lying in faces ``f`` (the inverse of
        :meth:`points`)."""
        d = P - self.V[self.F[f, 0]]
        w = np.einsum("ij,ij->i", d, self.e1[f]) + 1j * np.einsum("ij,ij->i", d, self.e2[f])
        M = self.M[f]
        zeta = (M[:, 1, 1] * w - M[:, 0, 1]) / (M[:, 0, 0] - M[:, 1, 0] * w)
        return np.einsum("nji,nj->ni", self.Rf[f], pc._unstereo(zeta))

    def spinor(self, x, f):
        """λ (quaternion, sign arbitrary per point) at sphere points ``x`` in faces ``f``:
        ``dX = M_f'(ζ) dζ`` written as a rotation and scale."""
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
def lune_nodes(mf, X, F, Ls: float, nmax: int = 8):
    """Nodes ``(x, w, (f0, f1))`` that move :func:`sphere_nodes`' great-arc sides to
    the faces' curved sides, to first order: per edge the strip between the arc and
    the curve, one node per arc point at the curve's half offset with weight
    ``offset × arc length``, which ``f1`` gains and ``f0`` loses (one node carries
    both: its value is ``w (λ_f1 − λ_f0)``; offsets are positive into ``f0``). The curve is the mean of the two faces' Möbius circles through the
    edge (:class:`MobiusFaces` maps the 3D edge to a circle; the two differ when
    the faces are not discretely conformal), so it is shared and the nodes tile.
    ``n = ⌈Ls · arc⌉`` Gauss points per edge, in ``[1, nmax]``; one is the midpoint
    with the weight of a parabolic offset (most edges of a dense mesh).

    Measured with the exact map and spinor signs at L=16: the great-arc sides cost
    +24 % (bottle, 162 vertices), +22 % (ellipsoid, 162) and +19 % (642) against
    the exact curved-patch quadrature; with the lunes they are within 2 %. On a
    cube the two circles are mirror images, the mean is the arc (its true edges
    are great arcs, by symmetry), and the nodes have zero weight."""
    n = len(X)
    m = len(F)
    fe, ke = np.repeat(np.arange(m), 3), np.tile(np.arange(3), m)
    i_, j_ = F[fe, ke], F[fe, (ke + 1) % 3]
    o = np.argsort(np.minimum(i_, j_) * n + np.maximum(i_, j_), kind="stable")
    f0, f1 = fe[o][0::2], fe[o][1::2]
    k0, k1 = ke[o][0::2], ke[o][1::2]
    A, B = X[F[f0, k0]], X[F[f0, (k0 + 1) % 3]]                    # f0 runs A → B, f1 runs B → A
    om = np.arctan2(np.linalg.norm(np.cross(A, B), axis=1), np.einsum("ij,ij->i", A, B))
    nab = pc._normalize(np.cross(A, B))                            # the arc's left normal (f0's side)
    # each face's circle through A, B and the preimage of its 3D side's midpoint: plane n_c·x = h
    p = mf.V[F]
    d_ = p - p[:, :1]
    wc = np.einsum("fki,fi->fk", d_, mf.e1) + 1j * np.einsum("fki,fi->fk", d_, mf.e2)

    def circle(f, k):
        wm = 0.5 * (wc[f, k] + wc[f, (k + 1) % 3])
        M = mf.M[f]
        zm = (M[:, 1, 1] * wm - M[:, 0, 1]) / (M[:, 0, 0] - M[:, 1, 0] * wm)
        C = np.einsum("fji,fj->fi", mf.Rf[f], pc._unstereo(zm))
        nc = pc._normalize(np.cross(B - A, C - A))
        return nc, np.einsum("ij,ij->i", nc, A)

    nc0, h0 = circle(f0, k0)
    nc1, h1 = circle(f1, k1)
    nl = np.clip(np.ceil(Ls * om).astype(int), 1, nmax)
    xs, ws, fs = [], [], []
    for q in np.unique(nl):
        e = np.flatnonzero(nl == q)
        if q == 1:                     # the midpoint, weighted for a parabolic offset: ∫ d = ⅔ d(½)
            s, wg = np.array([0.5]), np.array([4.0 / 3.0])
        else:
            xg, wg = np.polynomial.legendre.leggauss(q)
            s = (xg + 1) / 2
        so = np.sin(om[e])[:, None]
        P = (np.sin((1 - s) * om[e, None])[..., None] * A[e, None] + np.sin(s * om[e, None])[..., None] * B[e, None]) / so[..., None]
        off = []
        for nc, h in ((nc0[e], h0[e]), (nc1[e], h1[e])):
            # the circle's point on the great circle through P normal to the arc: cos d (n·P) + sin d (n·nab) = h
            p1 = np.einsum("rki,ri->rk", P, nc)
            p2 = np.einsum("ri,ri->r", nab[e], nc)[:, None]
            R_ = np.hypot(p1, p2)
            phi = np.arctan2(p2, p1)
            ac = np.arccos(np.clip(h[:, None] / R_, -1, 1))
            dp, dm = phi + ac, phi - ac
            dd = np.where(np.abs(dp) < np.abs(dm), dp, dm)
            off.append((dd + np.pi) % (2 * np.pi) - np.pi)
        d = 0.5 * (off[0] + off[1])
        x = pc._normalize(np.cos(d / 2)[..., None] * P + np.sin(d / 2)[..., None] * nab[e, None])
        wl = (d * om[e, None] * (wg / 2)).ravel()
        xs.append(x.reshape(-1, 3))
        ws.append(wl)
        fs.append(np.c_[np.repeat(f0[e], q), np.repeat(f1[e], q)])
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(fs)


def _ear_clip(P2, P3):
    """A triangulation of the simple polygon ``P2`` (k, 2; counter-clockwise) that
    depends only on its points: repeatedly cut the ear with the largest smallest
    angle, ties to the apex whose 3D position ``P3`` is lexicographically first.
    Returns index triples, or None if no ear is found."""
    idx = list(range(len(P2)))
    tris = []
    key = np.lexsort(np.round(P3, 9).T[::-1])
    rank = np.empty(len(P2), int)
    rank[key] = np.arange(len(P2))
    while len(idx) > 3:
        best, bq = None, None
        k = len(idx)
        for c in range(k):
            a, b, d = idx[c - 1], idx[c], idx[(c + 1) % k]
            u, v = P2[a] - P2[b], P2[d] - P2[b]
            area = u[0] * v[1] - u[1] * v[0]
            if area >= -1e-14 * (np.hypot(*u) * np.hypot(*v)):   # reflex or straight apex
                continue
            T = P2[[a, b, d]]
            inside = False
            for o in idx:
                if o in (a, b, d):
                    continue
                w = [((T[(s + 1) % 3, 0] - T[s, 0]) * (P2[o, 1] - T[s, 1])
                      - (T[(s + 1) % 3, 1] - T[s, 1]) * (P2[o, 0] - T[s, 0])) for s in range(3)]
                if min(w) >= -1e-14 or max(w) <= 1e-14:
                    inside = True
                    break
            if inside:
                continue
            L_ = [np.hypot(*(T[(s + 1) % 3] - T[s])) for s in range(3)]
            ang = [np.arccos(np.clip((L_[s] ** 2 + L_[(s + 2) % 3] ** 2 - L_[(s + 1) % 3] ** 2)
                                     / (2 * L_[s] * L_[(s + 2) % 3]), -1, 1)) for s in range(3)]
            q = (round(min(ang), 9), -rank[b])
            if bq is None or q > bq:
                best, bq = c, q
        if best is None:
            return None
        tris.append((idx[best - 1], idx[best], idx[(best + 1) % k]))
        idx.pop(best)
    tris.append(tuple(idx))
    return tris


def flat_reduce(V, F, tol: float = 1e-9):
    """The polyhedron behind the triangulation ``(V, F)``: faces joined across flat
    edges (dihedral angle < ``tol``) into planar regions, vertices that are not
    cones (angle defect < ``tol``) dropped where they are inside a region or on a
    straight crease between two, and every region with more than one triangle (or a
    dropped corner) triangulated anew by :func:`_ear_clip`, from its points alone.
    So a cube's triangulations (12 triangles, flipped grids, flat subdivisions) all
    reduce to the same 12 triangles. Regions whose boundary is not one simple loop
    keep their triangles and vertices.

    Returns ``None`` when nothing is flat (a scan, a smooth mesh), else
    ``(Vr, Fr, new, (dropped, face))``: vertices in a canonical (lexicographic)
    order, ``new[v]`` the index of input vertex ``v`` in ``Vr`` (−1 if dropped), and
    for each dropped vertex the face of ``Fr`` it lies on."""
    import scipy.sparse.csgraph as csg

    V, F = np.asarray(V, float), np.asarray(F)
    n, m = len(V), len(F)
    Nf = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
    fe, ke = np.repeat(np.arange(m), 3), np.tile(np.arange(3), m)
    a_, b_ = F[fe, ke], F[fe, (ke + 1) % 3]
    fwd = a_ * n + b_
    o = np.argsort(fwd)
    tw = o[np.searchsorted(fwd[o], b_ * n + a_)]                    # twin half-edge
    ft = fe[tw]
    flat = (np.einsum("ij,ij->i", Nf[fe], Nf[ft]) > 0) & (np.linalg.norm(np.cross(Nf[fe], Nf[ft]), axis=1) < tol)
    if not flat.any():
        return None
    G = sp.coo_matrix((np.ones(flat.sum()), (fe[flat], ft[flat])), shape=(m, m))
    _, reg = csg.connected_components(G, directed=False)
    # corner angles, angle sum per (vertex, region)
    ang = np.zeros((m, 3))
    for c in range(3):
        u_, w_ = V[F[:, (c + 1) % 3]] - V[F[:, c]], V[F[:, (c + 2) % 3]] - V[F[:, c]]
        ang[:, c] = np.arctan2(np.linalg.norm(np.cross(u_, w_), axis=1), np.einsum("ij,ij->i", u_, w_))
    defect = 2 * np.pi - np.bincount(F.ravel(), ang.ravel(), n)
    R = reg.max() + 1
    vr = F.ravel() * R + np.repeat(reg, 3)
    uvr, inv = np.unique(vr, return_inverse=True)
    sums = np.bincount(inv, ang.ravel())
    nreg = np.bincount(uvr // R, minlength=n)
    straight = np.ones(n, bool)
    np.logical_and.at(straight, uvr // R, np.abs(sums - np.pi) < tol)
    drop = (np.abs(defect) < tol) & ((nreg == 1) | ((nreg == 2) & straight))
    bnd = reg[fe] != reg[ft]                                       # region boundary half-edges
    hb = np.flatnonzero(bnd)
    hb = hb[np.argsort(reg[fe[hb]], kind="stable")]
    hptr = np.r_[0, np.cumsum(np.bincount(reg[fe[hb]], minlength=R))]
    size = np.bincount(reg, minlength=R)
    fo_ = np.argsort(reg, kind="stable")
    fptr = np.r_[0, np.cumsum(size)]
    frozen = np.zeros(R, bool)
    for _ in range(10):                                            # until the kept regions are consistent
        touched = np.zeros(R, bool)
        touched[reg[np.any(drop[F], 1)]] = True
        newF, gone = [], []
        for r in np.flatnonzero(((size > 1) | touched) & ~frozen):
            h = hb[hptr[r]:hptr[r + 1]]
            nxt = dict(zip(a_[h].tolist(), b_[h].tolist()))
            loop = []
            if len(nxt) == len(h) and len(h):
                v0 = v = int(a_[h[0]])
                while True:
                    loop.append(v)
                    v = nxt[v]
                    if v == v0 or len(loop) > len(h):
                        break
            tris = None
            if len(loop) == len(h) and len(h):
                fr = fo_[fptr[r]:fptr[r + 1]]
                poly = np.array([v for v in loop if not drop[v]])
                e1 = V[poly[1]] - V[poly[0]]
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(Nf[fr[0]], e1)
                tris = _ear_clip(np.c_[(V[poly] - V[poly[0]]) @ e1, (V[poly] - V[poly[0]]) @ e2], V[poly])
            if tris is None:                   # not one simple loop: the region stays as it is
                frozen[r] = True
                continue
            gone.append(fr)
            newF.append((fr.min(), [poly[list(t)] for t in tris]))
        drop_ = drop.copy()
        drop_[F[frozen[reg]].ravel()] = False                      # frozen regions keep their vertices
        if np.array_equal(drop_, drop):
            break
        drop = drop_
    keep_face = np.ones(m, bool)
    if gone:
        keep_face[np.concatenate(gone)] = False
    if not newF:
        return None
    keepv = np.flatnonzero(~drop)
    order = keepv[np.lexsort(np.round(V[keepv], 9).T[::-1])]
    new = np.full(n, -1)
    new[order] = np.arange(len(order))
    # the input's face order, a region's new triangles where its first face was (a dense
    # mesh with a few flat pairs keeps its order, and so its check faces)
    pieces = [(f, new[F[f]][None]) for f in np.flatnonzero(keep_face)]
    for f, tris in newF:
        Fn = new[np.array(tris, int)]
        Fn = Fn[np.arange(len(Fn))[:, None], (np.argmin(Fn, 1)[:, None] + np.arange(3)) % 3]
        pieces.append((f, Fn[np.lexsort(Fn.T[::-1])]))
    pieces.sort(key=lambda p_: p_[0])
    Fr = np.concatenate([p_[1] for p_ in pieces])
    isnew = np.concatenate([np.full(len(p_[1]), not keep_face[p_[0]]) for p_ in pieces])
    Vr = V[order]
    # each dropped vertex: a face of Fr in its region containing it
    dv = np.flatnonzero(drop)
    dface = np.zeros(len(dv), int)
    if len(dv):                                # (they lie in new triangles)
        cand = np.flatnonzero(isnew)
        T = Vr[Fr[cand]]
        Nr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        A2 = np.linalg.norm(Nr, axis=1)
        Nr /= A2[:, None]
        for q, v in enumerate(dv):
            bary = np.stack([np.einsum("ij,ij->i", np.cross(T[:, (c + 1) % 3] - V[v], T[:, (c + 2) % 3] - V[v]), Nr)
                             for c in range(3)], 1) / A2[:, None]
            off = np.abs(np.einsum("ij,ij->i", V[v] - T[:, 0], Nr))
            score = np.where(off < 1e-9 * (1 + np.abs(V).max()), bary.min(1), -np.inf)
            dface[q] = cand[int(np.argmax(score))]
    return Vr, Fr, new, (dv, dface)


def encode_chord(V, F, L: int, iters: int = 3, over: int = 3, lift=None, Ls: float | None = None,
                 n_check: int = 6000) -> dict:
    """Polyhedron ``(V, F)`` → closed ρ̂ at band ``L`` (module docstring). Returns
    ``{"coeffs", "sphere"`` (per input vertex), ``"gamma", "f_hat", "samples":
    (sphere points, weights, true surface points), "map", "info"}``."""
    from ..quad.lift import frame_quaternion
    from ..spectral.quaternion import qconj, qmul
    from . import dirac_spectral as ds
    from .dirac_quotient import _project_faces, close_code
    from .dirac_quotient import rho_quotient
    from ..quad.lift import SpectralLift
    from .spin import align_signs

    V, F = np.asarray(V, float), np.asarray(F)
    V0, red = V, flat_reduce(V, F)
    if red is not None:
        V, F, new, (dv, dface) = red
    X, info = chord_map(V, F)
    t = time.perf_counter()
    a_v = -pc.angle_defects(V, F) / (2 * np.pi)
    X = pc.cone_gauge(V, F, np.where(np.abs(a_v) > 1e-9 / (2 * np.pi), a_v, 0.0), X)
    Ls = ds.spinor_band(L, ds.potential_band_extra(L)) if Ls is None else Ls
    Lm = L + 0.5
    mf = MobiusFaces(X, V, F)
    xs, w, fo = sphere_nodes(X, F, Ls)
    xl, wl, fl = lune_nodes(mf, X, F, Ls)
    lc = align_signs(mf.spinor(pc._normalize(X[F].mean(1)), np.arange(len(F))), F)

    def spinor(x, f):
        lam = mf.spinor(x, f)
        return lam * np.where(np.einsum("ij,ij->i", lam, lc[f]) < 0, -1.0, 1.0)[:, None]

    wlam = np.r_[w[:, None] * spinor(xs, fo), wl[:, None] * (spinor(xl, fl[:, 1]) - spinor(xl, fl[:, 0]))]
    xs = np.r_[xs, xl]
    th, ph = np.arccos(np.clip(xs[:, 2], -1, 1)), np.arctan2(xs[:, 1], xs[:, 0])
    raw = qmul(qconj(frame_quaternion(th, ph)), wlam)
    v = _project_faces(Lm, th, ph, raw[:, 0] + 1j * raw[:, 1], raw[:, 2] - 1j * raw[:, 3])
    rho = rho_quotient(v, L, Ls, Lm, iters=iters, over=over)
    n, k = len(ds.sw.half_modes(0.5, Ls)), len(v) // 2
    rho, res, gamma = close_code(rho, L, Ls, v0=np.r_[v[:k], np.zeros(n - k), v[k:], np.zeros(n - k)])
    lift = SpectralLift(Ls) if lift is None else lift
    info["times"]["code"] = time.perf_counter() - t
    t = time.perf_counter()
    fs = np.unique(np.linspace(0, len(F) - 1, max(1, min(len(F), n_check // 3))).astype(int))
    xc, Xc, nbad = surface_samples(V, F, X, fs)
    info["times"]["samples"] = time.perf_counter() - t
    info.update(nodes=len(xs), sphere_weight=float(w.sum()), sample_faces_dropped=nbad,
                reduced=None if red is None else (len(V), len(F)))
    if red is not None:                         # the input's vertices: kept ones, dropped ones by their face's map
        Xin = np.zeros((len(V0), 3))
        Xin[new >= 0] = X[new[new >= 0]]
        if len(dv):
            Xin[dv] = mf.sphere_points(V0[dv], dface)
        X = Xin
    return {"coeffs": rho, "sphere": X, "gamma": gamma,
            "f_hat": lift.forward(np.r_[res.psi1, res.psi2])[0],
            "samples": (xc, np.ones(len(xc)), Xc), "map": mf, "info": info}


def surface_samples(V, F, X, fs, tol: float = 0.05):
    """Check points of faces ``fs`` and their true surface points under the sphere
    map ``X``: halfway from each face's chart centre to its corners, mapped by the
    polyhedral metric's developing map (rays from the centre,
    :class:`~shapegen.geom.poly_conformal.PolyMap`), not by the face's Möbius map,
    whose correspondence is off at strong cones (a cube scored 0.060 against it,
    0.0107 against the true points). Faces that do not develop to their 3D shape
    (log side / angle error > ``tol``) or whose centre develops outside them
    (slivers, which the exact path re-centres at ~4 s on a head) are dropped.
    Returns ``(x, X3, dropped)``."""
    pm = pc.PolyMap(V, F, S0=X)
    S = pc._normalize(X)
    Rf, cones, D = pm._develop_at(S, fs, S[F[fs]].mean(1))
    err, inner = pm._shape_err(D, fs)
    ok = np.flatnonzero((err < tol) & (inner > 0.0))
    fs, Rf, D = fs[ok], Rf[ok], D[ok]
    cones = cones.take(ok)
    e1, e2 = D[:, 1] - D[:, 0], D[:, 2] - D[:, 0]
    Mx = np.stack([np.stack([e1.real, e1.imag], -1), np.stack([e2.real, e2.imag], -1)], 1)
    AB = np.linalg.solve(Mx, np.stack([pm.E1[fs], pm.E2[fs]], 1))
    zc = (0.5 * pc.chart(Rf[:, None], S[F[fs]])).ravel()
    r = np.repeat(np.arange(len(fs)), 3)
    xc = np.einsum("nji,nj->ni", Rf[r], pc._unstereo(zc))
    Dz = pc._rays(zc, np.zeros(len(r)), cones.take(r), np.full(len(r), -1)) - D[r, 0]
    Xc = V[F[fs[r], 0]] + Dz.real[:, None] * AB[r, 0] + Dz.imag[:, None] * AB[r, 1]
    return xc, Xc, int(len(err) - len(ok))
