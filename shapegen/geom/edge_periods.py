"""Edge periods: the polyhedral sphere map as one square system, solved by Newton.

The map (:mod:`shapegen.geom.poly_conformal`) has, chart-free, the closed-form
conformal factor ``Π_k chord(x, X_k)^{a_k}`` (``a_k = −κ_k/2π``): in a face chart
the metric is ``|K_f W(ζ)| |dζ|`` with ``W = Π (ζ − p_k)^{a_k}``. Its only unknowns
are the vertices' sphere positions ``X``, up to Möbius. A straight segment of the
flat metric is a geodesic, so its length is the modulus of its **period**
``I_e = ∫ K_f W dζ`` along any path in its homotopy class; the great-circle arc
between the end points stands in for it.

**The system.** Unknowns ``X`` (2 tangent coordinates per vertex) and a log scale
``u`` per vertex; one row per edge,

    G_e = log |I_e(X)| + (u_i + u_j)/2 − log ℓ_e.

With ``E = 3n − 6`` edges and 6 Möbius directions it is square, and it stays
consistent far from the solution (``u`` absorbs what ``X`` cannot yet explain).
At the solution every ``u_i`` is the same, ``log c``; unequal ``u`` is a spurious
root.

**Turned-over triangles.** Each corner's great-arc angle is read on the branch
nearest its flat angle / (1 + a_k) (a cone magnifies angles at its vertex), so a
corner that crossed its opposite arc reads > π, continuously; the windings (angle
sums / 2π) then stay 1 through such flips, and only folds change them. The edge
opposite a corner wider than ``THETA`` comes from the other two sides instead of its
own arc (law of cosines in the developed plane, :meth:`lengths`); the two agree to
~1e-9 while the arc is in class, so the switch is seamless, and it is made *before*
the arc passes the cone, where ``|I_e|`` is not differentiable. The choice is frozen
during finite differences (a switch inside one is 1e-9 / 1e-7 = O(1) noise).

**Newton–Krylov with a trust region.** ``J·v`` by one-sided differences of ``G``.
The step is scaled by the largest *edge strain* ``|δ(X_i − X_j)| / |X_i − X_j|`` (and
``|δu|``), not by ``|v|``: an absolute step of 1e-7 is a 2e-3 strain on a 5e-5 edge,
deep in the nonlinear range (measured: J(v₁+v₂) − Jv₁ − Jv₂ was 27 % of J(v₁+v₂)).
Differences are taken at lower precision (``NQ_J``, ``P_J``: 1e-6 relative, half
the cost), panels frozen. The right preconditioner is the chord model
``log ℓ_e ≈ log|X_i − X_j| + (u_i + u_j)/2``, bordered by the Möbius directions.
The Arnoldi basis gives the linear model on its span exactly, so a step is chosen
in it with the max edge strain ≤ Δ (Tikhonov, reweighted onto the offending edges);
Δ adapts to the ratio of actual to predicted decrease, and a step that folds is
refused. Near the solution the Newton step itself is taken (quadratic).

**Local Levenberg–Marquardt.** A scan has tiny defects (a crumpled cluster of 3D
edges 5e-4 with cones a = −0.42 and +0.44 side by side) where the square Jacobian
is near singular: the Newton step asks for strains of 20–50 there, and the
Krylov-restricted trust-region step gains ~0.1 % per iteration. Where the Newton
step's strain exceeds ``HOT``, those vertices get a Levenberg–Marquardt solve of
their own (rows: the edges touching them), with the Jacobian's columns by
differences of a *local* evaluation: the cones outside the cluster stay put, so
their far field (FMM) is computed once per accepted step and only the cluster's
cones and edges are redone. One such step took the cluster's rows from rms 2e-2 to
2e-3 at strain 0.16.

**Periods.** Each edge is integrated in one of two global charts, near cones one
by one, far cones as a Taylor series: the arc ``x(t)`` (slerp, its angle from
``atan2`` — ``arccos`` of the dot product loses half the digits on short arcs:
4e-8 relative at 5e-5) is split into panels until every near cone but the end
points' own is ≥ ``SEP`` panel lengths from the panel, ``NQ``-point Gauss on each,
Gauss–Jacobi at a cone end.

**Classes by closure.** With its three arcs in class a face *closes*: each corner's
law-of-cosines value of the opposite side (:meth:`predictions`) equals that arc, at
any ``X`` (to ~1e-9). An arc no face predicts is out of class and takes the
prediction of a face whose other two arcs are verified (:meth:`_classes`); on the
head both faces of every such arc agree. A near-degenerate face (3D angle 176–179°)
and its mirror image have the same sides, so lengths cannot tell them apart: a step
that develops a face inside out (``inverted``, read at corners with verified sides)
is refused.

**Near-flat faces.** A face with a 3D angle near 180° (scans have them: collinear
lip vertices, 176–179.7°) is ambiguous on the sphere: its great-arc triangle flips
for a curvature of the true edges of a fraction of a degree, a neighbouring cone can
end up inside a face it is not a corner of (cone 25512 in face 49806: the face's
developed angles then sum to π ± 2π a_k), and the face and its mirror image have the
same sides (a spurious root with u spread 5e-3). So the rows are not the mesh's
edges there: every edge opposite a flat corner wider than ``FLAT`` is flipped
intrinsically (:func:`_unflatten`; lengths by unfolding, the metric unchanged; 235
faces on the head), which leaves the solution — the vertices' sphere points — the
same. The step guard refuses steps that develop a face inside out or break a face's
angle sum, and the local LM carries signed-angle rows for faces still flatter than
``DEGEN``.

**Head (26321 vertices, from the DU start):** 10 iterations, 701 evaluations
(local ones included), 956 s; max |G| 1e-11, u spread 6.3e-10, no face inside out,
every developed angle sum π to 1e-11. The u spread is the floor of double precision:
an edge of chord 1.5e-5 has its log length to ~1e-11 from unit-vector coordinates
(Möbius invariance holds to 1e-11 at exactly those edges), times the conditioning
of the tiny regions. Dead ends: intrinsic Delaunay rows everywhere (a spurious root
on ``hull``); angle rows in the square system (roots with u spread 0.045, or stalls).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import poly_conformal as pc

__all__ = ["EdgeSystem"]

NQ = 8          # Gauss nodes per panel
HALF = 3.0      # global charts: the FMM square |Re w|, |Im w| ≤ HALF; beyond it one series
P_FMM = 50      # FMM / series order (ratio ≤ 1/2 at the nodes: < 1e-15)
SEP = 1.0       # every near cone not at the panel's end ≥ SEP panel lengths away
NQ_J, P_J = 4, 25               # precision of the J·v differences (1e-6 relative)
THETA = np.radians(160.0)       # a corner wider than this: its opposite edge from the other two
FLAT = np.radians(170.0)        # rows: edges opposite flat corners wider than this are flipped
DEGEN = np.radians(170.0)       # a face this flat gets signed-angle rows in the local LM
CLOSE = 1e-7                    # a face prediction this close (log) verifies an arc
HOT = 1.0                       # Newton strain above which a vertex gets a local LM solve
CHARTS = (np.eye(3), np.diag([1.0, -1.0, -1.0]))


def _slerp(A, B, t):
    """Points and t-derivatives of the great arcs ``A → B`` at parameters ``t`` (…, k)."""
    cr = np.cross(A, B)
    beta = np.arctan2(np.linalg.norm(cr, axis=1), np.einsum("ij,ij->i", A, B))
    U = np.cross(cr, A)                                             # ⟂ A in the arc's plane
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-300)
    s = beta[:, None] * t
    x = A[:, None] * np.cos(s)[..., None] + U[:, None] * np.sin(s)[..., None]
    dx = beta[:, None, None] * (-A[:, None] * np.sin(s)[..., None] + U[:, None] * np.cos(s)[..., None])
    return x, dx


def _sum_rows(owner, vals, n):
    """``Σ`` of the rows of ``vals`` (entries, k) grouped by ``owner`` (sorted) into ``n`` rows."""
    out = np.zeros((n, vals.shape[1]), vals.dtype)
    if len(owner):
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        out[owner[starts]] = np.add.reduceat(vals, starts, axis=0)
    return out


def _seg_dist(P, z0, z1):
    """Distance from points ``P`` to the segments ``z0 → z1``."""
    d = z1 - z0
    s = np.clip(np.real((P - z0) * np.conj(d)) / np.maximum(np.abs(d) ** 2, 1e-300), 0.0, 1.0)
    return np.abs(P - z0 - s * d)


def _unflatten(V, F, flat):
    """Rows for a mesh with near-flat faces: intrinsic edge flips (lengths by
    unfolding, the metric unchanged) of every edge opposite a corner wider than
    ``flat`` whose quadrilateral is convex, until none is left (or none can go).
    Returns the faces ``(m, 3)`` and their side lengths ``L[f, c]`` (opposite corner
    ``c``). A flip that would join two already joined vertices is skipped."""
    F = np.asarray(F).copy()
    m = len(F)
    L = np.stack([np.linalg.norm(V[F[:, (c + 2) % 3]] - V[F[:, (c + 1) % 3]], axis=1) for c in range(3)], 1)

    def angles(L_):
        a, b, d = L_, np.roll(L_, -1, 1), np.roll(L_, -2, 1)
        return np.arccos(np.clip((b * b + d * d - a * a) / (2 * b * d), -1, 1))

    if not (angles(L) > flat).any():
        return F, L
    # twin of each side (face*3 + corner): the side with the same end points, reversed
    n = int(F.max()) + 1
    a_, b_ = F[:, [1, 2, 0]].ravel(), F[:, [2, 0, 1]].ravel()
    fwd, rev = a_ * n + b_, b_ * n + a_
    o = np.argsort(fwd)
    twin = -np.ones(3 * m, np.int64)
    pos = np.searchsorted(fwd[o], rev)
    hit = (pos < len(o)) & (fwd[o][np.minimum(pos, len(o) - 1)] == rev)
    twin[hit] = o[pos[hit]]
    twin = twin.reshape(m, 3)
    stack = [tuple(x) for x in np.argwhere(angles(L) > flat)]
    # joined vertex pairs: the mesh's edges (sorted keys) plus / minus those the flips change
    base = np.unique(np.minimum(a_, b_) * n + np.maximum(a_, b_))
    added, removed = set(), set()

    def joined(key):
        if key in added:
            return True
        k_ = np.searchsorted(base, key)
        return key not in removed and k_ < len(base) and base[k_] == key

    def angle(f, c):
        a, b, d = L[f, c], L[f, (c + 1) % 3], L[f, (c + 2) % 3]
        return np.arccos(min(1.0, max(-1.0, (b * b + d * d - a * a) / (2 * b * d))))

    def link(f, c, t):
        twin[f, c] = t
        twin[t // 3, t % 3] = 3 * f + c

    for _ in range(100 * len(stack) + 100):
        if not stack:
            break
        f, c = (int(v) for v in stack.pop())
        f2, c2 = divmod(int(twin[f, c]), 3)
        if angle(f, c) <= flat or angle(f, c) + angle(f2, c2) <= np.pi:
            continue
        a, i, j, b = int(F[f, c]), int(F[f, (c + 1) % 3]), int(F[f, (c + 2) % 3]), int(F[f2, c2])   # f = (a, i, j), f2 = (b, j, i)
        if a == b or joined(min(a, b) * n + max(a, b)):
            continue
        l_ja, l_ai = L[f, (c + 1) % 3], L[f, (c + 2) % 3]
        l_ib, l_bj = L[f2, (c2 + 1) % 3], L[f2, (c2 + 2) % 3]
        ang_i = angle(f, (c + 1) % 3) + angle(f2, (c2 + 2) % 3)
        l_ab = np.sqrt(max(l_ai ** 2 + l_ib ** 2 - 2 * l_ai * l_ib * np.cos(ang_i), 0.0))
        t_ja, t_ai = int(twin[f, (c + 1) % 3]), int(twin[f, (c + 2) % 3])
        t_ib, t_bj = int(twin[f2, (c2 + 1) % 3]), int(twin[f2, (c2 + 2) % 3])
        F[f], L[f] = [a, i, b], [l_ib, l_ab, l_ai]
        F[f2], L[f2] = [b, j, a], [l_ja, l_ab, l_bj]
        link(f, 0, t_ib); link(f, 2, t_ai); link(f2, 0, t_ja); link(f2, 2, t_bj)
        twin[f, 1], twin[f2, 1] = 3 * f2 + 1, 3 * f + 1
        k_old, k_new = min(i, j) * n + max(i, j), min(a, b) * n + max(a, b)
        added.discard(k_old); removed.add(k_old)
        removed.discard(k_new); added.add(k_new)
        stack += [(f, cc) for cc in range(3)] + [(f2, cc) for cc in range(3)]
    return F, L


def _arnoldi(Av, b, kmax, rtol):
    """GMRES's Arnoldi process on ``Av`` from ``b``: basis ``(N, k)``, Hessenberg
    ``(k+1, k)``, ``|b|``. Stops once the least-squares residual is below ``rtol |b|``."""
    beta = np.linalg.norm(b)
    Vk, H = [b / beta], np.zeros((kmax + 1, kmax))
    k = 0
    for k in range(1, kmax + 1):
        w = Av(Vk[-1])
        for _ in range(2):
            for l_ in range(k):
                h = Vk[l_] @ w
                H[l_, k - 1] += h
                w = w - h * Vk[l_]
        H[k, k - 1] = np.linalg.norm(w)
        e1 = np.zeros(k + 1)
        e1[0] = beta
        c = np.linalg.lstsq(H[:k + 1, :k], e1, rcond=None)[0]
        if H[k, k - 1] < 1e-14 * beta or np.linalg.norm(e1 - H[:k + 1, :k] @ c) < rtol * beta:
            break
        Vk.append(w / H[k, k - 1])
    return np.array(Vk[:k]).T, H[:k + 1, :k], beta


class _Local:
    """A cluster's local evaluation: the edges of the faces around ``S`` and, per
    chart, the far field of the cones outside ``S`` about each edge's reference
    midpoint (fixed while only ``S`` moves); ``S``'s cones join every edge's block."""

    def __init__(self, es, X, S):
        n, F = es.n, es.F
        self.S = S
        inS = np.zeros(n, bool)
        inS[S] = True
        fS = inS[F].any(1)
        self.mask = np.zeros(len(es.E), bool)
        self.mask[np.stack([es.corner[k][:, fS] for k in ("eki", "ekj", "eij")]).ravel()] = True
        self.faces = self.mask[es.corner["eij"]].all(0)              # all three sides local
        self.rows = np.flatnonzero(inS[es.E].any(1))
        self.cone_in = inS[es.ci]
        self.rep = None
        self.freeze = False
        self.charts = es._far_fields(X, self.mask, self.cone_in)


class EdgeSystem:
    """The edge-period system of the closed genus-0 polyhedron ``(V, F)``."""

    def __init__(self, V, F, cone_tol: float = 1e-9):
        self.V, self.F = np.asarray(V, float), np.asarray(F)
        n = self.n = len(self.V)
        kap = pc.angle_defects(self.V, self.F)
        cone = np.abs(kap) > cone_tol
        self.a_v = np.where(cone, -kap / (2 * np.pi), 0.0)
        self.ci = np.flatnonzero(cone)
        self.conepos = np.full(n, -1)
        self.conepos[self.ci] = np.arange(len(self.ci))
        A = 0.5 * np.linalg.norm(np.cross(self.V[self.F[:, 1]] - self.V[self.F[:, 0]],
                                          self.V[self.F[:, 2]] - self.V[self.F[:, 0]]), axis=1)
        self.vA = np.bincount(self.F.ravel(), np.repeat(A / 3, 3), n)
        self.F_mesh = self.F
        self.F, Lf = _unflatten(self.V, self.F, FLAT)                  # rows: no near-flat faces
        Fr = self.F
        side = np.sort(np.stack([np.c_[Fr[:, (c + 1) % 3], Fr[:, (c + 2) % 3]] for c in range(3)], 1), 2)
        keys = side[..., 0] * n + side[..., 1]
        uk, first = np.unique(keys.ravel(), return_index=True)
        if len(uk) != 3 * n - 6:
            raise ValueError("not a closed genus-0 triangle mesh")
        self.E = np.c_[uk // n, uk % n]
        self.logl = np.log(Lf.ravel()[first])
        # per face corner c: the corner k, the other two i, j (face order k, i, j) and
        # the edges k-i, k-j, i-j, for the class fix in :meth:`lengths`
        self._uk = uk                                                   # :attr:`corner`, built when first used
        ang = np.stack([np.arccos(np.clip((Lf[:, (c + 1) % 3] ** 2 + Lf[:, (c + 2) % 3] ** 2 - Lf[:, c] ** 2)
                                          / (2 * Lf[:, (c + 1) % 3] * Lf[:, (c + 2) % 3]), -1, 1))
                        for c in range(3)], 1)
        self._expect = ang / (1 + self.a_v[Fr])                         # flat angle / (1 + a_k)
        self._flat = ang
        self._degen = np.flatnonzero(ang.max(1) > DEGEN)               # near-flat faces: two roots
        self.nevals = 0
        self.inverted = np.zeros(0, int)
        self._valid = np.zeros((3, len(self.F)), bool)
        self.frozen = None
        self._prec = (NQ, P_FMM)
        self._rep = None

    @property
    def corner(self):
        """Per face corner ``c``: the corner ``k``, the other two ``i, j`` (face order ``k, i,
        j``) and the edges ``k-i``, ``k-j``, ``i-j``, for the class fix in :meth:`lengths`."""
        if "_corner" not in self.__dict__:
            n, uk, Fr = self.n, self._uk, self.F
            eid = lambda p, q: np.searchsorted(uk, np.minimum(p, q) * n + np.maximum(p, q))   # noqa: E731
            k, i, j = Fr.T, np.roll(Fr, -1, 1).T, np.roll(Fr, -2, 1).T
            self._corner = dict(k=k, i=i, j=j, eki=eid(k, i), ekj=eid(k, j), eij=eid(i, j))
        return self._corner

    # ------------------------------------------------------------------ periods
    def _far_fields(self, X, mask, cone_in):
        """Per chart, for the edges in ``mask``: the edges, their midpoints' chart
        points, the far Taylor coefficients of the cones outside ``cone_in`` and the
        blocks (CSR) with ``cone_in``'s cones added (:class:`_Local`)."""
        from .log_fmm import far_taylor

        X = pc._normalize(X)
        A, B = X[self.E[:, 0]], X[self.E[:, 1]]
        mid = _slerp(A, B, np.full((len(A), 1), 0.5))[0][:, 0]
        a, Xc = self.a_v[self.ci], X[self.ci]
        sin = np.flatnonzero(cone_in)
        out = []
        for ch, Rg in enumerate(CHARTS):
            es_ = np.flatnonzero(mask & ((mid[:, 2] <= 0) if ch == 0 else (mid[:, 2] > 0)))
            if not len(es_):
                out.append(None)
                continue
            w = pc.chart(Rg, Xc)
            di = np.flatnonzero((np.abs(w.real) <= HALF) & (np.abs(w.imag) <= HALF) & ~cone_in)
            c = pc.chart(Rg, mid[es_])
            ext = 0.6 * np.abs(pc.chart(Rg, B[es_]) - pc.chart(Rg, A[es_])) + 1e-300
            lev = np.clip(np.floor(np.log2(2 * HALF / (2 * ext))), 0, 30).astype(int)
            T, bptr, bidx = far_taylor(w[di], a[di], c, lev, HALF, P_FMM)
            cnt = np.diff(bptr) + len(sin)
            bk = np.concatenate([np.r_[di[bidx[bptr[q]:bptr[q + 1]]], sin] for q in range(len(es_))]) \
                if len(sin) else di[bidx]
            out.append((es_, c, T, np.r_[0, np.cumsum(cnt)], bk))
        return out

    def periods(self, X, phases: bool = False, local: _Local | None = None):
        """``I_e`` of every edge along its great arc (complex; only ``|I_e|`` is chart-free).

        Each edge is integrated in one of two global stereographic charts ``w`` (the
        one whose centre is in the hemisphere of the arc's midpoint ``c_e``), where
        the length element is ``2 |e^{Φ(w)}| |dw|``,
        ``Φ = Σ_k a_k [Log(w − w_k) + log 2 − ½ log(1 + |w_k|²)]``. The cones split
        into those of the 3 × 3 block of FMM boxes around ``c_e`` (box ≥ 2 × the
        arc's extent; summed directly, as ``Log((w − w_k)/(c_e − w_k))`` so the
        branch is continuous along the arc), the rest of the square
        ``|Re w|, |Im w| ≤ HALF`` (the FMM's Taylor series about ``c_e``) and those
        beyond it (one series about the chart centre). Imaginary constants are
        dropped: they rotate ``I_e``, which nothing reads. The scale is that of the
        face charts' ``K_f W`` (twice the length element above), so ``u`` is
        :class:`~shapegen.geom.poly_conformal.PolyMap`'s ``log c``. With ``local``
        only its edges are integrated, from its cached far field (others are 0)."""
        from .log_fmm import far_taylor

        self.nevals += 1
        nq, P = self._prec
        X = pc._normalize(X)
        i, j = self.E[:, 0], self.E[:, 1]
        ne = len(i)
        A, B = X[i], X[j]
        mid = _slerp(A, B, np.full((ne, 1), 0.5))[0][:, 0]
        a = self.a_v[self.ci]
        Xc = X[self.ci]
        out = np.zeros(ne, complex)
        psi = (np.zeros(ne), np.zeros(ne))
        own_i, own_j = self.conepos[i], self.conepos[j]
        frozen = self.frozen if local is None else None                # panels of a nearby X (J·v)
        panels = []
        for ch, Rg in enumerate(CHARTS):
            if local is not None:
                if local.charts[ch] is None:
                    panels.append(None)
                    continue
                es_, c, T, bptr, bk = local.charts[ch]
                T = T[:, :P + 1].copy()
            else:
                es_ = (np.flatnonzero((mid[:, 2] <= 0) if ch == 0 else (mid[:, 2] > 0))
                       if frozen is None else frozen[ch][0])
            if not len(es_):
                panels.append((es_, (np.zeros(0, int), np.zeros(0), np.zeros(0))))
                continue
            w = pc.chart(Rg, Xc)
            dom = (np.abs(w.real) <= HALF) & (np.abs(w.imag) <= HALF)
            di, polar = np.flatnonzero(dom), np.flatnonzero(~dom if local is None else ~dom & ~local.cone_in)
            const = np.sum(a * (np.log(2.0) - 0.5 * np.log1p(np.abs(w) ** 2)))
            pcoef = np.zeros(P + 1, complex)                           # Σ_polar a [log|w_k| + Log(1 − w/w_k)]
            if len(polar):
                pcoef[0] = np.sum(a[polar] * np.log(np.abs(w[polar])))
                pw = np.ones(len(polar), complex)
                for n_ in range(1, P + 1):
                    pw = pw / w[polar]
                    pcoef[n_] = -np.sum(a[polar] * pw) / n_
            wi, wj = pc.chart(Rg, A[es_]), pc.chart(Rg, B[es_])
            if local is None:
                c = pc.chart(Rg, mid[es_])
                ext = 0.6 * np.abs(wj - wi) + 1e-300
                lev = np.clip(np.floor(np.log2(2 * HALF / (2 * ext))), 0, 30).astype(int)
                T, bptr, bidx = far_taylor(w[di], a[di], c, lev, HALF, P)
                bk = di[bidx]                                          # block cones (cone index)
            bown = np.repeat(np.arange(len(es_)), np.diff(bptr))
            T[:, 0] += const + np.bincount(bown, a[bk] * np.log(np.abs(c[bown] - w[bk])), len(es_))
            if phases:                                                 # arg of the developed tangent at each end
                for end, (wq, tq, ownq) in enumerate(((wi, 0.0, own_i[es_]), (wj, 1.0, own_j[es_]))):
                    xq, dxq = _slerp(A[es_], B[es_], np.full((len(es_), 1), tq))
                    tau = pc._dsigma(np.einsum("ij,nkj->nki", Rg, xq), np.einsum("ij,nkj->nki", Rg, dxq))[:, 0]
                    dl = wq - c
                    ph = np.zeros(len(es_), complex)
                    for n_ in range(P, -1, -1):
                        ph = ph * dl + T[:, n_]
                    pv = np.zeros(len(es_), complex)
                    for n_ in range(P, -1, -1):
                        pv = pv * wq + pcoef[n_]
                    ph = ph.imag + pv.imag
                    mine = bk == ownq[bown]
                    rat = (wq[bown] - w[bk]) / (c[bown] - w[bk])
                    ph += np.bincount(bown, np.where(mine, 0.0, a[bk] * np.angle(np.where(mine, 1.0, rat))), len(es_))
                    town = tau if end == 0 else -tau
                    ph += np.where(ownq >= 0, self.a_v[(i, j)[end][es_]] * np.angle(town / (c - wq)), 0.0)
                    psi[end][es_] = ph + np.angle(tau)
            # panels: bisect until the block cones (but the ends' own) are ≥ SEP panel lengths away
            m_ = len(es_)
            oi, oj = own_i[es_], own_j[es_]
            if frozen is not None:
                r, t0, t1 = frozen[ch][1]
            else:
                r = np.r_[np.arange(m_), np.arange(m_)]
                t0, t1 = np.r_[np.zeros(m_), np.full(m_, 0.5)], np.r_[np.full(m_, 0.5), np.ones(m_)]
                R_, T0, T1 = [], [], []
                for _ in range(40):
                    xs, _ = _slerp(A[es_[r]], B[es_[r]], np.c_[t0, t1])
                    z = pc.chart(Rg, xs)
                    ow, ent = pc._ranges(bptr, r)
                    k_ = bk[ent]
                    d = _seg_dist(w[k_], z[ow, 0], z[ow, 1])
                    d[((t0[ow] == 0.0) & (k_ == oi[r][ow])) | ((t1[ow] == 1.0) & (k_ == oj[r][ow])) | (a[k_] == 0)] = np.inf
                    cnt = bptr[r + 1] - bptr[r]
                    dm = np.full(len(r), np.inf)
                    has = cnt > 0
                    if has.any():
                        starts = np.r_[0, np.cumsum(cnt)[:-1]]
                        dm[has] = np.minimum.reduceat(d, starts[has]) if len(d) else np.inf
                    ok = dm >= SEP * np.abs(z[:, 1] - z[:, 0])
                    R_.append(r[ok]); T0.append(t0[ok]); T1.append(t1[ok])
                    r, t0, t1 = r[~ok], t0[~ok], t1[~ok]
                    if len(r) == 0:
                        break
                    mid_t = 0.5 * (t0 + t1)
                    r, t0, t1 = np.r_[r, r], np.r_[t0, mid_t], np.r_[mid_t, t1]
                if len(r):
                    R_.append(r); T0.append(t0); T1.append(t1)
                r, t0, t1 = (np.concatenate(v) for v in (R_, T0, T1))
            panels.append((es_, (r, t0, t1)))
            right, left = t1 == 1.0, t0 == 0.0
            alpha = np.where(right, self.a_v[j[es_]][r], np.where(left, self.a_v[i[es_]][r], 0.0))
            x, wq = pc._gj_batch(alpha, nq)
            x = np.where(left[:, None], 1.0 - x, x)
            h = t1 - t0
            t = t0[:, None] + h[:, None] * x
            s_loc = np.where(left[:, None], x, 1.0 - x)
            # chunks of panels with ≤ budget (block entry × node) terms
            cnt = (bptr[r + 1] - bptr[r]) * nq
            cs = np.cumsum(cnt)
            cuts = np.r_[0, np.searchsorted(cs, np.arange(4_000_000, cs[-1] if len(cs) else 0, 4_000_000)), len(r)]
            for s0, s1 in zip(cuts[:-1], cuts[1:]):
                if s1 <= s0:
                    continue
                rr = r[s0:s1]
                xs, dxs = _slerp(A[es_[rr]], B[es_[rr]], t[s0:s1])
                zz = pc.chart(Rg, xs)
                dz = pc._dsigma(np.einsum("ij,nkj->nki", Rg, xs), np.einsum("ij,nkj->nki", Rg, dxs))
                dl = zz - c[rr][:, None]
                Phi = np.zeros_like(zz)
                for n_ in range(P, -1, -1):                            # far: Taylor about c_e
                    Phi = Phi * dl + T[rr, n_, None]
                pv = np.zeros_like(zz)
                for n_ in range(P, -1, -1):                            # polar: series about 0
                    pv = pv * zz + pcoef[n_]
                Phi += pv
                ow, ent = pc._ranges(bptr, rr)
                k_ = bk[ent]
                uq = (zz[ow] - w[k_][:, None]) / (c[rr][ow] - w[k_])[:, None]
                lt = (0.5 * np.log(uq.real ** 2 + uq.imag ** 2) + 1j * np.arctan2(uq.imag, uq.real)) * a[k_][:, None]
                Phi += _sum_rows(ow, lt, len(rr))
                Phi -= alpha[s0:s1, None] * np.log(np.maximum(s_loc[s0:s1], 1e-300))
                v = 4.0 * np.sum(np.exp(Phi) * dz * wq[s0:s1], 1) * h[s0:s1]       # 2 |e^Φ dw|, in K_f units
                out[es_] += np.bincount(rr, v.real, m_) + 1j * np.bincount(rr, v.imag, m_)
        if local is None:
            self.last_panels = panels
        return (out, psi[0], psi[1]) if phases else out

    def corner_angles(self, X):
        """Signed corner angles ``(m, 3)`` of the great-arc triangles, each on the
        branch nearest its flat angle / (1 + a_k): a corner that crossed its opposite
        arc reads > π, continuously (module docstring)."""
        T = X[self.F]
        ang = np.zeros(self.F.shape)
        for c in range(3):
            a_, b_, c_ = T[:, c], T[:, (c + 1) % 3], T[:, (c + 2) % 3]
            ui = b_ - a_ * np.einsum("ij,ij->i", b_, a_)[:, None]
            uj = c_ - a_ * np.einsum("ij,ij->i", c_, a_)[:, None]
            ang[:, c] = np.arctan2(np.einsum("ij,ij->i", np.cross(ui, uj), a_), np.einsum("ij,ij->i", ui, uj))
        return ang + 2 * np.pi * np.round((self._expect - ang) / (2 * np.pi))

    def predictions(self, X, local: _Local | None = None):
        """Arc lengths ``|I_e|`` and, per face corner ``k``, the length of the opposite
        side from the two sides at ``k`` (law of cosines in the developed plane: the
        angle between their chords at ``k`` is ``(1 + a_k)`` × the spherical angle
        between their arcs, measured through the face, corrected by each chord's angle
        to its arc's developed tangent, from the periods' end phases). With all three
        arcs in class this is an identity (Cauchy, any ``X``): a face that does not
        *close* holds an arc out of class. Returns ``(|I|, P (3, m), angles)``."""
        X = pc._normalize(X)
        I, p0, p1 = self.periods(X, phases=True, local=local)
        La = np.abs(I)
        ang = self.corner_angles(X)
        cv = self.corner
        P, self._G = np.zeros(ang.T.shape), np.zeros(ang.T.shape)
        for c in range(3):
            k, eki, ekj = cv["k"][c], cv["eki"][c], cv["ekj"][c]
            dlt = [np.angle(I[e]) - np.where(self.E[e, 0] == k, p0[e], p1[e]) for e in (eki, ekj)]
            g = -(1 + self.a_v[k]) * ang[:, c] + dlt[1] - dlt[0]
            self._G[c] = g
            A, B = La[eki], La[ekj]
            P[c] = np.sqrt(np.maximum(A ** 2 + B ** 2 - 2 * A * B * np.cos(g), 0.0))
        return La, P, ang

    def _classes(self, La, P, ang, local=None):
        """Which edges take a face's prediction instead of their own arc: ``(corner,
        face)`` pairs. An arc is *verified* if some face predicts it (to ``CLOSE``);
        an unverified arc, and an arc opposite a corner wider than ``THETA`` (smooth
        through the moment the corner crosses it, where ``|I_e|`` is not
        differentiable), takes the prediction of a face whose other two arcs are
        verified (the widest corner first)."""
        cv = self.corner
        ok = np.ones(P.shape, bool) if local is None else np.broadcast_to(local.faces, P.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            match = ok & (np.abs(np.log(P / La[cv["eij"]])) < CLOSE)
        good = np.zeros(len(La), bool)
        good[cv["eij"][match]] = True
        cand = ok & good[cv["eki"]] & good[cv["ekj"]] & (~good[cv["eij"]] | (ang.T > THETA))
        c, f = np.nonzero(cand)
        o = np.argsort(-ang.T[c, f])
        c, f = c[o], f[o]
        first = np.unique(cv["eij"][c, f], return_index=True)[1]
        # faces developed inside out (all lengths agree with the mirror image: a
        # near-degenerate face has two roots); read at corners with two verified sides
        valid = ok & good[cv["eki"]] & good[cv["ekj"]]
        self._valid = valid
        # and faces whose developed angles do not sum to π: a cone that is not one of
        # their corners lies inside (off by 2π a_k)
        whole = valid.all(0)
        defect = np.abs(np.angle(np.exp(-1j * (self._G.sum(0) + np.pi))))
        self.inverted = np.flatnonzero((valid & (np.sin(self._G) > 0)).any(0) | (whole & (defect > 1e-6)))
        c, f = c[first], f[first]
        fixed = np.zeros(len(La), bool)
        fixed[cv["eij"][c, f]] = True
        self.unresolved = int(np.sum(~good & ~fixed & (np.bincount(cv["eij"][ok], minlength=len(La)) > 0)))
        return c, f

    def lengths(self, X, local: _Local | None = None):
        """``|I_e|`` in each edge's class (:meth:`predictions`, :meth:`_classes`; the
        choice is frozen with the panels)."""
        La, P, ang = self.predictions(X, local)
        if local is not None and local.freeze:
            c, f = local.rep
        elif local is None and self.frozen is not None and self._rep is not None:
            c, f = self._rep
        else:
            c, f = self._classes(La, P, ang, local)
            if local is not None:
                local.rep = (c, f)
            else:
                self._rep = (c, f)
        self.flipped = len(f)
        L = La.copy()
        L[self.corner["eij"][c, f]] = P[c, f]
        return L

    def residual(self, X, u, local: _Local | None = None):
        L = self.lengths(X, local)
        if local is not None:
            e = local.rows
            return np.log(L[e]) + 0.5 * (u[self.E[e, 0]] + u[self.E[e, 1]]) - self.logl[e]
        return np.log(L) + 0.5 * (u[self.E[:, 0]] + u[self.E[:, 1]]) - self.logl

    # ------------------------------------------------------------------ Newton
    @staticmethod
    def _basis(X):
        t1 = pc._normalize(np.cross(X, np.where(np.abs(X[:, 2:3]) < 0.9, [0, 0, 1.0], [1.0, 0, 0])))
        return t1, np.cross(X, t1)

    def _retract(self, X, u, y):
        n = self.n
        t1, t2 = self._basis(X)
        return pc._normalize(X + y[:n, None] * t1 + y[n:2 * n, None] * t2), u + y[2 * n:]

    def _gauge(self, X):
        """Orthonormal basis of the Möbius directions (rotations, boosts) in the
        tangent coordinates, padded with zeros in ``u``."""
        n = self.n
        t1, t2 = self._basis(X)
        cols = []
        for k in range(3):
            e = np.eye(3)[k]
            for v in (np.cross(e, X), e - X * X[:, k:k + 1]):
                cols.append(np.r_[np.einsum("ij,ij->i", v, t1), np.einsum("ij,ij->i", v, t2), np.zeros(n)])
        Q, _ = np.linalg.qr(np.array(cols).T)
        return Q

    def _chord_jacobian(self, X):
        """The chord model's Jacobian (E × 3n, sparse)."""
        n, (i, j) = self.n, self.E.T
        t1, t2 = self._basis(X)
        D = X[i] - X[j]
        g = D / np.einsum("ij,ij->i", D, D)[:, None]
        e = np.arange(len(i))
        rows = np.tile(e, 6)
        cols = np.r_[i, n + i, j, n + j, 2 * n + i, 2 * n + j]
        vals = np.r_[np.einsum("ij,ij->i", g, t1[i]), np.einsum("ij,ij->i", g, t2[i]),
                     -np.einsum("ij,ij->i", g, t1[j]), -np.einsum("ij,ij->i", g, t2[j]),
                     np.full(len(i), 0.5), np.full(len(i), 0.5)]
        return sp.coo_matrix((vals, (rows, cols)), shape=(len(i), 3 * n)).tocsc()

    def _strain(self, X):
        """Sparse ``S``: a step ``y`` → per edge ``δ(X_i − X_j)/|X_i − X_j|`` (3
        components, stacked) and ``δu``; :func:`_strain_max` reads it per edge."""
        n, (i, j) = self.n, self.E.T
        ne = len(i)
        t1, t2 = self._basis(X)
        ch = np.linalg.norm(X[i] - X[j], axis=1)
        rows, cols, vals = [], [], []
        for c in range(3):
            r_ = c * ne + np.arange(ne)
            for vtx, sg in ((i, 1.0), (j, -1.0)):
                rows += [r_, r_]
                cols += [vtx, n + vtx]
                vals += [sg * t1[vtx, c] / ch, sg * t2[vtx, c] / ch]
        rows.append(3 * ne + np.arange(n)); cols.append(2 * n + np.arange(n)); vals.append(np.ones(n))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(3 * ne + n, 3 * n))

    def _strains(self, s):
        """Per edge strain and per vertex ``|δu|`` from ``S y`` (or ``S Y c``)."""
        ne = len(self.E)
        return np.r_[np.sqrt(s[:ne] ** 2 + s[ne:2 * ne] ** 2 + s[2 * ne:3 * ne] ** 2), np.abs(s[3 * ne:])]

    def centre(self, X):
        """Möbius-centre ``X``: the vertex-area centroid to the origin."""
        from .conformal import ball_automorphism

        w = self.vA / self.vA.sum()
        for _ in range(100):
            c = (w[:, None] * X).sum(0)
            if np.linalg.norm(c) < 1e-14:
                break
            X = ball_automorphism(X, c / max(1.0, 2.0 * np.linalg.norm(c)))
        return X

    def _orient(self, X):
        T = X[self.F]
        return np.sign(np.einsum("ij,ij->i", np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), T.mean(1)))

    def windings(self, X):
        """Each vertex's angle sum (:meth:`corner_angles`) / 2π: 1 everywhere while the
        great-arc triangulation covers the sphere once up to triangles turned over
        (whose edge is fixed in :meth:`lengths`); a fold changes it."""
        return np.bincount(self.F.ravel(), self.corner_angles(X).ravel(), self.n) / (2 * np.pi)

    def _unfolded(self, X):
        """No fold of the great-arc triangulation (:meth:`windings`)."""
        return np.abs(self.windings(X) - 1).max() < 1e-6

    def _trust_step(self, H, beta, SY, Delta):
        """Coefficients ``c`` of the step in the Arnoldi basis minimising the model
        ``|β e₁ − H c|`` with max strain ≤ ``Delta``: Tikhonov on ``S Y c`` with
        per-edge weights raised where the strain is over (an ∞-norm bound by
        reweighting). Returns ``(c, max strain)``."""
        ne = len(self.E)
        e1 = np.zeros(H.shape[0])
        e1[0] = beta
        c = np.linalg.lstsq(H, e1, rcond=None)[0]
        m = self._strains(SY @ c)
        if m.max() <= Delta:
            return c, m.max()
        HtH, g = H.T @ H, H.T @ e1
        mu, mu0 = np.zeros(len(m)), beta ** 2 / Delta ** 2 * 1e-3
        for _ in range(40):
            over = m > Delta
            if not over.any() and m.max() > 0.7 * Delta:
                break
            mu = np.where(over, np.maximum(mu * (m / Delta) ** 2, mu0), mu * np.where(m < 0.5 * Delta, 0.5, 1.0))
            wt = np.r_[mu[:ne], mu[:ne], mu[:ne], mu[ne:]]
            Pm = SY.T @ (wt[:, None] * SY)
            c = np.linalg.solve(HtH + Pm + 1e-14 * np.trace(HtH) * np.eye(len(g)), g)
            m = self._strains(SY @ c)
        if m.max() > Delta:
            c = c * (Delta / m.max())
        return c, min(m.max(), Delta)

    def _local_lm(self, X, u, S, iters: int = 6, log=None):
        """Levenberg–Marquardt on the cluster ``S`` (positions and scales; rows: the
        edges touching it, plus a signed angle row — developed angle minus flat
        angle — at each verified corner of its near-flat faces, which lengths alone
        cannot tell from their mirror images), the Jacobian's position columns by
        differences of the local evaluation (:class:`_Local`), scaled by each
        vertex's shortest edge; steps that fold, or develop more faces inside out,
        are refused. Returns ``(X, u, iterations)``."""
        n, k = self.n, len(S)
        loc = _Local(self, X, S)

        def res(X_, u_, sel=None):
            r_ = self.residual(X_, u_, loc)
            if sel is None:
                fd = self._degen[loc.faces[self._degen]]
                c_, f_ = np.nonzero(self._valid[:, fd])
                sel = (c_, fd[f_])
            a_ = np.angle(np.exp(1j * (self._G[sel] + self._flat[sel[1], sel[0]])))
            return np.r_[r_, a_], sel

        r, sel = res(X, u)
        ninv = len(self.inverted)
        R = self.E[loc.rows]
        ucol = 0.5 * ((R[:, :, None] == S[None, None, :]).sum(1))
        mu, its = 1e-2, 0
        for its in range(1, iters + 1):
            t1, t2 = self._basis(X)
            ch = np.linalg.norm(X[R[:, 0]] - X[R[:, 1]], axis=1)
            hmin = np.full(n, np.inf)
            np.minimum.at(hmin, R[:, 0], ch)
            np.minimum.at(hmin, R[:, 1], ch)
            hS = hmin[S]
            J = np.zeros((len(r), 3 * k))
            loc.freeze = True
            for col in range(2 * k):
                v = S[col % k]
                h = 1e-6 * hS[col % k]
                X1 = X.copy()
                X1[v] = pc._normalize(X[v] + h * (t1 if col < k else t2)[v])
                J[:, col] = (res(X1, u, sel)[0] - r) / h
            loc.freeze = False
            J[:len(ucol), 2 * k:] = ucol
            D2 = np.r_[1 / hS ** 2, 1 / hS ** 2, np.ones(k)]
            Ah, g = J.T @ J, J.T @ r
            r0 = r @ r
            while mu < 1e12:
                d = -np.linalg.solve(Ah + mu * np.diag(D2), g)
                X1 = X.copy()
                X1[S] = pc._normalize(X[S] + d[:k, None] * t1[S] + d[k:2 * k, None] * t2[S])
                u1 = u.copy()
                u1[S] += d[2 * k:]
                if self._unfolded(X1):
                    r1 = res(X1, u1, sel)[0]
                    if r1 @ r1 < r0 and len(self.inverted) <= ninv:
                        X, u, r = X1, u1, r1
                        ninv = len(self.inverted)
                        mu /= 3
                        break
                mu *= 4
            else:
                break
            if log:
                log(f"   local {its}: rows rms {np.sqrt(np.mean(r ** 2)):.2e} max {np.abs(r).max():.1e} "
                    f"mu {mu:.0e} angle rows {len(sel[0])} inverted {ninv}")
            if r @ r > 0.9 * r0:
                break
            loc = _Local(self, X, S)                                    # far field about the moved edges
            r, sel = res(X, u)
        return X, u, its

    def solve(self, X, tol: float = 1e-11, max_iter: int = 30, gmres_tol: float = 1e-1,
              eps: float = 1e-6, kmax: int = 60, log=None):
        """Newton–Krylov with a strain trust region, plus local LM on the clusters
        where the Newton step is wild (module docstring), from sphere positions
        ``X``. Returns ``(X, u, info)``. Stops at
        ``max |G| < tol`` or once a step near it gains less than 2× (round-off of
        the periods, ~1e-12)."""
        n = self.n
        X = self.centre(pc._normalize(np.asarray(X, float)))
        lnL = np.log(self.lengths(X))
        lu_ = self.logl - lnL                                         # u from the edges' mean
        u = np.bincount(self.E.ravel(), np.repeat(lu_, 2), n) / np.bincount(self.E.ravel(), minlength=n)
        r = self.residual(X, u)
        info = dict(rms0=float(np.sqrt(np.mean(r ** 2))), newton=0, gmres=0, halvings=0, local=0)
        nbad = len(self.inverted)
        Delta = 0.5
        for it in range(max_iter):
            if np.abs(r).max() < tol:
                break
            info["newton"] += 1
            panels = self.last_panels                                  # the last evaluation was at X
            Q = self._gauge(X)
            lu = spla.splu(sp.vstack([self._chord_jacobian(X), sp.csr_matrix(Q.T)]).tocsc())
            Sx = self._strain(X)
            self.frozen, self._prec = panels, (NQ_J, P_J)
            r_lo = self.residual(X, u)

            def Av(z, X=X, u=u, r_lo=r_lo, Q=Q, lu=lu, Sx=Sx):
                y = lu.solve(z)
                s = self._strains(Sx @ y).max()
                if s == 0:
                    return np.zeros_like(z)
                h = eps / s
                X1, u1 = self._retract(X, u, h * y)
                return np.r_[(self.residual(X1, u1) - r_lo) / h, Q.T @ y]

            rms = np.sqrt(np.mean(r ** 2))
            Vk, H, beta = _arnoldi(Av, np.r_[-r, np.zeros(6)], kmax, max(gmres_tol * min(1.0, rms), 1e-6))
            self.frozen, self._prec = None, (NQ, P_FMM)
            info["gmres"] += H.shape[1]
            Y = lu.solve(Vk)
            SY = Sx @ Y
            e1 = np.zeros(H.shape[0])
            e1[0] = beta
            mN = self._strains(SY @ np.linalg.lstsq(H, e1, rcond=None)[0])
            got = None
            for _ in range(30):                                        # trust region on the max strain
                c, m = self._trust_step(H, beta, SY, Delta)
                pred = beta ** 2 - np.sum((e1 - H @ c) ** 2)
                X1, u1 = self._retract(X, u, Y @ c)
                X1 = self.centre(X1)
                ratio = -1.0
                if pred > 0 and self._unfolded(X1):
                    r1 = self.residual(X1, u1)
                    ratio = (r @ r - r1 @ r1) / pred if len(self.inverted) <= nbad else -2.0
                if log and ratio <= 0.1:
                    log(f"   trial: strain {m:.1e} pred {pred / beta ** 2:.2e} ratio {ratio:.2f} inverted {self.inverted[:6]}")
                if ratio > 0.1:
                    got = X1, u1, r1, m
                    nbad = len(self.inverted)
                    if ratio > 0.75 and m > 0.9 * Delta:
                        Delta *= 2
                    elif ratio < 0.25:
                        Delta *= 0.3
                    break
                Delta = 0.3 * min(Delta, m)
                info["halvings"] += 1
            prev, prev2 = np.abs(r).max(), r @ r
            if got is not None:
                X, u, r, m = got
            # clusters where the Newton step is wild: a local LM of their own
            ne = len(self.E)
            vs = mN[ne:].copy()                                        # per vertex: max strain of its edges, |δu|
            np.maximum.at(vs, self.E[:, 0], mN[:ne])
            np.maximum.at(vs, self.E[:, 1], mN[:ne])
            if vs.max() > max(Delta, HOT):
                S = np.flatnonzero(vs > HOT)
                S = np.sort(S[np.argsort(-vs[S])[:400]])
                X, u, nl = self._local_lm(X, u, S, log=log)
                info["local"] += nl
                r = self.residual(X, u)
                nbad = len(self.inverted)
            elif got is None:
                break
            self._state = (X, u)
            if log:
                log(f"it {it} rms {np.sqrt(np.mean(r ** 2)):.2e} max {np.abs(r).max():.1e} Delta {Delta:.1e} "
                    f"newton strain {mN.max():.1e} krylov {H.shape[1]} evals {self.nevals} replaced {self.flipped}")
            if np.abs(r).max() > 0.5 * prev and np.abs(r).max() < 1e3 * tol:     # at the evaluation's round-off
                break
            if got is None and r @ r > (1 - 1e-3) * prev2 and Delta < 1e-6:      # stuck (a step folds or inverts)
                break
        info.update(rms=float(np.sqrt(np.mean(r ** 2))), max=float(np.abs(r).max()),
                    u_spread=float(u.max() - u.min()), evals=self.nevals)
        return X, u, info
