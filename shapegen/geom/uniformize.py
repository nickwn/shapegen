"""Discrete uniformization of a genus-0 triangle mesh to the sphere.

The start of the exact polyhedral map (:mod:`shapegen.geom.poly_conformal`): the
discrete conformal map of Springborn (*Ideal hyperbolic polyhedra and discrete
uniformization*, 2020), computed as in Gillespie, Springborn & Crane (*Discrete
conformal equivalence of polyhedral surfaces*, 2021). A Newton solve over log
scale factors ``u`` on an intrinsic triangulation kept Delaunay by edge flips;
the flips update the Penner coordinates by Ptolemy's relation, so they never
change the metric and keep the energy convex. Flips exist only inside this
module: the vertices, and so the output, are the input's. On a 26k-vertex head
it takes ~0.7 s (8 Newton steps, 15k flips, each round of flips rechecking only
the edges it touched; after the first factorisation CG preconditioned by the
last one, refactored when it stalls) and starts the exact solve at a median
face-angle error of 0.13° (a Tutte embedding: 18°).

**Vertex at infinity.** One vertex ``N`` goes to ∞ (the sphere's north pole). In
the upper half-space picture its horosphere is a horizontal plane, so every link
vertex ``j`` of ``N`` gets the Dirichlet value ``u_j = −2 log ℓ_Nj``: then
``ℓ_jk e^{(u_j+u_k)/2} = ℓ_jk / (ℓ_Nj ℓ_Nk)``, the inversion of the link about
``N``. Every other vertex is flat (``Θ_i = 2π``). The link is kept as it is; a
face whose λ-lengths break the triangle inequality takes the angles (π, 0, 0),
the convex C¹ extension of the energy (Bobenko, Pinkall & Springborn 2015). For
an inscribed polyhedron the result is its own vertices up to Möbius (to 1e-11);
where the true convex link differs from N's star a few faces end degenerate and
the start can fold a face, which the exact solve's first phase undoes.

**Decorated Delaunay.** An interior edge with opposite corners in faces of
λ-lengths ``(a₁, b₁, c)``, ``(a₂, b₂, c)`` is Delaunay iff
``(a₁²+b₁²−c²)/(a₁b₁) + (a₂²+b₂²−c²)/(a₂b₂) ≥ 0`` (``cos θ₁ + cos θ₂ ≥ 0`` when
the faces are Euclidean); it needs no triangle inequality.

The Newton step solves the cotan Laplacian; the line search finds a root of the
convex energy's directional derivative, so no energy value is needed. The flat
disk is laid out face by face, sent to the sphere by inverse stereographic
projection and Möbius-centred (area centroid → 0).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla

__all__ = ["uniformize"]


def _unstereo(A, B):
    """Unit sphere points of the planar points ``A / B`` (homogeneous; north = ∞)."""
    a2, b2 = np.abs(A) ** 2, np.abs(B) ** 2
    s = a2 + b2
    w = A * np.conj(B)
    return np.stack([2 * w.real / s, 2 * w.imag / s, (a2 - b2) / s], -1)


def _mobius_centre(A0, B0, weight):
    """Sphere points of the planar points ``A0 / B0`` after the Möbius map that puts
    their ``weight``-ed centroid at the origin, accumulated as one 2 × 2 matrix."""
    M = np.eye(2, dtype=complex)
    vw = weight / weight.sum()
    for _ in range(200):
        S = _unstereo(M[0, 0] * A0 + M[0, 1] * B0, M[1, 0] * A0 + M[1, 1] * B0)
        cv = vw @ S
        t_ = np.linalg.norm(cv)
        if t_ < 1e-12:
            break
        nx, ny, nz = cv / t_
        # rotation taking the centroid's direction (planar point A/B) to north: [[Ā, B̄], [−B, A]]
        A, B = (nx + 1j * ny, 1.0 - nz) if nz < 0 else (1.0 + nz, nx - 1j * ny)
        R = np.array([[np.conj(A), np.conj(B)], [-B, A]])
        lam = (1 - min(t_, 0.5)) / (1 + min(t_, 0.5))              # pull points away from north
        D = np.array([[np.sqrt(lam), 0], [0, 1 / np.sqrt(lam)]])
        M = np.linalg.solve(R, D @ R) @ M
        M /= np.sqrt(np.linalg.det(M))
    return _unstereo(M[0, 0] * A0 + M[0, 1] * B0, M[1, 0] * A0 + M[1, 1] * B0)


class _Tri:
    """An intrinsic triangulation: faces ``T`` (m, 3), the edge opposite each corner
    ``E`` (m, 3), and per edge the log Penner coordinate ``lt`` and its two
    (face, corner) sides ``ef`` (face −1 on the boundary)."""

    def __init__(self, T, lt_of):
        self.T = np.asarray(T).copy()
        m = len(T)
        a, b = np.roll(self.T, -1, 1), np.roll(self.T, -2, 1)          # side opposite each corner
        key = np.minimum(a, b) * (self.T.max() + 1) + np.maximum(a, b)
        uk, inv = np.unique(key.ravel(), return_inverse=True)
        self.E = inv.reshape(m, 3)
        nv = self.T.max() + 1
        self.lt = lt_of(uk // nv, uk % nv)
        self.ne = len(uk)
        self._sides()

    def _sides(self):
        e = self.E.ravel()
        o = np.argsort(e, kind="stable")
        first = np.r_[True, e[o][1:] != e[o][:-1]]
        self.ef = -np.ones((self.ne, 2), np.int64)                   # flat (face*3 + corner)
        self.ef[e[o][first], 0] = o[first]
        self.ef[e[o][~first], 1] = o[~first]

    def loglen(self, u):
        """Scaled log length of each face side ``(m, 3)``."""
        return self.lt[self.E] + 0.5 * (u[self.T[:, [1, 2, 0]]] + u[self.T[:, [2, 0, 1]]])

    def _cos_sum(self, sides, u):
        """``(a²+b²−c²)/(ab)`` at flat sides ``face*3 + corner`` (``c`` the side opposite the corner)."""
        f, c = sides // 3, sides % 3
        T, E = self.T[f], self.E[f]
        k = np.arange(len(f))
        ll = self.lt[E] + 0.5 * (u[T[k[:, None], (np.arange(3) + 1) % 3]] + u[T[k[:, None], (np.arange(3) + 2) % 3]])
        c_, a_, b_ = ll[k, c], ll[k, (c + 1) % 3], ll[k, (c + 2) % 3]
        x = np.minimum(np.stack([a_ - b_, b_ - a_, 2 * c_ - a_ - b_]), 700.0)
        return np.exp(x[0]) + np.exp(x[1]) - np.exp(x[2])

    def delaunay(self, u, tol=1e-12, max_rounds=10_000):
        """Flip interior edges until the decorated Delaunay condition holds, in
        independent sets (no two share a face); returns the number of flips. After
        the first round only edges whose faces changed (or that were left bad) are
        checked: an edge's condition depends on its two faces alone."""
        nflip = 0
        act = np.flatnonzero(self.ef[:, 1] >= 0)
        for rnd in range(max_rounds):
            if rnd == 0:                                               # every face at once
                c_ = self.loglen(u)
                l_ = np.exp(c_ - c_.max(1, keepdims=True))             # one exp per side, face-scaled
                a_, b_ = l_[:, [1, 2, 0]], l_[:, [2, 0, 1]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = ((a_ * a_ + b_ * b_ - l_ * l_) / (a_ * b_)).ravel()
                s = term[self.ef[act, 0]] + term[self.ef[act, 1]]
            else:
                s = self._cos_sum(self.ef[act, 0], u) + self._cos_sum(self.ef[act, 1], u)
            bad = s < -tol
            if not bad.any():
                return nflip
            cand, pri = act[bad], -s[bad]
            f1, f2 = self.ef[cand, 0] // 3, self.ef[cand, 1] // 3
            best = np.full(len(self.T), -np.inf)
            np.maximum.at(best, f1, pri)
            np.maximum.at(best, f2, pri)
            pick = (best[f1] == pri) & (best[f2] == pri)
            used = np.zeros(len(self.T), bool)                         # ties: keep one per face
            keep = []
            for i in np.flatnonzero(pick):
                if not (used[f1[i]] or used[f2[i]]):
                    used[f1[i]] = used[f2[i]] = True
                    keep.append(i)
            keep = np.asarray(keep, int)
            self._flip(cand[keep], u)
            nflip += len(keep)
            left = np.ones(len(cand), bool)
            left[keep] = False
            act = np.unique(np.r_[cand[left], self.E[f1[keep]].ravel(), self.E[f2[keep]].ravel()])
            act = act[self.ef[act, 1] >= 0]
        raise RuntimeError("Delaunay flips did not terminate")

    def _flip(self, es, u):
        s1, s2 = self.ef[es, 0], self.ef[es, 1]
        f1, c1, f2, c2 = s1 // 3, s1 % 3, s2 // 3, s2 % 3
        T, E = self.T, self.E
        a, i, j = T[f1, c1], T[f1, (c1 + 1) % 3], T[f1, (c1 + 2) % 3]
        b = T[f2, c2]
        e_ib, e_bj = E[f2, (c2 + 1) % 3], E[f2, (c2 + 2) % 3]          # f2 = (b, j, i)
        e_ai, e_ja = E[f1, (c1 + 2) % 3], E[f1, (c1 + 1) % 3]
        lt = self.lt
        # Ptolemy: ℓ̃_ab ℓ̃_ij = ℓ̃_ai ℓ̃_bj + ℓ̃_aj ℓ̃_bi
        self.lt[es] = np.logaddexp(lt[e_ai] + lt[e_bj], lt[e_ja] + lt[e_ib]) - lt[es]
        T[f1] = np.c_[a, i, b]
        E[f1] = np.c_[e_ib, es, e_ai]
        T[f2] = np.c_[b, j, a]
        E[f2] = np.c_[e_ja, es, e_bj]
        # the sides that moved (no face is in two flips, so no side is written twice)
        self.ef[es] = np.c_[3 * f1 + 1, 3 * f2 + 1]
        for e_, old, new in ((e_ib, 3 * f2 + (c2 + 1) % 3, 3 * f1), (e_ai, 3 * f1 + (c1 + 2) % 3, 3 * f1 + 2),
                             (e_ja, 3 * f1 + (c1 + 1) % 3, 3 * f2), (e_bj, 3 * f2 + (c2 + 2) % 3, 3 * f2 + 2)):
            col = (self.ef[e_, 1] == old).astype(int)
            self.ef[e_, col] = new

    def angles(self, u):
        """Corner angles ``(m, 3)`` from the scaled lengths (half-angle formula)."""
        L = self.loglen(u)
        L = L - L.max(1, keepdims=True)
        l = np.exp(L)
        a, b, c = l, np.roll(l, -1, 1), np.roll(l, -2, 1)              # a opposite the corner
        s = 0.5 * (a + b + c)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.sqrt(np.maximum((s - b) * (s - c), 0) / np.maximum(s * (s - a), 1e-300))
        return 2 * np.arctan(t)


CG_MAX = 10        # Newton steps after the first: CG preconditioned by the last factorisation
CG_TOL = 1e-3


def _spd_factor(H):
    """LU of the sparse symmetric (positive definite) ``H``: a symmetric fill-reducing
    order and diagonal pivots, ~15 % faster than :func:`~scipy.sparse.linalg.splu`'s default."""
    return spla.splu(H, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))


def _cot_laplacian(tri, ang, n, free=None):
    """``−∂Θ/∂u``: ½ Σ cot of the opposite angles on each edge, as a Laplacian (each
    face adds ``½ cot`` of its angle opposite the edge; with ``¼`` every Newton step
    was exactly twice too long, ``φ'(1) = −φ'(0)``, and the line search spent three
    gradients a step finding ``t = ½``). A degenerate face (angles π, 0, 0: the
    convex extension of the energy) is linear in ``u`` and adds nothing. ``free`` (sorted vertex indices): the block on those
    rows and columns, built directly (CSC)."""
    a, b = tri.T[:, [1, 2, 0]].ravel(), tri.T[:, [2, 0, 1]].ravel()
    flat = (ang.min(1) < 1e-12)[:, None]
    with np.errstate(divide="ignore"):
        w = np.where(flat, 0.0, 0.5 / np.tan(ang)).ravel()
    if free is None:
        W = sp.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
        return sp.diags(np.asarray(W.sum(1)).ravel()) - W
    pos = np.full(n, -1)
    pos[free] = np.arange(len(free))
    pa, pb = pos[a], pos[b]
    diag = np.bincount(a, w, n) + np.bincount(b, w, n)
    off = (pa >= 0) & (pb >= 0)
    m = len(free)
    return sp.csc_matrix((np.r_[diag[free], -w[off], -w[off]],
                          (np.r_[np.arange(m), pa[off], pb[off]], np.r_[np.arange(m), pb[off], pa[off]])),
                         shape=(m, m))


def _layout(tri, u, n):
    """Planar positions of the flat triangulation, laid out face by face over a
    breadth-first tree of the dual graph, one tree level at a time: a face's new
    corner is placed off the side it shares with its parent (placed a level
    before), and a corner reached by several faces of one level takes the first."""
    m = len(tri.T)
    l = np.exp(tri.loglen(u))
    ang = tri.angles(u)
    s1, s2 = tri.ef[:, 0], tri.ef[:, 1]
    inner = s2 >= 0
    G = sp.coo_matrix((np.ones(inner.sum() * 2), (np.r_[s1[inner] // 3, s2[inner] // 3],
                                                   np.r_[s2[inner] // 3, s1[inner] // 3])), shape=(m, m)).tocsr()
    order, pred = csg.breadth_first_order(G, 0, directed=False, return_predecessors=True)
    depth = csg.shortest_path(G, unweighted=True, indices=0).astype(int)
    rank = np.empty(m, int)
    rank[order] = np.arange(m)
    z = np.full(n, np.nan + 0j)
    f0 = tri.T[0]
    z[f0[0]] = 0.0
    z[f0[1]] = l[0, 2]                                                 # side opposite corner 2 = (0, 1)
    z[f0[2]] = l[0, 1] * np.exp(1j * ang[0, 0])
    # each face's corner off its parent's side
    f = order[1:]
    inpar = (tri.T[f][:, :, None] == tri.T[pred[f]][:, None, :]).any(2)
    c = np.argmin(inpar, 1)
    lev = depth[f]
    srt = np.lexsort((rank[f], lev))
    f, c, lev = f[srt], c[srt], lev[srt]
    bounds = np.flatnonzero(np.r_[True, lev[1:] != lev[:-1], True])
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        ff, cc = f[a_:b_], c[a_:b_]
        T = tri.T[ff]
        k = np.arange(len(ff))
        vc, vi, vj = T[k, cc], T[k, (cc + 1) % 3], T[k, (cc + 2) % 3]
        new = np.isnan(z[vc])
        if not new.any():
            continue
        ff, cc, vc, vi, vj = ff[new], cc[new], vc[new], vi[new], vj[new]
        _, first = np.unique(vc, return_index=True)                     # rows are in BFS order
        ff, cc, vc, vi, vj = ff[first], cc[first], vc[first], vi[first], vj[first]
        # ccw (c, i, j): from i, turn by the angle at i toward c
        d = z[vj] - z[vi]
        z[vc] = z[vi] + d / np.abs(d) * l[ff, (cc + 2) % 3] * np.exp(1j * ang[ff, (cc + 1) % 3])
    return z


def uniformize(V: np.ndarray, F: np.ndarray, tol: float = 1e-10, max_iter: int = 100,
               info: dict | None = None, lengths=None) -> np.ndarray:
    """Unit-sphere positions ``(n, 3)`` of the vertices of the closed genus-0 mesh
    ``(V, F)``: its discrete uniformization (module docstring). ``tol`` bounds the
    angle-sum error ``max |Θ_i − 2π|``. ``lengths = (E, log ℓ)`` replaces the 3D edge
    lengths (every edge of ``F`` listed once in ``E``, any order): the uniformization
    of that metric on ``F``."""
    V, F = np.asarray(V, float), np.asarray(F)
    n = len(V)
    if lengths is None:
        def lt_of(i, j):
            return np.log(np.linalg.norm(V[i] - V[j], axis=1))
    else:
        E_, logl = np.sort(np.asarray(lengths[0]), 1), np.asarray(lengths[1], float)
        key = E_[:, 0] * n + E_[:, 1]
        o = np.argsort(key)
        key, logl = key[o], logl[o]

        def lt_of(i, j):
            return logl[np.searchsorted(key, np.minimum(i, j) * n + np.maximum(i, j))]
    # N: the vertex whose star has the best smallest angle
    e = np.stack([np.exp(lt_of(F[:, (c + 1) % 3], F[:, (c + 2) % 3])) for c in range(3)], 1)
    cosang = np.stack([(e[:, (c + 1) % 3] ** 2 + e[:, (c + 2) % 3] ** 2 - e[:, c] ** 2)
                       / (2 * e[:, (c + 1) % 3] * e[:, (c + 2) % 3]) for c in range(3)], 1)
    worst = np.full(n, np.inf)
    np.minimum.at(worst, F.ravel(), np.repeat(np.arccos(np.clip(cosang, -1, 1)).min(1), 3))
    N = int(np.argmax(worst))
    starN = (F == N).any(1)
    link = np.unique(F[starN])
    link = link[link != N]

    tri = _Tri(F[~starN], lt_of)
    u = np.zeros(n)
    u[link] = -2 * lt_of(np.full(len(link), N), link)
    fixed = np.zeros(n, bool)
    fixed[link] = fixed[N] = True
    I = np.flatnonzero(~fixed)
    # start: the boundary values extended harmonically over the graph
    a_, b_ = np.roll(tri.T, -1, 1).ravel(), np.roll(tri.T, -2, 1).ravel()
    A = sp.coo_matrix((np.ones(2 * len(a_)), (np.r_[a_, b_], np.r_[b_, a_])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    Lg = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    u[I] = _spd_factor(Lg[I][:, I].tocsc()).solve(-(Lg[I][:, fixed] @ u[fixed]))
    lu = None

    def grad():
        nflip = tri.delaunay(u)
        ang = tri.angles(u)
        Th = np.bincount(tri.T.ravel(), ang.ravel(), n)
        return 2 * np.pi - Th, ang, nflip

    g, ang, flips = grad()
    it = 0
    for it in range(max_iter):
        I = np.flatnonzero(~fixed)
        if np.abs(g[I]).max(initial=0.0) < tol:
            break
        H = _cot_laplacian(tri, ang, n, I)
        d = None
        if lu is not None:                     # the last factorisation preconditions CG (H moves little)
            d, ok = spla.cg(H, -g[I], M=spla.LinearOperator(H.shape, lu.solve), rtol=CG_TOL, maxiter=CG_MAX)
            d = d if ok == 0 else None
        if d is None:
            lu = _spd_factor(H)
            d = -lu.solve(g[I])
        u0 = u[I].copy()
        # E is convex: find φ'(t) = ∇E·d ≤ 0 with |φ'(t)| ≤ 0.1 |φ'(0)|, by false position on
        # the bracket [lo, hi] (φ' is nearly linear along a Newton step), halving as a fallback
        lo, slo, hi, shi, t = 0.0, g[I] @ d, 1.0, None, 1.0
        for k in range(40):
            u[I] = u0 + t * d
            gt, ang_t, nf = grad()
            flips += nf
            slope = gt[I] @ d
            if slope <= 0:
                if t == 1.0 or slope > 0.1 * (g[I] @ d):
                    break
                lo, slo = t, slope
            else:
                hi, shi = t, slope
            t = lo + (hi - lo) * slo / (slo - shi) if shi is not None and k < 8 else 0.5 * (lo + hi)
            t = min(max(t, lo + 0.05 * (hi - lo)), hi - 0.05 * (hi - lo))
        g, ang = gt, ang_t
    I = np.flatnonzero(~fixed)
    z = _layout(tri, u, n)
    if info is not None:
        info.update(newton=it, flips=flips, angle_err=float(np.abs(g[I]).max(initial=0.0)), N=N, link=int(fixed.sum() - 1),
                    reflex=int(np.sum(np.bincount(tri.T.ravel(), ang.ravel(), n)[fixed] > np.pi + 1e-9)),
                    degenerate=int(np.sum(ang.min(1) < 1e-12)))
    # N → north pole, then centre. Seen from outside, projection from the north pole
    # mirrors the plane, so the counter-clockwise layout goes in conjugated.
    A0, B0 = np.conj(z), np.ones(n, complex)
    A0[N], B0[N] = 1.0, 0.0
    vA = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    return _mobius_centre(A0, B0, np.bincount(F.ravel(), np.repeat(vA, 3), n))
