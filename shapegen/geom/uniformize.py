"""Discrete uniformization of a genus-0 triangle mesh to the sphere.

The start of the exact polyhedral map (:mod:`shapegen.geom.poly_conformal`): the
discrete conformal map of Springborn (*Ideal hyperbolic polyhedra and discrete
uniformization*, 2020), computed as in Gillespie, Springborn & Crane (*Discrete
conformal equivalence of polyhedral surfaces*, 2021). A Newton solve over log
scale factors ``u`` on an intrinsic triangulation kept Delaunay by edge flips;
the flips update the Penner coordinates by Ptolemy's relation, so they never
change the metric and keep the energy convex. Flips exist only inside this
module: the vertices, and so the output, are the input's. On a 26k-vertex head
it takes ~6 s (8 Newton steps) and starts the exact solve at a median face-angle
error of 0.13° (a Tutte embedding: 18°).

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
        a, b = np.roll(self.T, -1, 1), np.roll(self.T, -2, 1)
        return self.lt[self.E] + 0.5 * (u[a] + u[b])

    def delaunay(self, u, tol=1e-12, max_rounds=10_000):
        """Flip interior edges until the decorated Delaunay condition holds, in
        independent sets (no two share a face); returns the number of flips."""
        nflip = 0
        for _ in range(max_rounds):
            c_ = self.loglen(u)                                        # the side opposite each corner
            a_, b_ = np.roll(c_, -1, 1), np.roll(c_, -2, 1)
            x = np.minimum(np.stack([a_ - b_, b_ - a_, 2 * c_ - a_ - b_]), 700.0)
            term = (np.exp(x[0]) + np.exp(x[1]) - np.exp(x[2])).ravel()   # (a²+b²−c²)/(ab)
            inner = np.flatnonzero(self.ef[:, 1] >= 0)
            s = term[self.ef[inner, 0]] + term[self.ef[inner, 1]]
            bad = s < -tol
            if not bad.any():
                return nflip
            cand, pri = inner[bad], -s[bad]
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
            self._flip(cand[keep], u)
            nflip += len(keep)
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
        self._sides()

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


def _cot_laplacian(tri, ang, n):
    """``−∂Θ/∂u``: ½ Σ cot of the opposite angles on each edge, as a Laplacian. A
    degenerate face (angles π, 0, 0: the convex extension of the energy) is linear
    in ``u`` and adds nothing."""
    a, b = np.roll(tri.T, -1, 1).ravel(), np.roll(tri.T, -2, 1).ravel()
    flat = (ang.min(1) < 1e-12)[:, None]
    with np.errstate(divide="ignore"):
        w = np.where(flat, 0.0, 0.25 / np.tan(ang)).ravel()
    W = sp.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return sp.diags(np.asarray(W.sum(1)).ravel()) - W


def _layout(tri, u, n):
    """Planar positions of the flat triangulation, laid out face by face (BFS)."""
    m = len(tri.T)
    l = np.exp(tri.loglen(u))
    ang = tri.angles(u)
    s1, s2 = tri.ef[:, 0], tri.ef[:, 1]
    inner = s2 >= 0
    G = sp.coo_matrix((np.ones(inner.sum() * 2), (np.r_[s1[inner] // 3, s2[inner] // 3],
                                                   np.r_[s2[inner] // 3, s1[inner] // 3])), shape=(m, m))
    order, pred = csg.breadth_first_order(G.tocsr(), 0, directed=False, return_predecessors=True)
    z = np.full(n, np.nan + 0j)
    f0 = tri.T[0]
    z[f0[0]] = 0.0
    z[f0[1]] = l[0, 2]                                                 # side opposite corner 2 = (0, 1)
    z[f0[2]] = l[0, 1] * np.exp(1j * ang[0, 0])
    for f in order[1:]:
        T = tri.T[f]
        k = np.flatnonzero(np.isnan(z[T]))
        if len(k) == 0:
            continue
        c = int(k[0])                                                  # the new corner
        i, j = T[(c + 1) % 3], T[(c + 2) % 3]
        # ccw (c, i, j): from i, turn by the angle at i toward c
        z[T[c]] = z[i] + (z[j] - z[i]) / abs(z[j] - z[i]) * l[f, (c + 2) % 3] * np.exp(1j * ang[f, (c + 1) % 3])
    return z


def uniformize(V: np.ndarray, F: np.ndarray, tol: float = 1e-10, max_iter: int = 100,
               info: dict | None = None) -> np.ndarray:
    """Unit-sphere positions ``(n, 3)`` of the vertices of the closed genus-0 mesh
    ``(V, F)``: its discrete uniformization (module docstring). ``tol`` bounds the
    angle-sum error ``max |Θ_i − 2π|``."""
    V, F = np.asarray(V, float), np.asarray(F)
    n = len(V)
    # N: the vertex whose star has the best smallest angle
    e = np.stack([np.linalg.norm(V[F[:, (c + 2) % 3]] - V[F[:, (c + 1) % 3]], axis=1) for c in range(3)], 1)
    cosang = np.stack([(e[:, (c + 1) % 3] ** 2 + e[:, (c + 2) % 3] ** 2 - e[:, c] ** 2)
                       / (2 * e[:, (c + 1) % 3] * e[:, (c + 2) % 3]) for c in range(3)], 1)
    worst = np.full(n, np.inf)
    np.minimum.at(worst, F.ravel(), np.repeat(np.arccos(np.clip(cosang, -1, 1)).min(1), 3))
    N = int(np.argmax(worst))
    starN = (F == N).any(1)
    link = np.unique(F[starN])
    link = link[link != N]

    def lt_of(i, j):
        return np.log(np.linalg.norm(V[i] - V[j], axis=1))

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
    u[I] = spla.spsolve(Lg[I][:, I].tocsc(), -(Lg[I][:, fixed] @ u[fixed]))

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
        H = _cot_laplacian(tri, ang, n)[I][:, I].tocsc()
        d = -spla.spsolve(H, g[I])
        u0 = u[I].copy()
        lo, hi, t = 0.0, 1.0, 1.0                                     # E is convex: find φ'(t) = ∇E·d ≤ 0
        for _ in range(40):
            u[I] = u0 + t * d
            gt, ang_t, nf = grad()
            flips += nf
            slope = gt[I] @ d
            if slope <= 0:
                if t == 1.0 or slope > 0.1 * (g[I] @ d):
                    break
                lo = t
            else:
                hi = t
            t = 0.5 * (lo + hi)
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
