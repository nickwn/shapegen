"""Face charts of a polyhedron on the sphere: developing each face from its sphere
image, for :mod:`~shapegen.geom.chord_map` (``codec.encode(method="polyhedral")``).

A closed polyhedron is flat except at its vertices, cones of angle defect κ, so
its conformal map to the sphere has a closed form: in a stereographic chart the
metric is ``c |W(ζ)| |dζ|`` with ``W = Π (ζ − p_v)^{a_v}``, ``a_v = −κ_v/2π``.
Given sphere positions ``p_v``, :meth:`PolyMap._develop_at` develops faces by
integrating ``W`` along rays from a chart centre to the corners (adaptive
Gauss–Jacobi rules at the cones, the near cones one by one and the rest as a
per-face far-field series, :func:`_far_field`). The encoder uses it to put its
check points on the true surface, and :class:`~shapegen.geom.edge_periods.EdgeSystem`
uses the same charts and rules for its edge periods.

The exact encoder that once lived here (a global Newton fit of the map and a
closed-form curved-patch spinor quadrature) was removed on 2026-10-11 once the
chord-map encoder matched it on coarse polyhedra (within −5 … +2 % at L = 16) and
became triangulation-invariant; the map's Newton solve survives as
:meth:`EdgeSystem.solve`, the encoder's fallback on very coarse meshes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["angle_defects", "PolyMap"]

NQ = 8             # Gauss nodes per ray segment
SEP = 1.0          # ... once every cone not at the segment's end is ≥ SEP segment lengths away
NEAR = 2.0         # cones within NEAR face radii enter W one by one, the rest as a Taylor series
NT = 64            # Taylor terms of the far cones (ratio ≤ 1.15 / NEAR at the nodes: < 1e-15)
RC = 1.2           # their check circle, in face radii: just outside the nodes (|ζ| ≤ R_f), so the
                   # tree's block (≥ the check circle) stays small; no loss as RC > |ζ|/R_f
HALF = 3.0         # far-field tree: the square |Re w|, |Im w| ≤ HALF of each global chart
P_FMM = 56         # its expansion order (error ~0.64^P at a face's check circle)


# ------------------------------------------------------------------ polyhedron data
def angle_defects(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """κ_v = 2π − Σ face angles at v."""
    k = np.full(len(V), 2 * np.pi)
    for c in range(3):
        i, j, l = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        u, w = V[j] - V[i], V[l] - V[i]
        np.add.at(k, i, -np.arctan2(np.linalg.norm(np.cross(u, w), axis=1),
                                    np.einsum("ij,ij->i", u, w)))
    return k


def _normalize(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def face_rotations(C: np.ndarray) -> np.ndarray:
    """``R_f`` with ``R_f c_f = (0, 0, −1)``: face centre ``c_f`` → ζ = 0 of its chart."""
    C = _normalize(C)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(C, b)
    cth = C @ b
    K = np.zeros((len(C), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    return np.eye(3) + K + K @ K / (1.0 + cth)[:, None, None]


def chart(R: np.ndarray, X: np.ndarray, inverse=False) -> np.ndarray:
    """Stereographic coordinate ζ (pole at +z) of unit vectors ``X`` (…,3) after
    rotation ``R``; ``1/ζ`` where ``inverse`` (a bool or mask). Near the pole
    ``1 − Y_z`` cancels, so there it is ``(Y_x² + Y_y²)/(1 + Y_z)``: a cone opposite
    a face (a regular tetrahedron's) would otherwise get 1e-4 relative error."""
    Y = np.einsum("...ij,...j->...i", R, X)
    w, north = Y[..., 0] + 1j * Y[..., 1], Y[..., 2] > 0
    up, down = 1.0 + Y[..., 2], 1.0 - Y[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(north, up / np.conj(w), w / down)
        if np.any(inverse):
            z = np.where(inverse, np.where(north, np.conj(w) / up, down / w), z)
    return z


def _unstereo(z: np.ndarray) -> np.ndarray:
    r2 = np.abs(z) ** 2
    return np.stack([2 * z.real / (r2 + 1), 2 * z.imag / (r2 + 1), (r2 - 1) / (r2 + 1)], -1)


def cone_gauge(V, F, a_v, X):
    """The code's gauge of sphere positions ``X`` (per vertex) of the polyhedron
    ``(V, F)`` with cone exponents ``a_v``: the ``|a|``-weighted cone centroid to 0
    (exact), then the rotation that best matches the cones (``|a|``-weighted) to
    their directions from the surface centroid. The rotation follows the input's
    orientation instead of the start's, so the code is deterministic."""
    from .conformal import ball_automorphism

    ci = np.flatnonzero(a_v != 0)
    w = np.abs(a_v[ci]) / np.abs(a_v[ci]).sum()
    Sv = X                                                              # the cones are among the vertices
    for _ in range(200):
        cvec = w @ Sv[ci]
        if np.linalg.norm(cvec) < 1e-13:
            break
        Sv = ball_automorphism(Sv, cvec / max(1.0, 2.0 * np.linalg.norm(cvec)))
    Xc = Sv[ci]
    E1, E2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    A = 0.5 * np.linalg.norm(np.cross(E1, E2), axis=1)
    Y = _normalize(V[ci] - A @ V[F].mean(1) / A.sum())
    U, _, Vt = np.linalg.svd((w[:, None] * Y).T @ Xc)                   # Kabsch: R X ≈ Y
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return Sv @ R.T


# ------------------------------------------------------------------ the cones a face sees
def _ranges(ptr, rows):
    """For CSR row pointers ``ptr``: the entries of ``rows``, concatenated, and which
    position in ``rows`` each belongs to."""
    cnt = ptr[rows + 1] - ptr[rows]
    owner = np.repeat(np.arange(len(rows)), cnt)
    return owner, np.arange(cnt.sum()) + np.repeat(ptr[rows] - (np.cumsum(cnt) - cnt), cnt)


class _Cones:
    """The cones as each face's chart sees them, one row per face (or per point,
    after :meth:`take`). ``log(K_f W(ζ)) = C + Σ_near a log(1 − ζq) + Σ_n G_n ζⁿ``:

    - near cones (within ``NEAR`` face radii, CSR ``ptr``; cone ``idx``, ``q = 1/P``,
      exponent ``an``, phase form ``pn``) enter one by one. Every row has at least
      one entry; empty rows hold a dummy with ``a = 0``.
    - far cones enter through the Taylor coefficients ``G`` (rows, ``NT``) of
      their ``Σ a log(1 − ζq)``, and their part of the constant ``C``."""

    def __init__(self, ptr, idx, q, an, pn, C, G):
        self.ptr, self.idx, self.q, self.an, self.pn, self.C, self.G = ptr, idx, q, an, pn, C, G

    def take(self, rows):
        _, pos = _ranges(self.ptr, rows)
        ptr = np.r_[0, np.cumsum(self.ptr[rows + 1] - self.ptr[rows])]
        return _Cones(ptr, self.idx[pos], self.q[pos], self.an[pos], self.pn[pos],
                      self.C[rows], self.G[rows])

    def replace(self, rows, other):
        """A copy with rows ``rows`` replaced by the rows of ``other`` (in order)."""
        n = len(self.ptr) - 1
        src = np.zeros(n, int)                                        # 0: self, 1: other
        src[rows] = 1
        pos_in = np.zeros(n, int)
        pos_in[rows] = np.arange(len(rows))
        cnt = np.where(src == 1, np.diff(other.ptr)[pos_in], np.diff(self.ptr))
        ptr = np.r_[0, np.cumsum(cnt)]
        out = {}
        keep = np.flatnonzero(src == 0)
        dst_k = _ranges(ptr, keep)[1]
        dst_o = _ranges(ptr, rows)[1]
        src_k = _ranges(self.ptr, keep)[1]
        src_o = np.arange(other.ptr[-1])
        for name in ("idx", "q", "an", "pn"):
            a_, b_ = getattr(self, name), getattr(other, name)
            v = np.zeros(ptr[-1], a_.dtype)
            v[dst_k], v[dst_o] = a_[src_k], b_[src_o]
            out[name] = v
        C, G = self.C.copy(), self.G.copy()
        C[rows], G[rows] = other.C, other.G
        return _Cones(ptr, out["idx"], out["q"], out["an"], out["pn"], C, G)

    @property
    def owner(self):
        return np.repeat(np.arange(len(self.ptr) - 1), np.diff(self.ptr))

    def logW(self, z, deriv: bool = False, arg: bool = False):
        """``log(K_f W(z))`` at points ``z`` (rows, n), the branch fixed at z = 0, and
        with ``deriv`` also its ζ-derivative ``Σ −a q/(1 − ζq) + Σ (n+1) G_n ζⁿ``.
        Real log and ``arctan2`` separately: half the cost of a complex log; ``arg``
        returns only the imaginary part (``arg K_f W``), without the log."""
        u = 1.0 - z[self.owner] * self.q[:, None]
        if arg:
            t = np.arctan2(u.imag, u.real) * self.an[:, None]
        else:
            t = (0.5 * np.log(u.real ** 2 + u.imag ** 2) + 1j * np.arctan2(u.imag, u.real)) * self.an[:, None]
        acc = np.zeros_like(z)
        for n in range(self.G.shape[1] - 1, -1, -1):              # Horner: Σ G_n zⁿ⁺¹
            acc = (acc + self.G[:, n, None]) * z
        lw = np.add.reduceat(t, self.ptr[:-1], axis=0) + ((self.C[:, None] + acc).imag if arg else self.C[:, None] + acc)
        if not deriv:
            return lw
        d = np.zeros_like(z)
        for n in range(self.G.shape[1] - 1, 0, -1):
            d = (d + (n + 1) * self.G[:, n, None]) * z
        d += self.G[:, 0, None]
        return lw, d + np.add.reduceat(-(self.an * self.q)[:, None] / u, self.ptr[:-1], axis=0)

    def rows(self, z, rows=None, deriv: bool = False, budget: int = 4_000_000):
        """:meth:`logW` of rows ``rows`` (all if None) at ``z`` (len(rows), n), in
        chunks of ≤ ``budget`` (entry × point) terms."""
        rows = np.arange(len(self.ptr) - 1) if rows is None else rows
        cnt = np.cumsum((self.ptr[rows + 1] - self.ptr[rows]) * z.shape[1])
        cuts = np.r_[0, np.searchsorted(cnt, np.arange(budget, cnt[-1] if len(cnt) else 0, budget)), len(rows)]
        out = [self.take(rows[a:b]).logW(z[a:b], deriv) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
        if deriv:
            return (np.concatenate([o[0] for o in out]) if out else np.zeros_like(z),
                    np.concatenate([o[1] for o in out]) if out else np.zeros_like(z))
        return np.concatenate(out) if out else np.zeros_like(z)


def _cone_terms(q, a, pn):
    """Each cone's constant ``log(−P) − ½log(1+|P|²)`` (``pn``: near the chart centre,
    ``|P| ≤ 1``, or a corner of the face), else ``−½log(1+|q|²)``. The second drops a
    phase (a rotation of the face's developed plane, which nothing depends on), so no
    term grows with ``|P|``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return a * np.where(pn, -np.log(-q) - 0.5 * np.log1p(1.0 / np.abs(q) ** 2),
                            -0.5 * np.log1p(np.abs(q) ** 2))


def _face_cones(S, F, Rf, ci, a, conepos, C=None):
    """:class:`_Cones` of every face, from the vertices' sphere positions ``S``; ``C``:
    the faces' chart centres (default the normalised vertex means)."""
    from scipy.spatial import cKDTree

    m, X = len(F), S[ci]
    C = _normalize(S[F].mean(1)) if C is None else _normalize(C)
    rho = np.arccos(np.clip(np.einsum("fki,fi->fk", S[F], C), -1, 1)).max(1)    # face radius
    reach = np.minimum(NEAR * rho, np.pi)
    lists = cKDTree(X).query_ball_point(C, 2 * np.sin(reach / 2) + 1e-12)
    cnt = np.array([max(len(l), 1) for l in lists])
    ptr = np.r_[0, np.cumsum(cnt)]
    idx = np.zeros(ptr[-1], int)
    an = np.zeros(ptr[-1])
    for f, l in enumerate(lists):                                  # dummies keep a = 0
        if l:
            idx[ptr[f]:ptr[f + 1]] = l
            an[ptr[f]:ptr[f + 1]] = a[l]
    owner = np.repeat(np.arange(m), cnt)
    q = chart(Rf[owner], X[idx], inverse=True)
    own = np.zeros(len(idx), bool)
    for k in range(3):
        own |= idx == conepos[F[owner, k]]
    pn = own | (np.abs(q) >= 1)
    Cf = np.add.reduceat(_cone_terms(q, an, pn), ptr[:-1])
    far_C, G = _far_field(X, a, C, Rf, np.tan(rho / 2), ptr, idx, an)
    Cf += far_C
    return _Cones(ptr, idx, q, an, pn, Cf, G)


def _dsigma(y, v):
    """Differential of the stereographic chart at the rotated point ``y`` along ``v``."""
    return ((v[..., 0] + 1j * v[..., 1]) / (1 - y[..., 2])
            + (y[..., 0] + 1j * y[..., 1]) * v[..., 2] / (1 - y[..., 2]) ** 2)


def _far_field(X, a, C, Rf, R, ptr, idx, an):
    """The far cones' part of every face's ``log(K_f W)``: its constant (real part)
    and Taylor coefficients ``G`` (faces, ``NT``) in the face chart.

    The cones split, per face, into near ones (entered exactly elsewhere), a *tree*
    part and *series* ones. Each face is served by one of two global stereographic
    charts ``w`` (the one whose centre is in its hemisphere). If its check circle
    (radius ``RC R_f`` in the face chart) lies in the chart's core ``|w| ≤ 1.5``,
    :func:`~shapegen.geom.log_fmm.far_taylor` sums the cones outside a 3 × 3 block of
    boxes around it, at the level whose boxes cover its near cones and are ≥ 4 × its
    check circle, and the cones near the chart's pole (``|w| > HALF``) are one series
    about the chart centre. That tree part is taken to the face chart on the check
    circle, where ``Σ a [log(ζ−ζ_k) − ½log(1+|ζ_k|²)] = Σ a [log(w−w_k) −
    ½log(1+|w_k|²)] + (A/2) log(dζ/dw)`` up to an imaginary constant, and read off by
    FFT. Every other cone (the rest of the block; all cones for a face outside the
    core) is a series ``−a qⁿ/n`` in the face chart itself."""
    from .log_fmm import far_taylor

    m, PT = len(C), NT + 32                                        # aliasing ≤ Σ|a| (RC/NEAR)^PT
    owner = np.repeat(np.arange(m), np.diff(ptr))
    nearent = np.flatnonzero(an != 0)
    Cfar = np.zeros(m)
    G = np.zeros((m, NT), complex)
    th = 2 * np.pi * np.arange(PT) / PT
    pf_all, pk_all = [], []
    for Rg, sel in ((np.eye(3), C[:, 2] <= 0), (np.diag([1.0, -1.0, -1.0]), C[:, 2] > 0)):
        fs = np.flatnonzero(sel)
        if len(fs) == 0:
            continue
        fpos = np.full(m, -1)
        fpos[fs] = np.arange(len(fs))
        w = chart(Rg, X)
        c = chart(Rg, C[fs])
        zc = (RC * R[fs])[:, None] * np.exp(1j * th)               # check circle, face chart
        xc = np.einsum("fji,fpj->fpi", Rf[fs], _unstereo(zc))
        wc = chart(Rg, xc)
        ent = nearent[sel[owner[nearent]]]
        ef, ek = fpos[owner[ent]], idx[ent]                         # near cones of these faces
        r_near = np.zeros(len(fs))
        np.maximum.at(r_near, ef, np.abs(w[ek] - c[ef]))
        r_check = np.abs(wc - c[:, None]).max(1)
        dom = np.abs(w) <= HALF
        core = np.abs(wc).max(1) <= 1.5
        # finest level whose box covers the near cones and puts the check circle within
        # 0.6 of the local expansion's radius (1.5 box widths from the box centre)
        lev = np.zeros(len(fs), int)
        for L_ in range(20, 0, -1):
            hb = 2 * HALF / 2 ** L_
            ctr = (-HALF + (np.floor((c.real + HALF) / hb) + 0.5) * hb) + 1j * (-HALF + (np.floor((c.imag + HALF) / hb) + 0.5) * hb)
            ok = (lev == 0) & (hb >= r_near) & (np.abs(c - ctr) + r_check <= 0.9 * hb)
            lev[ok] = L_
        lev[~core] = 0
        di = np.flatnonzero(dom)
        T, bptr, bidx = far_taylor(w[di], a[di], np.where(core, c, 0.0), lev, HALF, P_FMM)
        T[~core] = 0.0
        # series cones: the block minus the near ones (core), else every cone but the near ones
        nc_ = len(X)
        chw = -0.5 * a * np.log1p(np.abs(w) ** 2)
        polar = np.flatnonzero(~dom)
        bf = np.repeat(np.arange(len(fs)), np.diff(bptr))
        bk = di[bidx]
        nearkey = ef * nc_ + ek
        keep = core[bf] & ~np.isin(bf * nc_ + bk, nearkey)
        of = np.flatnonzero(~core)
        of_f, of_k = np.repeat(of, nc_), np.tile(np.arange(nc_), len(of))
        okeep = ~np.isin(of_f * nc_ + of_k, nearkey)
        pf_all.append(fs[np.r_[bf[keep], of_f[okeep]]])
        pk_all.append(np.r_[bk[keep], of_k[okeep]])
        npol = ~dom[ek]
        A_tree = (a[di].sum() - np.bincount(bf, a[bk], len(fs)) + a[polar].sum()
                  - np.bincount(ef, a[ek] * npol, len(fs)))
        ch_tree = (chw[di].sum() - np.bincount(bf, chw[bk], len(fs)) + chw[polar].sum()
                   - np.bincount(ef, chw[ek] * npol, len(fs)))
        # the tree part on the check circle (core faces)
        cf = np.flatnonzero(core)
        if len(cf):
            dw = wc[cf] - c[cf, None]
            val = np.zeros(dw.shape, complex)
            for n_ in range(P_FMM, -1, -1):
                val = val * dw + T[cf, n_, None]
            if len(polar):                                           # polar cones about the chart centre
                pc_ = np.zeros(P_FMM + 1, complex)
                inv = 1.0 / w[polar]
                pc_[0] = np.sum(a[polar] * np.log(np.abs(w[polar])))
                pwp = np.ones(len(polar), complex)
                for n_ in range(1, P_FMM + 1):
                    pwp = pwp * inv
                    pc_[n_] = -np.sum(a[polar] * pwp) / n_
                pv = np.zeros(dw.shape, complex)
                for n_ in range(P_FMM, -1, -1):
                    pv = pv * wc[cf] + pc_[n_]
                val += pv
                cpos = np.full(len(fs), -1)
                cpos[cf] = np.arange(len(cf))
                pol = (~dom[ek]) & core[ef]                          # the near ones out again
                for e_f, e_k in zip(ef[pol].tolist(), ek[pol].tolist()):
                    t_ = np.full(PT, a[e_k] * np.log(np.abs(w[e_k])), complex)
                    pwp = np.ones(PT, complex)
                    for n_ in range(1, P_FMM + 1):
                        pwp = pwp * wc[e_f] / w[e_k]
                        t_ -= a[e_k] * pwp / n_
                    val[cpos[e_f]] -= t_
            tv = np.cross(xc[cf], np.where(np.abs(xc[cf][..., 2:3]) < 0.9, [0, 0, 1.0], [1.0, 0, 0]))
            ratio = (_dsigma(np.einsum("fij,fpj->fpi", Rf[fs[cf]], xc[cf]), np.einsum("fij,fpj->fpi", Rf[fs[cf]], tv))
                     / _dsigma(np.einsum("ij,fpj->fpi", Rg, xc[cf]), np.einsum("ij,fpj->fpi", Rg, tv)))
            lr = np.log(np.abs(ratio)) + 1j * np.unwrap(np.angle(ratio), axis=1)
            val += ch_tree[cf, None] + 0.5 * A_tree[cf, None] * lr
            f_ = np.fft.fft(val, axis=1) / PT
            Cfar[fs[cf]] = f_[:, 0].real
            G[fs[cf]] = f_[:, 1:NT + 1] / ((RC * R[fs[cf]])[:, None] ** np.arange(1, NT + 1))
    # series cones, in each face's own chart: powers by cumulative products, summed
    # per face over the face-sorted pairs. A cone at |P| contributes (1.25 R/|P|)ⁿ at
    # the nodes (|ζ| ≤ R, measured), so it gets only the terms that reach 1e-16:
    # most of a block is many face radii away and needs a fraction of NT.
    pf, pk = np.concatenate(pf_all), np.concatenate(pk_all)
    srt = np.argsort(pf, kind="stable")
    pf, pk = pf[srt], pk[srt]
    inv_n = -1.0 / np.arange(1, NT + 1)
    for s0 in range(0, len(pf), 500_000):
        f_, k_ = pf[s0:s0 + 500_000], pk[s0:s0 + 500_000]
        q = chart(Rf[f_], X[k_], inverse=True)
        Cfar += np.bincount(f_, -0.5 * a[k_] * np.log1p(np.abs(q) ** 2), m)
        with np.errstate(divide="ignore"):
            nt = np.ceil(-37.0 / np.log(np.minimum(1.25 * R[f_] * np.abs(q), 0.99)))
        nt = np.minimum(8 * np.ceil(np.maximum(nt, 1) / 8), NT).astype(int)
        for n_ in np.unique(nt):
            sel = np.flatnonzero(nt == n_)
            fsel = f_[sel]
            v = np.cumprod(np.broadcast_to(q[sel, None], (len(sel), n_)), axis=1) * (a[k_[sel]][:, None] * inv_n[:n_])
            st = np.flatnonzero(np.r_[True, fsel[1:] != fsel[:-1]])
            G[fsel[st], :n_] += np.add.reduceat(v, st, axis=0)
    return Cfar, G


# ------------------------------------------------------------------ W along rays
def _segments(end, cones, own):
    """Split rays ``0 → end`` (one per row of ``cones``) into ``(ray, t0, t1)``
    segments, bisecting until every near cone but the endpoint's own (``own``, −1 if
    none) is ≥ ``SEP`` segment lengths away: ``NQ``-point Gauss then errs < ~1e-13
    (Bernstein ellipse ρ ≥ 5.8). Far cones are that far by construction."""
    r, t0, t1 = np.arange(len(end)), np.zeros(len(end)), np.ones(len(end))
    out = []
    for _ in range(50):
        c = cones.take(r)
        o = c.owner
        with np.errstate(divide="ignore", invalid="ignore"):
            P = 1.0 / c.q
        P = np.where(np.isfinite(P), P, 1e300)
        e = end[r][o]
        s = np.clip(np.real(P * np.conj(e)) / np.maximum(np.abs(e) ** 2, 1e-300), t0[o], t1[o])
        d = np.abs(P - s * e)
        d[(t1[o] == 1.0) & (c.idx == own[r][o]) | (c.an == 0)] = np.inf
        ok = np.minimum.reduceat(d, c.ptr[:-1]) >= SEP * (t1 - t0) * np.abs(end[r])
        out.append((r[ok], t0[ok], t1[ok]))
        r, t0, t1 = r[~ok], t0[~ok], t1[~ok]
        if len(r) == 0:
            break
        mid = 0.5 * (t0 + t1)
        r, t0, t1 = np.r_[r, r], np.r_[t0, mid], np.r_[mid, t1]
    out.append((r, t0, t1))                                        # a cone on the ray: give up
    return (np.concatenate(c) for c in zip(*out))


def _gj_batch(alpha, n):
    """Gauss–Jacobi nodes/weights on [0, 1] for ``∫ (1−t)^α g(t) dt``, one rule per
    ``alpha`` (Golub–Welsch on the stacked Jacobi matrices: a scan has a different
    cone exponent at every vertex). Solved once per distinct exponent."""
    ua, inv = np.unique(np.asarray(alpha, float), return_inverse=True)
    x, w = _gj_rules(ua, n)
    return x[inv], w[inv]


def _gj_rules(alpha, n):
    al = np.asarray(alpha, float)[:, None]
    k = np.arange(n, dtype=float)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -al ** 2 / ((2 * k + al) * (2 * k + al + 2))
        d[:, 0] = -al[:, 0] / (al[:, 0] + 2)
        kk = k[:, 1:]
        off = np.sqrt(4 * kk * (kk + al) * kk * (kk + al) / ((2 * kk + al) ** 2 * (2 * kk + al + 1) * (2 * kk + al - 1)))
    J = np.zeros((len(al), n, n))
    i = np.arange(n)
    J[:, i, i] = d
    J[:, i[:-1], i[1:]] = J[:, i[1:], i[:-1]] = off
    x, vec = np.linalg.eigh(J)
    w = (2.0 ** (al[:, 0] + 1) / (al[:, 0] + 1))[:, None] * vec[:, 0, :] ** 2
    return (1 + x) / 2, w * 2.0 ** (-al - 1)


def _quadrature(end, aend, cones, own):
    """Nodes and weights of ``∫_0^{end} W dζ``: ``(ray, t (s, NQ), w (s, NQ), a_end (s,))``
    per segment. A segment ending at a cone uses Gauss–Jacobi with that cone's
    exponent, and its weights carry ``(1−t)^a``."""
    r, t0, t1 = _segments(end, cones, own)
    av = np.where(t1 == 1.0, aend[r], 0.0)
    x, w = _gj_batch(av, NQ)
    h = (t1 - t0)[:, None]
    return r, t0[:, None] + h * x, h ** (1 + av[:, None]) * w, av


def _chunks(r, cones, budget=4_000_000):
    """Index slices of the segments ``r`` with ≤ ``budget`` (entry × node) terms each."""
    cnt = np.diff(cones.ptr)[r] * NQ
    cs = np.cumsum(cnt)
    cuts = np.searchsorted(cs, np.arange(budget, cs[-1] if len(cs) else 0, budget))
    edges = np.r_[0, cuts, len(r)]
    return [slice(a_, b_) for a_, b_ in zip(edges[:-1], edges[1:]) if b_ > a_]


def _rays(end, aend, cones, own):
    """``K_f ∫_0^{end} W dζ`` per ray (one per row of ``cones``). ``aend`` is the
    endpoint's exponent (0 if flat) and ``own`` its cone (−1 if flat)."""
    out = np.zeros(len(end), complex)
    R_, T_, W_, A_ = _quadrature(end, aend, cones, own)
    for sl in _chunks(R_, cones):
        r, t, w, av = R_[sl], T_[sl], W_[sl], A_[sl]
        L = cones.take(r).logW(t * end[r][:, None]) - av[:, None] * np.log1p(-t)
        v = end[r] * np.sum(np.exp(L) * w, 1)
        out += np.bincount(r, v.real, len(end)) + 1j * np.bincount(r, v.imag, len(end))
    return out


# ------------------------------------------------------------------ the map
class PolyMap:
    """A polyhedron's face developments from given sphere positions: its cones
    (``a_v``, ``ci``, ``conepos``), its 3D edges and angles, and the chart
    orientation ``sigma`` read off the start ``S0``."""

    def __init__(self, V, F, S0=None, cone_tol: float = 1e-9):
        self.V, self.F = np.asarray(V, float), np.asarray(F)
        kap = angle_defects(self.V, self.F)
        self.cone = np.abs(kap) > cone_tol
        self.a_v = np.where(self.cone, -kap / (2 * np.pi), 0.0)
        self.ci = np.flatnonzero(self.cone)
        self.conepos = np.full(len(self.V), -1)                    # vertex → cone index
        self.conepos[self.ci] = np.arange(len(self.ci))
        E1, E2 = self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        self.E1, self.E2 = E1, E2
        self.ll1, self.ll2 = np.log(np.linalg.norm(E1, axis=1)), np.log(np.linalg.norm(E2, axis=1))
        self.phi = np.arctan2(np.linalg.norm(np.cross(E1, E2), axis=1), np.einsum("ij,ij->i", E1, E2))
        if S0 is None:                                             # radial start
            S0 = _normalize(self.V - self.V.mean(0))
        zf = chart(face_rotations(S0[self.F].mean(1))[:, None], S0[self.F])
        o = np.sign(np.imag(np.conj(zf[:, 1] - zf[:, 0]) * (zf[:, 2] - zf[:, 0])))
        self.sigma = 1.0 if np.sum(o > 0) >= np.sum(o < 0) else -1.0   # chart orientation

    def _develop_at(self, S, faces, C):
        """Charts, cones and developed corners of ``faces`` with chart centres ``C``."""
        Rf = face_rotations(_normalize(C))
        cones = _face_cones(S, self.F[faces], Rf, self.ci, self.a_v[self.ci], self.conepos, C=C)
        zeta = chart(Rf[:, None], S[self.F[faces]])
        D = np.stack([_rays(zeta[:, k], self.a_v[self.F[faces, k]], cones, self.conepos[self.F[faces, k]])
                      for k in range(3)], 1)
        return Rf, cones, D

    def _shape_err(self, D, faces=None):
        """Per face: how far the developed triangle is from the 3D one (log side ratio,
        angle; scale-free), and the smallest barycentric coordinate of the chart
        centre (developed at 0) in it."""
        f = np.arange(len(self.F)) if faces is None else faces
        e1, e2 = D[:, 1] - D[:, 0], D[:, 2] - D[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.maximum(np.abs(np.log(np.abs(e1) / np.abs(e2)) - self.ll1[f] + self.ll2[f]),
                             np.abs(np.angle(e2 / e1 * np.exp(-1j * self.sigma * self.phi[f]))))
            A = np.imag(np.conj(e1) * e2)
            b = np.stack([np.imag(np.conj(D[:, 1]) * D[:, 2]), np.imag(np.conj(D[:, 2]) * D[:, 0]),
                          np.imag(np.conj(D[:, 0]) * D[:, 1])], 1) / A[:, None]
        return np.where(np.isfinite(err), err, np.inf), np.where(np.isfinite(b), b, -np.inf).min(1)
