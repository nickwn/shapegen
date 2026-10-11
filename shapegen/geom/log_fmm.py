"""2-D fast multipole sums for ``Φ(w) = Σ_k a_k log(w − w_k)``.

Greengard & Rokhlin's method on a sparse quadtree over the square
``[−half, half]²``: multipole expansions of the non-empty boxes (P2M at the
finest level, M2M up), local expansions of the boxes that are queried or have a
queried descendant (M2L across each one's interaction list, L2L down).
:class:`LogFMM` hands each query point the Taylor series, about that point, of
the sources outside the 3 × 3 block of boxes around it at a level it chooses,
and the list of the sources inside that block. Error ~ ρᵖ with ρ the ratio of a
point's distance from its box centre to 1.5 box widths; the constant term's
imaginary part depends on the logarithms' branches, its real part does not.
Expansions are kept in box units (``a_k / hᵏ``, ``b_l hˡ``), so deep levels do not
underflow.

The plan (tree, transfer matrices, block lists) depends only on the points, so
one plan serves many charge vectors: :meth:`LogFMM.taylor` is the map from charges
to Taylor coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb

__all__ = ["LogFMM", "far_taylor"]


class LogFMM:
    """The FMM plan for sources ``src`` and query points ``centers`` at tree levels
    ``level`` (one per query). Points and sources must lie in ``[−half, half]²``.
    ``ptr, idx``: each query's block sources, as CSR."""

    def __init__(self, src, centers, level, half, p: int = 40):
        src, centers = np.asarray(src, complex), np.asarray(centers, complex)
        level = np.asarray(level, int)
        self.p, self.half, self.nsrc = p, half, len(src)
        m = len(centers)
        L = self.L = int(level.max(initial=0))
        B = self.B = comb(np.arange(2 * p + 2)[:, None], np.arange(2 * p + 2)[None, :])
        l = np.arange(1, p + 1)

        def cell(w, lev):
            n_ = 2 ** lev
            h_ = 2 * half / n_
            return (np.clip(((w.real + half) / h_).astype(np.int64), 0, n_ - 1),
                    np.clip(((w.imag + half) / h_).astype(np.int64), 0, n_ - 1))

        def centre(ix, iy, lev):
            h_ = 2 * half / 2 ** lev
            return (-half + (ix + 0.5) * h_) + 1j * (-half + (iy + 0.5) * h_)

        # ---- P2M: each source's row of its finest box's multipole (box units)
        sx, sy = cell(src, L)
        skeys = [None] * (L + 1)
        uk, inv = np.unique(sx * 2 ** L + sy, return_inverse=True)
        d = (src - centre(sx, sy, L)) / (2 * half / 2 ** L)
        phi = np.empty((len(src), p + 1), complex)
        phi[:, 0] = 1.0
        pw = np.ones_like(d)
        for j in range(1, p + 1):
            pw = pw * d
            phi[:, j] = -pw / j
        self.p2m = (inv, phi)
        skeys[L] = uk
        # ---- M2M maps, finest level up: (level, parent index, selection, T)
        self.m2m = []
        for lev in range(L, 0, -1):
            n_ = 2 ** lev
            ix, iy = skeys[lev] // n_, skeys[lev] % n_
            upk, pinv = np.unique((ix // 2) * (n_ // 2) + iy // 2, return_inverse=True)
            for dx in (0, 1):
                for dy in (0, 1):
                    sel = np.flatnonzero((ix % 2 == dx) & (iy % 2 == dy))
                    if not len(sel):
                        continue
                    u = (dx - 0.5) + 1j * (dy - 0.5)              # (child − parent centre) / child size
                    T = np.zeros((p + 1, p + 1), complex)
                    T[0, 0] = 1.0
                    T[1:, 0] = -u ** l / (l * 2.0 ** l)
                    for ll in range(1, p + 1):
                        kk = np.arange(1, ll + 1)
                        T[ll, kk] = u ** (ll - kk) * B[ll - 1, kk - 1] / 2.0 ** ll
                    self.m2m.append((lev, pinv[sel], sel, T))
            skeys[lev - 1] = upk
        self.nsk = [len(k) for k in skeys]
        # ---- boxes needing local expansions: each query's box at its level, and their ancestors
        qx, qy = np.empty(m, np.int64), np.empty(m, np.int64)
        for lev in np.unique(level):
            q = level == lev
            qx[q], qy[q] = cell(centers[q], lev)
        tkeys = [None] * (L + 1)
        for lev in range(L, -1, -1):
            own = qx[level == lev] * 2 ** lev + qy[level == lev]
            if lev < L:
                n_c = 2 ** (lev + 1)
                ch = tkeys[lev + 1]
                own = np.r_[own, ((ch // n_c) // 2) * 2 ** lev + (ch % n_c) // 2]
            tkeys[lev] = np.unique(own)
        self.ntk = [len(k) for k in tkeys]
        # ---- per level: L2L from the parent, M2L over the interaction list
        k = np.arange(1, p + 1)
        self.l2l, self.m2l = [], []
        for lev in range(2, L + 1):
            n_ = 2 ** lev
            hl = 2 * half / n_
            tx, ty = tkeys[lev] // n_, tkeys[lev] % n_
            pidx = np.searchsorted(tkeys[lev - 1], (tx // 2) * (n_ // 2) + ty // 2)
            for dx in (0, 1):
                for dy in (0, 1):
                    sel = np.flatnonzero((tx % 2 == dx) & (ty % 2 == dy))
                    if not len(sel):
                        continue
                    u = (dx - 0.5) + 1j * (dy - 0.5)              # in child-box units
                    T = np.zeros((p + 1, p + 1), complex)
                    for ll in range(p + 1):
                        kk = np.arange(ll, p + 1)
                        T[ll, kk] = B[kk, ll] * u ** (kk - ll) / 2.0 ** kk
                    self.l2l.append((lev, pidx[sel], sel, T))
            sk = skeys[lev]
            for ox in range(-3, 4):
                for oy in range(-3, 4):
                    if abs(ox) <= 1 and abs(oy) <= 1:
                        continue
                    sxs, sys_ = tx + ox, ty + oy
                    ok = (sxs >= 0) & (sxs < n_) & (sys_ >= 0) & (sys_ < n_)
                    ok &= (np.abs((sxs // 2) - (tx // 2)) <= 1) & (np.abs((sys_ // 2) - (ty // 2)) <= 1)
                    cand = sxs * n_ + sys_
                    pos = np.minimum(np.searchsorted(sk, cand), len(sk) - 1)
                    ok &= sk[pos] == cand
                    if not ok.any():
                        continue
                    u = ox + 1j * oy                               # (source − target centre) / box size
                    T = np.zeros((p + 1, p + 1), complex)
                    T[0, 0] = np.log(-u) + np.log(hl)
                    T[0, 1:] = (-1.0) ** k / u ** k
                    T[1:, 0] = -1.0 / (l * u ** l)
                    T[1:, 1:] = (B[l[:, None] + k[None, :] - 1, k[None, :] - 1] * (-1.0) ** k[None, :]
                                 / u ** (l[:, None] + k[None, :]))
                    self.m2l.append((lev, np.flatnonzero(ok), pos[ok], T))
        # ---- per query: shift its box's local expansion to the point; list its 3x3 block's sources
        self.shift = []
        owners, members = [], []
        for lev in np.unique(level):
            q = np.flatnonzero(level == lev)
            n_ = 2 ** lev
            bx, by = qx[q], qy[q]
            box = np.searchsorted(tkeys[lev], bx * n_ + by)
            hq = 2 * half / n_
            dlt = (centers[q] - centre(bx, by, lev)) / hq
            pwq = np.ones((len(q), p + 1), complex)
            for j in range(1, p + 1):
                pwq[:, j] = pwq[:, j - 1] * dlt
            self.shift.append((lev, q, box, pwq, hq))
            sbx, sby = cell(src, lev)
            sb = sbx * n_ + sby
            order = np.argsort(sb, kind="stable")
            sbs = sb[order]
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    x_, y_ = bx + ox, by + oy
                    ok = (x_ >= 0) & (x_ < n_) & (y_ >= 0) & (y_ < n_)
                    b = x_[ok] * n_ + y_[ok]
                    lo, hi = np.searchsorted(sbs, b), np.searchsorted(sbs, b, side="right")
                    cnt = hi - lo
                    owners.append(np.repeat(q[ok], cnt))
                    members.append(order[np.arange(cnt.sum()) + np.repeat(lo - (np.cumsum(cnt) - cnt), cnt)])
        own = np.concatenate(owners) if owners else np.zeros(0, int)
        mem = np.concatenate(members) if members else np.zeros(0, int)
        srt = np.argsort(own, kind="stable")
        self.m = m
        self.ptr, self.idx = np.r_[0, np.cumsum(np.bincount(own, minlength=m))], mem[srt]

    def taylor(self, a):
        """Taylor coefficients ``(m, p+1)``, about each query point, of the far sum
        ``Σ a_k log(w − w_k)`` (charges may be complex)."""
        p, L = self.p, self.L
        a = np.asarray(a)
        inv, phi = self.p2m
        mult = [np.zeros((n_, p + 1), complex) for n_ in self.nsk]
        np.add.at(mult[L], inv, a[:, None] * phi)
        for lev, par, sel, T in self.m2m:
            np.add.at(mult[lev - 1], par, mult[lev][sel] @ T.T)
        loc = [np.zeros((n_, p + 1), complex) for n_ in self.ntk]
        i_l2l = i_m2l = 0
        for lev in range(2, L + 1):
            while i_l2l < len(self.l2l) and self.l2l[i_l2l][0] == lev:
                _, par, sel, T = self.l2l[i_l2l]
                loc[lev][sel] += loc[lev - 1][par] @ T.T
                i_l2l += 1
            while i_m2l < len(self.m2l) and self.m2l[i_m2l][0] == lev:
                _, tgt, srcb, T = self.m2l[i_m2l]
                loc[lev][tgt] += mult[lev][srcb] @ T.T
                i_m2l += 1
        coef = np.zeros((self.m, p + 1), complex)
        Bn = self.B
        for lev, q, box, pwq, hq in self.shift:
            a_ = loc[lev][box]
            for ll in range(p + 1):
                kk = np.arange(ll, p + 1)
                coef[q, ll] = (a_[:, kk] * pwq[:, kk - ll]) @ Bn[kk, ll] / hq ** ll
        return coef


def far_taylor(src, a, centers, level, half, p: int = 40):
    """For each query point ``centers[i]`` and its tree ``level[i]``: the Taylor
    coefficients ``(m, p+1)`` about the point of ``Σ a_k log(w − w_k)`` over the
    sources outside the 3 × 3 block of boxes around it at that level, and that
    block's sources as CSR ``(ptr, idx)``. Points and sources must lie in
    ``[−half, half]²``. Charges ``a`` may be complex."""
    f = LogFMM(src, centers, level, half, p)
    return f.taylor(a), f.ptr, f.idx
