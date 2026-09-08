"""Smoothed-head dataset from FaceScape ``models_reg``, for position models.

This is deliberately **not** the ρ-codec path. That path caps the four boundary
loops so cMCF and the Dirac solve have a closed genus-0 surface; a model over
vertex positions needs none of that, and the caps are invented geometry — the
neck opening alone is a 168 mm hole closed by a single centroid fan. So this
loader skips them.

What the meshes actually are: a **full head**, not a face mask — bbox 209 x 275
x 209 mm, so as deep as it is wide, with cranium, ears and a neck stub. Four
boundary loops (neck 96, mouth 96, two eye sockets 91/94), 26317 vertices,
52261 faces, connectivity identical across every downloaded root.

Three stages, in this order, each measured on 200 subjects (train 160, held-out
40, scored against the smoothed test face):

    align      Procrustes to the iterated mean. 13.5% of the raw spread is pose;
               removing it is worth 5-7% at useful k and costs ~1 s for 200.
    smooth     implicit Laplacian ``(M + tL)V' = M V``. **This is the one that
               matters.** PCA on raw heads *plateaus* at ~0.0106 (k=64 -> 128
               barely moves) because that floor IS the per-mesh scan noise:
               uncorrelated across subjects, so no basis can represent it, and it
               sits in the target. Smooth the target and the curve keeps falling
               — 0.0038 at k=128, **2.8x better**.
    encode     see :mod:`bottlegen.models.basis`.

Caveat worth carrying: a large share of those 26317 vertices sit on the bald
cranium and neck, which are nearly featureless and nearly identical between
subjects. A mean shape error over the whole head therefore flatters any model —
a basis that nails the skull and fumbles the nose still scores well.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from ..geom.mesh import Mesh
from ..geom.operators import cotan_laplacian
from .registered import read_obj

__all__ = ["load_heads", "procrustes_align", "laplacian_smooth", "build_dataset"]


def load_heads(roots, expression: str = "1_neutral", limit: int | None = None,
               normalise: bool = True, verbose: bool = False):
    """Load FaceScape ``models_reg`` heads from one or several dataset roots.

    ``roots`` is a path or an iterable of paths — FaceScape ships its ~450
    subjects split across several ``facescape_trainset_*`` directories, and they
    share one connectivity, so they concatenate into a single registered set.
    That is *checked* here rather than assumed: a mismatch would silently make
    the codes mean different things per subject.

    Subjects missing ``expression`` are skipped (446 of 447 have ``1_neutral``).

    ``normalise`` centres each head and scales it to unit mean radius. Do not
    read that as alignment — it fixes translation and scale only, and ~13.5% of
    the remaining spread is head pose. Run :func:`procrustes_align` for that.
    """
    roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(r) for r in roots]
    out, F0, skipped = [], None, 0
    for root in roots:
        subs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: (len(d.name), d.name))
        for d in subs:
            if limit is not None and len(out) >= limit:
                break
            path = d / "models_reg" / f"{expression}.obj"
            if not path.exists():
                skipped += 1
                continue
            V, F = read_obj(path)
            if F0 is None:
                F0 = F
            elif not np.array_equal(F, F0):
                raise ValueError(
                    f"{path}: connectivity differs from the first mesh loaded — "
                    f"the set is not template-registered, and concatenating these "
                    f"roots would produce codes that mean different things")
            if normalise:
                V = V - V.mean(0)
                V = V / np.linalg.norm(V, axis=1).mean()
            out.append(Mesh(V, F0))
            if verbose:
                print(f"  {root.name}/{d.name}  V={len(V)}", flush=True)
    if not out:
        raise FileNotFoundError(
            f"no {expression}.obj under {[str(r) for r in roots]}/*/models_reg/")
    if verbose and skipped:
        print(f"  ({skipped} subject(s) had no {expression}.obj)", flush=True)
    return out


def procrustes_align(X: np.ndarray, iters: int = 3) -> np.ndarray:
    """Rotate every head onto the iterated mean shape. ``X`` is ``(N, n, 3)``.

    Rotation only — translation and scale are already fixed by ``normalise``, and
    letting Procrustes rescale here would re-introduce a size gauge the basis
    then has to spend components on.

    Iterated because the mean of unaligned shapes is a poor target; three passes
    is well past convergence on FaceScape.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(1, keepdims=True)
    mu = Xc[0]
    for _ in range(iters):
        out = np.empty_like(Xc)
        for i, x in enumerate(Xc):
            U, _, Vt = np.linalg.svd(x.T @ mu)
            R = U @ Vt
            if np.linalg.det(R) < 0:            # reflection is not a pose
                Vt[-1] *= -1
                R = U @ Vt
            out[i] = x @ R
        Xc, mu = out, out.mean(0)
    return Xc


def laplacian_smooth(V: np.ndarray, F: np.ndarray, t: float = 1e-3) -> np.ndarray:
    """One implicit (backward-Euler) mean-curvature smoothing step.

    Solves ``(M + tL) V' = M V`` — unconditionally stable, so a single solve
    reaches any smoothing scale, and ``t`` sets that scale directly. At
    ``t=1e-3`` a normalised FaceScape head moves ~0.98% (about 2 mm), which is
    where the scan noise lives and well below any real feature.

    Explicit smoothing (``V += tLV``) would need many tiny steps for the same
    scale and shrinks the shape as it goes.
    """
    L, M = cotan_laplacian(V, F)
    solve = spla.factorized((M + t * L).tocsc())
    return np.stack([solve(M @ V[:, k]) for k in range(3)], axis=1)


def build_dataset(roots, expression: str = "1_neutral", t_smooth: float = 1e-3,
                  align: bool = True, limit: int | None = None,
                  verbose: bool = False):
    """``roots -> (X (N,n,3), F)``: loaded, aligned, smoothed heads.

    ``t_smooth=0`` skips smoothing, which is measurably the wrong choice — see
    the module docstring — but is left reachable so the difference stays testable.
    """
    meshes = load_heads(roots, expression=expression, limit=limit, verbose=verbose)
    F = meshes[0].faces
    X = np.stack([m.vertices for m in meshes])
    if verbose:
        print(f"loaded {len(X)} heads, {X.shape[1]} verts", flush=True)
    if align:
        X = procrustes_align(X)
    if t_smooth > 0:
        X = np.stack([laplacian_smooth(x, F, t_smooth) for x in X])
    return X, F
