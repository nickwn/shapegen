"""Smoothed-head dataset from FaceScape ``models_reg``.

The meshes are **full heads**, not face masks: bbox 209 x 275 x 209 mm, with
cranium, ears and a neck stub; 26317 vertices, 52261 faces, four boundary loops
(neck 96, mouth 96, eye sockets 91/94); connectivity identical across every
downloaded root.

Both representations are served, and they want different preprocessing:

===============  ==================  ==================================
stage            positions (PCA)     ρ̂ (spin codec)
===============  ==================  ==================================
``cap``          off                 **on** -- the sphere map and the ρ
                                     codec are defined for closed genus-0
                                     surfaces only
``align``        on (worth 5-7%)     **off** -- ρ is exactly
                                     rotation-invariant on a shared
                                     template (measured 4e-15)
``smooth``       on                  **on**, t=1e-4
===============  ==================  ==================================

The caps are invented geometry (the neck opening is a 168 mm hole closed by one
centroid fan), which is why they are off unless ρ needs them.

Stages, in order:

    align      Procrustes to the iterated mean. 13.5% of the raw spread is pose.
    smooth     implicit Laplacian ``(M + tL)V' = M V``. Scan noise is
               uncorrelated across subjects, so no basis can represent it: raw
               PCA plateaus at ~0.0106 (k=64 -> 128 barely moves), smoothed PCA
               keeps falling (0.0038 at k=128; scored against the smoothed test
               face, so directional only). For ρ it matters more, because H is a
               second derivative and amplifies noise by ~ℓ². The amount: at
               ``remesh=0.8``, L=32, 6 heads, face-region chamfer against the raw
               scan, **t=1e-4** is the optimum; t=0 costs ~30% on the face and
               1e-3 ~10% (measured with the since-removed residual encoder and
               mesh decoder; not re-measured on the current codec).
    resample   optional, see :func:`build_dataset`.
    encode     see :mod:`shapegen.models.basis`.

**Trap:** most of those vertices are bald cranium and neck, nearly featureless
and nearly identical between subjects, so a whole-head mean error flatters any
model. Score the face region separately (see :func:`build_dataset`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from ..geom.mesh import Mesh
from ..geom.operators import cotan_laplacian
from .registered import apply_caps, boundary_loops, fill_holes, read_obj

__all__ = ["load_heads", "procrustes_align", "laplacian_smooth",
           "isotropic_remesh", "build_dataset"]


def load_heads(roots, expressions="1_neutral", limit: int | None = None,
               normalise: bool = True, cap: bool = False, verbose: bool = False):
    """Load FaceScape ``models_reg`` heads from one or several dataset roots.

    ``roots`` is a path or an iterable of paths — FaceScape ships its ~450
    subjects split across several ``facescape_trainset_*`` directories, and they
    share one connectivity, so they concatenate into a single registered set.
    That is *checked* here rather than assumed: a mismatch would silently make
    the codes mean different things per subject.

    ``expressions`` is one name or a sequence of them; each (subject, expression)
    pair becomes one sample. FaceScape ships **20 expressions per subject**, so
    446 x 20 is ~8 900 meshes against 446 — by far the largest data lever
    available, and the difference between a code space this dataset can pin and
    one it cannot. Two caveats, both real:

    * expressions of one subject are strongly correlated, so the *effective*
      sample size is well below the count;
    * the mouth cap is invented geometry whose boundary moves with expression, so
      on a multi-expression set it enters the codes looking like measured shape.
      Consider excluding the extreme mouth expressions.

    Subjects missing an expression are skipped (446 of 447 have ``1_neutral``).

    ``normalise`` centres each head and scales it to unit mean radius. Do not
    read that as alignment — it fixes translation and scale only, and ~13.5% of
    the remaining spread is head pose. Run :func:`procrustes_align` for that.

    ``cap`` closes the four boundary loops with centroid fans. **Required for the
    ρ codec** (the sphere map and ρ are defined for closed genus-0 surfaces
    only), pointless for a position model, hence off by default. The cap face
    list is computed once from the first mesh and replayed on every other, so the
    set stays registered; only the four centroids move.
    """
    roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(r) for r in roots]
    exprs = [expressions] if isinstance(expressions, str) else list(expressions)
    out, F0, skipped = [], None, 0
    loops = added = None
    for root in roots:
        subs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: (len(d.name), d.name))
        for d in subs:
            for expr in exprs:
                if limit is not None and len(out) >= limit:
                    break
                path = d / "models_reg" / f"{expr}.obj"
                if not path.exists():
                    skipped += 1
                    continue
                V, F = read_obj(path)
                if cap:
                    if loops is None:           # cap combinatorics: compute once
                        loops = boundary_loops(F)
                        _, _, added = fill_holes(V, F, loops)
                    V, F = apply_caps(V, F, loops, added)
                if F0 is None:
                    F0 = F
                elif not np.array_equal(F, F0):
                    raise ValueError(
                        f"{path}: connectivity differs from the first mesh loaded "
                        f"— the set is not template-registered, and concatenating "
                        f"these would produce codes that mean different things")
                if normalise:
                    V = V - V.mean(0)
                    V = V / np.linalg.norm(V, axis=1).mean()
                out.append(Mesh(V, F0))
                if verbose:
                    print(f"  {root.name}/{d.name}/{expr}  V={len(V)}", flush=True)
    if not out:
        raise FileNotFoundError(
            f"no {exprs} .obj under {[str(r) for r in roots]}/*/models_reg/")
    if verbose and skipped:
        print(f"  ({skipped} subject/expression pair(s) missing)", flush=True)
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
    ``t=1e-3`` a normalised FaceScape head moves ~0.98% (about 2 mm). The dataset
    default is ``t=1e-4`` (see the module docstring); the ``1e-3`` default here
    is only the function's.
    """
    L, M = cotan_laplacian(V, F)
    solve = spla.factorized((M + t * L).tocsc())
    return np.stack([solve(M @ V[:, k]) for k in range(3)], axis=1)


def isotropic_remesh(V, F, target_pct: float = 0.8, iterations: int = 6):
    """Isotropic explicit remeshing via pymeshlab. Returns ``(V2, F2)``.

    FaceScape's registered mesh is badly shaped -- min angle **0.04°**, 24.9% of
    triangles under 20°. At ``target_pct=0.8`` the min angle rises to 0.49° and
    triangles under 20° fall to 0.5%. :func:`build_dataset` remeshes the mean
    head before building its sphere map.

    ``target_pct`` is a percentage of the bounding-box diagonal (~404 mm for a
    head); **smaller means more vertices**: 0.8 gives ~17k from 26k, 2.5 gives
    ~1.9k (10 mm edges, wider than a nostril). 0.8 is the default. Measured with
    the removed cMCF sphere and mesh decoder: remeshing at all was worth ~14%;
    2.5% -> 0.8% was worth 8% whole-head and 0% on the face; 0.4% bought nothing
    over 0.8% at 4x the vertices.

    **Breaks the registered correspondence** -- the output is a new vertex set.
    :func:`build_dataset` restores a shared topology by composing the remesh's
    barycentric weights with the icosphere resampling's.
    """
    import pymeshlab  # noqa: PLC0415
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(V, dtype=float),
                               face_matrix=np.asarray(F, dtype=np.int32)))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        targetlen=pymeshlab.PercentageValue(target_pct))
    m = ms.current_mesh()
    return (np.asarray(m.vertex_matrix(), dtype=float),
            np.asarray(m.face_matrix(), dtype=np.int64))


def build_dataset(roots, expressions="1_neutral", t_smooth: float = 1e-4,
                  align: bool = True, cap: bool = False, base: str = "native",
                  subdiv: int = 6, remesh: float | None = None,
                  limit: int | None = None, verbose: bool = False):
    """``roots -> (X (N,n,3), F)``: loaded, aligned, smoothed heads.

    ``t_smooth=0`` skips smoothing (measurably worse, see the module docstring;
    kept reachable so the difference stays testable). ``cap=True`` and
    ``align=False`` for the ρ codec; ``align=True`` is worth 5-7% for a position
    basis.

    ``base="icosphere"`` resamples every head onto a uniform icosphere at
    ``subdiv`` and returns ``(X, F, sphere)`` instead. The correspondence is the
    sphere map of the mean head, :func:`shapegen.codec.base_sphere` (the Laplace
    eigenbasis sphere, :func:`shapegen.geom.eigen_sphere.conformal_sphere`),
    computed once and then only sampled at uniform directions. That map is far
    from area-uniform (vertex-area spread ~2e7 on the remeshed mean head), and the
    ρ projection weights by sphere vertex areas, so the uniform resample is what
    makes the template usable: its base-sphere residual is 0.002 at subdiv 6,
    against 7.1 for the mean head's own sphere. ``remesh`` (a
    :func:`isotropic_remesh` target, 0.8 in the scripts) remeshes the mean head
    before building its sphere, and the two transfers are composed so every head
    still lands on one icosphere. Measured on 6 heads, the eigen sphere builds
    9x faster than the cMCF it replaced (3.7 s vs 34.7 s) at equal resampling
    fidelity (chamfer vs original 0.01847 vs 0.01851 whole-head, 0.00943 vs
    0.00947 face). ``base="native"`` (the default) keeps the source connectivity
    and leaves the caller to build a sphere.

    .. warning::

       **This function moves the reconstruction target.** ``remesh`` and
       ``base="icosphere"`` return a *resampled* head, and a coarser setting
       returns one further from the input (a 2.5% remesh sits ~35% further from
       the source head than 0.8%). Scoring a decode against that output grades
       coarser settings on a softer reference and reports gains that are not
       real. **Score against the ORIGINAL head with a topology-free metric**
       (:func:`~shapegen.eval.metrics.chamfer`), never against what this
       function returned.

       **And score the face separately.** A whole-head mean is dominated by
       cranium and neck and moves for reasons unrelated to the face. Report
       :func:`~shapegen.eval.metrics.chamfer_region` about
       :func:`~shapegen.eval.metrics.nose_tip` alongside the whole-head number.
    """
    meshes = load_heads(roots, expressions=expressions, limit=limit, cap=cap,
                        verbose=verbose)
    F = meshes[0].faces
    X = np.stack([m.vertices for m in meshes])
    if verbose:
        print(f"loaded {len(X)} heads, {X.shape[1]} verts", flush=True)
    if align:
        X = procrustes_align(X)
    if t_smooth > 0:
        X = np.stack([laplacian_smooth(x, F, t_smooth) for x in X])
    if base == "native":
        return X, F
    if base != "icosphere":
        raise ValueError(f"base must be 'native' or 'icosphere', got {base!r}")

    from .. import codec  # noqa: PLC0415
    from ..geom.mesh import icosphere  # noqa: PLC0415
    from ..geom.resample import locate_on_mesh, locate_on_sphere  # noqa: PLC0415

    ico = icosphere(subdiv)
    P = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    mean = X.mean(0)

    if remesh is None:
        # icosphere directions -> the mean head's sphere map -> the heads
        S = codec.base_sphere(Mesh(mean, F))
        idx, bary, nf = locate_on_sphere(S, F, P)
        Xr = np.stack([(x[F[idx]] * bary[:, :, None]).sum(1) for x in X])
    else:
        # Remesh the mean head, build its sphere map, then compose the two
        # transfers so every head still lands on ONE icosphere: remeshing alone
        # would break the shared topology the codes depend on.
        M, FM = isotropic_remesh(mean, F, target_pct=remesh)
        iW, W, dmax = locate_on_mesh(mean, F, M)    # M's vertices, back onto F
        S = codec.base_sphere(Mesh(M, FM))          # sphere of the REMESHED mean
        iV, V, nf = locate_on_sphere(S, FM, P)      # icosphere dirs, onto M
        if verbose:
            print(f"remeshed mean {len(mean)} -> {len(M)} verts "
                  f"(target {remesh}%), transfer error {dmax:.2e}", flush=True)
        # Compose: direction j -> 3 verts of M (weights V) -> each -> 3 verts of
        # F (weights W). Nine source vertices per direction, weights multiplied;
        # they still sum to 1: sum_a V[j,a] * sum_k W[.,k] = sum_a V[j,a] = 1.
        tri = FM[iV]                                # (m,3) vertex ids in M
        f9 = F[iW[tri]]                             # (m,3,3) vertex ids in F
        w9 = V[:, :, None] * W[tri]                 # (m,3,3)
        Xr = np.stack([(x[f9] * w9[:, :, :, None]).sum((1, 2)) for x in X])
    if verbose:
        print(f"resampled onto icosphere subdiv {subdiv}: {len(P)} verts, "
              f"{nf} fallbacks", flush=True)
    return Xr, ico.faces, P
