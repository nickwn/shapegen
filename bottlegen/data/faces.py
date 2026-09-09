"""Smoothed-head dataset from FaceScape ``models_reg``.

Serves both representations, and they want different preprocessing:

===============  ==================  ==================================
stage            positions (PCA)     ρ̂ (spin codec)
===============  ==================  ==================================
``cap``          off                 **on** — cMCF, the Dirac eigensolve
                                     and the cotan-Poisson integration
                                     are undefined with boundary
``align``        on (worth 5-7%)     **off** — ρ is exactly
                                     rotation-invariant on a shared
                                     template, measured 4e-15, so
                                     aligning is pure cost
``smooth``       on                  **on**, and it matters more: H is a
                                     second derivative, so it amplifies
                                     scan noise by ~ℓ²
===============  ==================  ==================================

The caps are invented geometry — the neck opening alone is a 168 mm hole closed
by a single centroid fan — which is why they are off unless ρ asks for them.

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
    ρ codec** — cMCF, the Dirac eigensolve and the cotan-Poisson integration are
    all undefined with boundary — and pointless for a position model, which is
    why it is off by default. The cap face list is computed once from the first
    mesh and replayed on every other, so the set stays registered; only the four
    centroids move. Note the mouth cap is invented geometry whose boundary shifts
    with expression, so on a multi-expression set it enters the codes looking
    like measured shape.
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
    ``t=1e-3`` a normalised FaceScape head moves ~0.98% (about 2 mm), which is
    where the scan noise lives and well below any real feature.

    Explicit smoothing (``V += tLV``) would need many tiny steps for the same
    scale and shrinks the shape as it goes.
    """
    L, M = cotan_laplacian(V, F)
    solve = spla.factorized((M + t * L).tocsc())
    return np.stack([solve(M @ V[:, k]) for k in range(3)], axis=1)


def isotropic_remesh(V, F, target_pct: float = 0.8, iterations: int = 6):
    """Isotropic explicit remeshing via pymeshlab. Returns ``(V2, F2)``.

    FaceScape's registered mesh is badly shaped — min angle **0.04°** with 24.9%
    of triangles under 20° — and cMCF carries that faithfully onto the sphere.
    Remeshing attacks the source: measured at ``target_pct=0.8``, min angle rises
    to 0.49°, triangles under 20° fall to **0.5%**, and the resulting conformal map
    is far better, **qc 0.0059 against 0.03**.

    **It does not fix the sphere, and on its own it makes things worse.** A uniform
    head maps to a sphere with 1.4e8 area spread (the source mesh gave 2.1e6),
    because a conformal map cannot be uniform on both sides — that is what
    conformal area distortion *is*. Encoding on that sphere reconstructs at
    0.054-0.067 against the icosphere route's 0.034-0.037. So remeshing belongs
    *before* the resampling, never instead of it.

    ``target_pct`` is a percentage of the bounding-box diagonal; **smaller means
    more vertices** (0.8 gives ~17k from FaceScape's 26k, 2.5 gives ~4k). The
    measured optimum is **2.5**, and it is an interior one -- 8.0 is twice as bad
    (0.0415 against 0.0238) because the remesh starts discarding real geometry.
    The target also sets the best bandwidth: 2.5 peaks at L=24, 5.0 at L=32, and
    those two tie on error. See ``scripts/encode_faces.py`` for the paired numbers
    and the gamma-based rule for choosing L.

    Note this **breaks the registered correspondence** — the output is a new vertex
    set. :func:`build_dataset` restores a shared topology afterwards by composing
    the remesh's barycentric weights with the icosphere resampling's.
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


def build_dataset(roots, expressions="1_neutral", t_smooth: float = 1e-3,
                  align: bool = True, cap: bool = False, base: str = "native",
                  subdiv: int = 6, remesh: float | None = None,
                  limit: int | None = None, verbose: bool = False):
    """``roots -> (X (N,n,3), F)``: loaded, aligned, smoothed heads.

    ``t_smooth=0`` skips smoothing, which is measurably the wrong choice — see
    the module docstring — but is left reachable so the difference stays testable.

    ``cap=True`` for the ρ codec, which needs a closed surface. ``align`` is worth
    5-7% for a position basis and is **pure cost for ρ**, which is exactly
    rotation-invariant on a shared template (measured 4e-15) — so pass
    ``align=False`` there.

    ``base="icosphere"`` resamples every mesh onto a uniform icosphere at
    ``subdiv``, returning ``(X, F, sphere)`` instead of ``(X, F)``. The cMCF sphere
    of the mean shape is computed once as the correspondence and then discarded,
    so the parameterisation is unchanged — only the sampling. Worth ~27% of
    reconstruction error, because two stages of the codec divide by sphere
    triangle areas that otherwise span 1e6x. ``base="native"`` (the default) keeps
    the source connectivity and leaves the caller to build a sphere.
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
        # icosphere directions -> the mean head's own cMCF sphere -> the heads
        S = codec.base_sphere(Mesh(mean, F))
        idx, bary, nf = locate_on_sphere(S, F, P)
        Xr = np.stack([(x[F[idx]] * bary[:, :, None]).sum(1) for x in X])
    else:
        # Remesh first, for a better-conditioned cMCF and a more conformal map,
        # then compose the two transfers so every head still lands on ONE
        # icosphere. Remeshing alone would break the shared topology the codes
        # depend on; composing restores it.
        M, FM = isotropic_remesh(mean, F, target_pct=remesh)
        iW, W, dmax = locate_on_mesh(mean, F, M)    # M's vertices, back onto F
        S = codec.base_sphere(Mesh(M, FM))          # cMCF of the REMESHED mean
        iV, V, nf = locate_on_sphere(S, FM, P)      # icosphere dirs, onto M
        if verbose:
            print(f"remeshed mean {len(mean)} -> {len(M)} verts "
                  f"(target {remesh}%), transfer error {dmax:.2e}", flush=True)
        # Compose: direction -> 3 verts of M -> 9 verts of F, weights multiplied.
        # direction j -> 3 verts of M (weights V) -> each of those -> 3 verts of
        # F (weights W). Nine source vertices per direction, weights multiplied.
        # They still sum to 1: sum_a V[j,a] * sum_k W[.,k] = sum_a V[j,a] = 1.
        tri = FM[iV]                                # (m,3) vertex ids in M
        f9 = F[iW[tri]]                             # (m,3,3) vertex ids in F
        w9 = V[:, :, None] * W[tri]                 # (m,3,3)
        Xr = np.stack([(x[f9] * w9[:, :, :, None]).sum((1, 2)) for x in X])
    if verbose:
        print(f"resampled onto icosphere subdiv {subdiv}: {len(P)} verts, "
              f"{nf} fallbacks", flush=True)
    return Xr, ico.faces, P
