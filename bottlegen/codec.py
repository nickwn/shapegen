"""The ρ codec: mesh ↔ band-limited spherical-harmonic coefficients of ρ.

One import for everything downstream. A generative model over ρ̂ should need this
module and nothing else — the Dirac operator, its adjoint and the optimiser stay
behind it.

    sphere = base_sphere(mesh)                  # cMCF + Möbius gauge fix
    res    = encode(mesh, L=12, sphere=sphere)  # mesh -> rho_hat
    X      = decode(res.coeffs, sphere, mesh.faces)
    err    = shape_error(mesh.vertices, X)

**ρ̂ is defined relative to a base sphere**, so the sphere travels with the
coefficients — decoding against a different parameterisation of the same shape
gives a different mesh. :class:`EncodeResult` carries it for that reason.

What the codec does *not* fix: the decode determines an immersion only up to a
similarity, and Möbius normalisation pins everything except **rotation**. Two
shapes differing by a rotation encode to entirely different ρ̂. Settle that
before training anything on these coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import conformal as _cf
from .geom import encode as _enc

__all__ = ["EncodeResult", "template_sphere", "base_sphere", "encode",
           "decode", "shape_error", "n_coeffs", "band_index", "encode_dataset"]

n_coeffs = _enc.n_coeffs
band_index = _enc.band_index


@dataclass
class EncodeResult:
    """An encoded shape, plus what you need to judge whether to trust it.

    ``niter`` and ``gnorm`` are not diagnostics to ignore: a run that stopped at
    the iteration cap with a gradient still well above zero is an **upper bound**
    on the error, not a converged encode. :meth:`converged` is the check.
    """
    coeffs: np.ndarray
    sphere: np.ndarray
    L: int
    err: float
    err0: float
    niter: int
    nfev: int
    gnorm: float
    status: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """False when the optimiser ran out of budget rather than converging.

        A ``direct`` encode is not an optimisation, so it reports True — there was
        no budget to exhaust. Read it as "this is the answer this method gives",
        not as "this is optimal".
        """
        return "LIMIT" not in self.status

    def __repr__(self):
        flag = "" if self.converged else "  [ITERATION-CAPPED]"
        return (f"EncodeResult(L={self.L}, err={self.err:.5f}, "
                f"niter={self.niter}, |g|={self.gnorm:.1e}){flag}")


def template_sphere(subdiv: int = 3) -> np.ndarray:
    """The canonical base sphere: ONE sphere shared by an entire dataset.

    Use this, not :func:`base_sphere`, whenever codes must be comparable — which
    is always, for generative modelling. The reason is not convenience:

    ρ̂ is defined *relative to* a base sphere. ``ρ = H·e^u − 1`` where ``e^u`` is
    the conformal factor of the map from that sphere, so changing the sphere
    changes ρ even for an identical surface. Codes encoded against per-shape cMCF
    spheres therefore do not share a meaning, and decoding one against a different
    sphere gives a measurably different shape (chamfer 0.047 vs 0.007). At
    generation time there is no mesh, hence no cMCF sphere to recover — so the
    sphere must be fixed in advance, and encoding must use the same one.

    The cost is a reconstruction floor of roughly 4.4% instead of cMCF's 1.3%,
    flat in resolution because it is *correspondence* error, not discretisation:
    the identity map from template to shape carries ~6.6% quasi-conformal shear
    that the spin transform structurally cannot express.
    """
    from .geom.mesh import icosphere  # noqa: PLC0415
    V = icosphere(subdiv).vertices
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def base_sphere(mesh, iters: int = 300) -> np.ndarray:
    """Sphere parameterisation of ``mesh`` on its own connectivity, gauge-fixed.

    Conformalized mean-curvature flow, then Möbius-normalised against the mesh's
    own conformal density — which kills the ℓ=1 freedom and leaves only rotation.

    A ``method="willmore"`` alternative (conformal Willmore flow via the spin
    transform) was removed. It was selected on quasi-conformal distortion, where
    it wins narrowly on a bottle at subdiv 3 (0.0170 against cMCF's 0.0194) — but
    qc turns out not to predict reconstruction accuracy at all (see
    :func:`~bottlegen.geom.spin.qc_distortion`), so the gap bought nothing, and
    reaching it made this package import from ``scripts/``. The flow itself is
    still there as ``scripts/willmore_flow.py`` if you want to experiment.
    """
    return _cf.mobius_normalize(
        mesh, _cf.spherical_conformal_map(mesh, iters=iters))


def encode(mesh, L: int, sphere=None, maxiter: int = 200,
           method: str = "direct", **kw) -> EncodeResult:
    """``mesh → ρ̂ ∈ ℝ^((L+1)²)``.

    Two encoders:

    * ``"direct"`` (default) — ``ρ = H·e^u − 1`` read straight off the geometry,
      with ``e^u`` from :func:`~bottlegen.geom.conformal.u_edge_lsq`. Closed-form,
      so nothing to converge and no iteration cap to police (``niter`` is 0 and
      means it). A measurement of a continuum quantity, so it converges under
      refinement and stays **comparable across meshes** (per-band coefficients
      differ 7.7% at L=4, 12.4% at L=12 between two triangulations of one
      surface). Its accuracy is **not monotone in L** — see below.
    * ``"adjoint"`` — optimises decoded shape error through the adjoint. Several
      times more accurate on a fixed mesh (0.0127 against 0.0644 on a bottle at
      L=12) and ~26x the cost: 19 s/mesh at subdiv 3, and **321-376 s** at 26k
      vertices. Its fit absorbs the discretisation it was fitted against, so its
      codes are *not* comparable across meshes — the same 7.4% at L=4 becomes
      **46.0%** at L=12.

    **Why ``direct`` is the default.** Accuracy on one fixed mesh is the wrong
    axis for the thing this codec exists to feed. A dataset encoder has to be
    affordable (the adjoint puts ~9000 meshes at two to three months of compute)
    and its codes have to mean the same thing across meshes, which the adjoint's
    measurably do not. Reach for ``adjoint`` when you are reconstructing one
    known mesh as accurately as possible.

    The trade is real and it is not small: on a smoothed FaceScape head ``direct``
    sits at ~0.084 *regardless of L* while the adjoint reaches 0.021. On faces
    specifically, ρ_direct comes out 10-30x too small — ``e^u`` exponentiates the
    error in ``u`` — and that is worth fixing before leaning on this default for
    scanned data.

    **For ``direct``, ``L`` is a regularisation parameter, not a fidelity knob** —
    and that is a fact about *this estimator*, not about ρ̂. With ``adjoint`` the
    error falls **monotonically** with bandwidth, as one would expect: on a
    FaceScape head against a shared template, 0.056 (L=8) → 0.043 (L=12) → 0.035
    (L=16), against ``direct``'s 0.092 / 0.090 / 0.088 on the same heads. Do not
    read the paragraph below as a limit of the representation; it is a limit of a
    biased local measurement.
    Measured on a bottle at subdiv 3: 0.115 (L=4), **0.054 (L=8)**, 0.064 (L=12),
    0.062 (L=16), 0.0641 (L=24) — and decoding ρ_direct with *no band limit at
    all* gives 0.0641 to five digits. So nothing runs away; the curve converges to
    the error of reproducing ρ_direct faithfully, and **the low-L values are
    anomalously good rather than the high-L ones anomalously bad**. Band-limiting
    is accidentally truncating the estimator's bias, which lives in the high
    bands: against the shape-optimal ρ, per-band error is 0.15-0.33x for ℓ ≤ 6 and
    1.3-11x for ℓ ≥ 8. γ says the same thing independently, rising monotonically
    with bandwidth (1.5e-3 at L=4 to 3.7e-2 unlimited, at subdiv 4) — each extra
    band makes ρ *less* realisable as any surface's curvature half-density.

    Consequence for ``direct``: **choose L by validation, and re-choose it per
    resolution.** The optimum moves — L=8 at subdiv 3-4, L=12 at subdiv 5 —
    because it tracks the discretisation floor, not the shape. If you want
    accuracy to keep improving with bandwidth, use ``adjoint``; that is the axis
    it wins on, and it is worth 2.5x at L=16 on a face.

    A third encoder, ``"lambda"`` (read λ off the correspondence, least-squares ρ
    against the Dirac energy), was removed: ``direct`` beat it by 20-35% at every
    bandwidth and resolution tried, at the same cost.

    A base sphere is computed if none is given; pass one when encoding the same
    mesh at several bandwidths, since the parameterisation is the expensive step.

    Optimises *at* bandwidth ``L`` rather than encoding then truncating — the
    decode is nonlinear, so the band-limited optimum is not the projection of the
    unconstrained one, and the difference grows as L falls.
    """
    if method not in ("direct", "adjoint"):
        raise ValueError(f"method must be 'direct' or 'adjoint', got {method!r}")
    o = _enc.encode_sh(mesh, L, sph=sphere, maxiter=maxiter,
                       optimise=(method == "adjoint"), **kw)
    return EncodeResult(coeffs=o["coeffs"], sphere=o["sph"], L=L, err=o["err"],
                        err0=o["err0"], niter=o["niter"], nfev=o["nfev"],
                        gnorm=o["gnorm"], status=o.get("status", ""),
                        meta={"alpha": o.get("alpha")})


def decode(coeffs: np.ndarray, sphere: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``ρ̂ → vertex positions``, on ``faces``.

    Determined only up to a similarity, so compare with :func:`shape_error` and
    never with a raw vertex distance.
    """
    return _enc.decode_sh(np.asarray(coeffs, float), sphere, faces)


def shape_error(target: np.ndarray, X: np.ndarray) -> float:
    """Relative Procrustes error — the quantity :func:`encode` minimises.

    Invariant to rotation, translation and scale, which is required rather than
    convenient: the decode fixes the immersion only up to a similarity, so a
    metric that could see the difference would be measuring a gauge.
    """
    return float(np.sqrt(_enc.procrustes(target, X)[0]))


def encode_dataset(meshes, L: int, method: str = "direct", sphere=None,
                   maxiter: int = 200, n_jobs: int = 1, verbose: bool = False):
    """Encode many meshes into a comparable code set.

    Returns ``(coeffs (N,K), results, decode_sphere)``. **The decode sphere is
    returned separately and deliberately**: the two methods mean different things
    by "the sphere", and conflating them is silent rather than loud.

    **``sphere`` decides the sphere policy, not ``method``.** They used to be
    coupled — ``adjoint`` implied a shared sphere and ``direct`` a per-mesh one,
    with a shared sphere *rejected* for ``direct``. That guard blocked the one
    operation a generative pipeline needs most: measure a whole dataset against a
    single template, cheaply. It was also too strong on its own evidence, since a
    direct code decodes on a sphere it never saw at 0.5–0.9x the error it gets on
    its own — portable, not meaningless.

    ``sphere=<array>``   every mesh is encoded against that one sphere, whichever
                         method. **This is what generation requires**: at sample
                         time there is no mesh, so no cMCF sphere to recover.
    ``sphere=None``      ``adjoint`` picks :func:`template_sphere` by vertex
                         count; ``direct`` measures each mesh on its own cMCF
                         sphere and returns a canonical decode sphere separately.
                         Möbius normalisation (inside :func:`base_sphere`) is what
                         makes those per-shape spheres mutually consistent, and it
                         is not optional in that mode.

    Encoding is embarrassingly parallel across meshes, so ``n_jobs > 1`` spreads
    it over a **process** pool — NumPy/SciPy release work to BLAS but the Python
    driving loop does not, so threads would not help. ``n_jobs=-1`` uses every
    core. Worth it for the adjoint (tens of seconds per mesh); for ``direct`` the
    per-mesh work is under a second and process startup can dominate, so measure
    before assuming. Measured on 8 bottles at subdiv 3: 13.3 s serial, 4.4 s at
    ``n_jobs=4``.

    Parallel and serial codes agree to ~1e-10 relative, **not bit-for-bit** — but
    neither do two serial runs: ARPACK's iterative eigensolve varies at 6e-11
    between invocations, so the adjoint encoder was never bit-reproducible and the
    pool adds nothing on top of that. ``direct`` is exactly identical either way.
    """
    if method not in ("direct", "adjoint"):
        raise ValueError(f"method must be 'direct' or 'adjoint', got {method!r}")
    meshes = list(meshes)
    n = len(meshes[0].vertices)

    if sphere is None and method == "adjoint":
        # The adjoint has no per-mesh mode: its fit absorbs whatever sphere it
        # was given, so that sphere has to be the same one every time.
        for sub in range(1, 8):                       # match the vertex count
            t = template_sphere(sub)
            if len(t) == n:
                sphere = t
                break
        else:
            raise ValueError(
                f"no icosphere has {n} vertices; pass `sphere` explicitly "
                f"(for registered data, the cMCF sphere of the dataset's own "
                f"template mesh is the natural choice), or remesh the inputs")

    if sphere is not None:
        per_mesh_sphere, decode_sphere = sphere, sphere
    else:
        per_mesh_sphere = None                        # direct, per-mesh cMCF
        decode_sphere = _canonical_for(meshes[0])

    if n_jobs == 1:
        out = []
        for i, m in enumerate(meshes):
            r = encode(m, L, sphere=per_mesh_sphere, maxiter=maxiter, method=method)
            if verbose:
                print(f"[{i + 1}/{len(meshes)}] {r}", flush=True)
            out.append(r)
    else:
        import concurrent.futures as _cf  # noqa: PLC0415
        import os  # noqa: PLC0415
        workers = os.cpu_count() if n_jobs < 0 else n_jobs
        args = [(m.vertices, m.faces, L, per_mesh_sphere, maxiter, method)
                for m in meshes]
        # Results are collected by index rather than as they complete: the codes
        # must stay aligned with the input order or the whole set is scrambled in
        # a way nothing downstream would notice.
        with _cf.ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_encode_one, args))
        if verbose:
            for i, r in enumerate(out):
                print(f"[{i + 1}/{len(meshes)}] {r}", flush=True)
    return np.stack([r.coeffs for r in out]), out, decode_sphere


def _encode_one(args):
    """Module-level so it pickles; ``ProcessPoolExecutor`` cannot ship a closure."""
    from .geom.mesh import Mesh  # noqa: PLC0415
    V, F, L, sphere, maxiter, method = args
    return encode(Mesh(V, F), L, sphere=sphere, maxiter=maxiter, method=method)


def _canonical_for(mesh) -> np.ndarray:
    """A decode sphere with this mesh's connectivity.

    An icosphere when the vertex count matches; otherwise the cMCF sphere of the
    mesh itself, which is a perfectly good canonical choice — the codec needs one
    *fixed* genus-0 sphere-topology mesh, not a uniform grid. For a registered
    dataset the natural choice is the dataset's own template mesh, and a
    category-adapted sphere measured ~2.8x better than a uniform icosphere.
    """
    n = len(mesh.vertices)
    for sub in range(1, 8):
        t = template_sphere(sub)
        if len(t) == n:
            return t
    return base_sphere(mesh)
