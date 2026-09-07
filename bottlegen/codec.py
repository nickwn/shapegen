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
           method: str = "adjoint", **kw) -> EncodeResult:
    """``mesh → ρ̂ ∈ ℝ^((L+1)²)``.

    Two encoders:

    * ``"adjoint"`` (default) — optimises decoded shape error through the adjoint.
      The most accurate and ~26x the cost (19 s/mesh against 0.74 at subdiv 3).
      Its fit absorbs the discretisation it was fitted against, so its codes are
      *not* comparable across meshes: same surface at two triangulations, per-band
      coefficients differ 7.4% at L=4 but 46.0% at L=12.
    * ``"direct"`` — ``ρ = H·e^u − 1`` read straight off the geometry, with
      ``e^u`` from :func:`~bottlegen.geom.conformal.u_edge_lsq`. Closed-form, so
      nothing to converge and no iteration cap to police. A measurement of a
      continuum quantity, so it converges under refinement and stays comparable
      across meshes (7.7% → 12.4% over the same range). Note its accuracy is
      **not monotone in L** — it peaks around L=8-12 and degrades after, because
      extra bandwidth faithfully reproduces more of a biased local measurement.

    Choose on what has to be true: ``adjoint`` for reconstruction accuracy on a
    *fixed* discretisation, ``direct`` for speed and for codes that mean the same
    thing across meshes.

    A third encoder, ``"lambda"`` (read λ off the correspondence, least-squares ρ
    against the Dirac energy), was removed: ``direct`` beat it by 20-35% at every
    bandwidth and resolution tried, at the same cost.

    A base sphere is computed if none is given; pass one when encoding the same
    mesh at several bandwidths, since the parameterisation is the expensive step.

    Optimises *at* bandwidth ``L`` rather than encoding then truncating — the
    decode is nonlinear, so the band-limited optimum is not the projection of the
    unconstrained one, and the difference grows as L falls.
    """
    if method not in ("adjoint", "direct"):
        raise ValueError(f"method must be 'adjoint' or 'direct', got {method!r}")
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


def encode_dataset(meshes, L: int, method: str = "adjoint", sphere=None,
                   maxiter: int = 200, n_jobs: int = 1, verbose: bool = False):
    """Encode many meshes into a comparable code set.

    Returns ``(coeffs (N,K), results, decode_sphere)``. **The decode sphere is
    returned separately and deliberately**: the two methods mean different things
    by "the sphere", and conflating them is silent rather than loud.

    ``method="adjoint"`` — one **shared** sphere for every mesh, defaulting to
    :func:`template_sphere`. The fit absorbs that sphere's discretisation, which
    is harmless precisely because it never changes. Decode against the same
    sphere. More accurate (chamfer 0.020 vs 0.032 at L=16) and ~26x slower
    (19 s/mesh against 0.74).

    ``method="direct"`` — each mesh is measured on its **own** cMCF sphere, which
    is then discarded. ρ = H·e^u − 1 is a measurement of a function on S², so the
    per-shape sphere is the instrument and not part of the answer: measured, a
    direct code decodes on a sphere it never saw at 0.5–0.9x the error it gets on
    its own. Möbius normalisation (inside ``base_sphere``) is what makes the
    per-shape spheres mutually consistent and hence the codes comparable — it is
    not optional here.

    Encoding is embarrassingly parallel across meshes. ``n_jobs > 1`` needs a
    process pool (NumPy/SciPy-bound, so threads will not help); left as a hook
    rather than a guess at your scheduler.
    """
    if method not in ("adjoint", "direct"):
        raise ValueError(f"method must be 'adjoint' or 'direct', got {method!r}")
    if n_jobs != 1:
        raise NotImplementedError(
            "parallel encoding is a deliberate hook, not an omission: use a "
            "process pool over `encode` and pick the executor to suit your "
            "environment.")
    meshes = list(meshes)
    n = len(meshes[0].vertices)

    if method == "adjoint":
        if sphere is None:
            for sub in range(1, 8):                   # match the vertex count
                t = template_sphere(sub)
                if len(t) == n:
                    sphere = t
                    break
            else:
                raise ValueError(
                    f"no icosphere has {n} vertices; pass `sphere` explicitly "
                    f"(for registered data, the cMCF sphere of the dataset's own "
                    f"template mesh is the natural choice), or remesh the inputs")
        per_mesh_sphere, decode_sphere = sphere, sphere
    else:
        # Direct: each mesh gets its own cMCF sphere. Passing a shared one here
        # would make this the adjoint's initialiser rather than a measurement.
        if sphere is not None:
            raise ValueError(
                "method='direct' measures each mesh on its OWN cMCF sphere; a "
                "shared `sphere` is meaningless here. Pass the decode sphere to "
                "`decode` instead.")
        per_mesh_sphere = None
        decode_sphere = _canonical_for(meshes[0])

    out = []
    for i, m in enumerate(meshes):
        r = encode(m, L, sphere=per_mesh_sphere, maxiter=maxiter, method=method)
        if verbose:
            print(f"[{i + 1}/{len(meshes)}] {r}", flush=True)
        out.append(r)
    return np.stack([r.coeffs for r in out]), out, decode_sphere


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
