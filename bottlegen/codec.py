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
shapes differing by a rotation encode to entirely different ρ̂. See
``SIMPLIFICATION_PLAN.md`` §7 before training anything on these coefficients.
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

        A ``local`` encode is not an optimisation, so it reports True — there was
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


def base_sphere(mesh, method: str = "cmcf", **kw) -> np.ndarray:
    """Sphere parameterisation of ``mesh`` on its own connectivity, gauge-fixed.

    ``method="cmcf"``      conformalized mean-curvature flow.
    ``method="willmore"``  conformal Willmore flow via the spin transform.

    Both are Möbius-normalised against the mesh's own conformal density, which
    kills the ℓ=1 freedom and leaves only rotation. Measured on a bottle at
    subdiv 3, quasi-conformal distortion is 0.0194 (cmcf) and 0.0170 (willmore).

    That gap matters less than it looks: the encoder optimises decoded *shape*
    error rather than the correspondence, so the parameterisation's conformal
    quality is largely compensated — the round trip lands roughly 6× below either
    figure. Pick on cost and robustness, not distortion.
    """
    if method == "cmcf":
        return _cf.mobius_normalize(
            mesh, _cf.spherical_conformal_map(mesh, iters=kw.pop("iters", 300)))
    if method == "willmore":
        from scripts.willmore_flow import flow_trajectory  # noqa: PLC0415
        traj = flow_trajectory(mesh.vertices, mesh.faces,
                               tau=kw.pop("tau", 0.5), steps=kw.pop("steps", 30),
                               mode="gradual")
        S = traj[-1] - traj[-1].mean(0)
        return _cf.mobius_normalize(mesh, S / np.linalg.norm(S, axis=1, keepdims=True))
    raise ValueError(f"unknown method {method!r} (expected 'cmcf' or 'willmore')")


def encode(mesh, L: int, sphere=None, maxiter: int = 200,
           method: str = "adjoint", **kw) -> EncodeResult:
    """``mesh → ρ̂ ∈ ℝ^((L+1)²)``.

    ``method="adjoint"`` (default) optimises decoded shape error through the
    adjoint. ``method="local"`` returns the direct geometric measurement instead
    — roughly 2× less accurate, but far more comparable across meshes, because a
    measurement of a continuum quantity converges under refinement while a fit
    absorbs the discretisation it was fitted against.

    Same surface, two triangulations, per-band coefficient difference::

        L          local     adjoint
        4           7.7%        7.4%
        8           8.9%       28.3%
        12         12.4%       46.0%

    Use ``local`` when codes must be comparable across meshes or resolutions;
    ``adjoint`` when reconstruction on a *fixed* discretisation is what matters.

    Computes a base sphere if none is given; pass one when encoding the same mesh
    at several bandwidths, since the parameterisation is the expensive step.

    Optimises *at* bandwidth ``L`` rather than encoding then truncating — the
    decode is nonlinear, so the band-limited optimum is not the projection of the
    unconstrained one, and the difference grows as L falls.
    """
    if method not in ("adjoint", "local"):
        raise ValueError(f"method must be 'adjoint' or 'local', got {method!r}")
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


def encode_dataset(meshes, L: int, sphere=None, maxiter: int = 200,
                   n_jobs: int = 1, verbose: bool = False):
    """Encode many meshes against **one shared sphere**; returns ``(coeffs, results)``.

    ``sphere`` defaults to :func:`template_sphere` sized to the meshes. Passing
    per-shape spheres here would defeat the purpose — see that function for why
    the codes would then not share a meaning.

    Encoding is ~26 s/mesh at subdiv 3 / L=12 and is embarrassingly parallel
    across meshes. ``n_jobs > 1`` needs a process pool (the work is NumPy/SciPy
    bound, so threads will not help); left as a hook rather than a guess at your
    scheduler.
    """
    if n_jobs != 1:
        raise NotImplementedError(
            "parallel encoding is a deliberate hook, not an omission: use a "
            "process pool over `encode` and pick the executor to suit your "
            "environment.")
    meshes = list(meshes)
    if sphere is None:
        n = len(meshes[0].vertices)
        for sub in range(1, 8):                       # match the vertex count
            t = template_sphere(sub)
            if len(t) == n:
                sphere = t
                break
        else:
            raise ValueError(
                f"no icosphere has {n} vertices; pass `sphere` explicitly, or "
                f"remesh the inputs onto a template first")
    out = []
    for i, m in enumerate(meshes):
        r = encode(m, L, sphere=sphere, maxiter=maxiter)
        if verbose:
            print(f"[{i + 1}/{len(meshes)}] {r}", flush=True)
        out.append(r)
    return np.stack([r.coeffs for r in out]), out
