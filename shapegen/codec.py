"""The ρ codec: mesh ↔ band-limited spherical-harmonic coefficients of ρ.

One import for everything downstream::

    res = encode(mesh, L=32)                          # spectral adjoint (default)
    X   = decode(res.coeffs, res.sphere)              # positions at res.sphere
    err = shape_error(mesh.vertices, X)               # == res.err

**ρ̂ is defined relative to a sphere parameterisation** of the shape, which
:class:`EncodeResult` carries. The default is the input's own conformal sphere
from its Laplace eigenbasis (:func:`base_sphere`). The decoder is mesh-free: the
spectral Dirac spinor, its surface solved in spherical harmonics
(:func:`shapegen.quad.lift.lift_spectral`), read off at whatever sphere points
are asked for. The decode is defined up to a similarity, so compare with
:func:`shape_error`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import encode as _enc

__all__ = ["EncodeResult", "template_sphere", "base_sphere", "encode",
           "decode", "shape_error", "n_coeffs", "band_index", "encode_dataset",
           "METHODS"]

n_coeffs = _enc.n_coeffs
band_index = _enc.band_index

METHODS = ("spectral", "dirac", "quotient", "direct", "polyhedral")


@dataclass
class EncodeResult:
    """An encoded shape, plus what you need to judge whether to trust it.

    ``err`` is what :func:`decode` gives back (relative Procrustes). ``gamma`` is
    the Dirac residual of the code: ``γ = 0`` means some immersion has exactly
    this ρ. Spectral codes are closed (``ℓ = 0`` set so γ ~1e-15, see
    :func:`shapegen.geom.dirac_spectral.close_potential`); ``direct`` codes are
    not measured. :attr:`converged` is False when the optimiser stopped at its
    iteration cap.
    """
    coeffs: np.ndarray
    sphere: np.ndarray
    L: int
    err: float
    err0: float
    niter: int
    nfev: int
    gamma: float = float("nan")
    status: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return "LIMIT" not in self.status

    def __repr__(self):
        flag = "" if self.converged else "  [ITERATION-CAPPED]"
        return (f"EncodeResult(L={self.L}, err={self.err:.5f}, niter={self.niter}, "
                f"gamma={self.gamma:.2e}){flag}")


def template_sphere(subdiv: int = 3) -> np.ndarray:
    """A unit icosphere: one shared sphere for a registered dataset whose meshes
    all have its connectivity."""
    from .geom.mesh import icosphere
    V = icosphere(subdiv).vertices
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def base_sphere(mesh) -> np.ndarray:
    """Conformal sphere parameterisation of ``mesh`` on its own connectivity,
    Möbius-normalised against the mesh's own areas (only rotation left free):
    :func:`shapegen.geom.eigen_sphere.conformal_sphere`, the input's Laplace
    eigenbasis driven to spherical harmonics. Under a second, and 10-30x more
    conformal than the cMCF it replaced on scanned heads -- worth ~3x in
    spectral-encoder accuracy, and 9x faster in the FaceScape dataset build at
    equal fidelity."""
    from .geom.eigen_sphere import conformal_sphere
    return conformal_sphere(mesh.vertices, mesh.faces)


REFINE_BELOW = 10000      # faces: coarser inputs are remeshed before the sphere map
REFINE_TO = 12000


def encode(mesh, L: int, sphere=None, maxiter: int = 300,
           method: str = "spectral", refine="auto") -> EncodeResult:
    """``mesh → ρ̂ ∈ ℝ^((L+1)²)``, relative to ``sphere`` (default
    :func:`base_sphere`, per mesh).

    **Coarse meshes** (``refine="auto"``: under ``REFINE_BELOW`` faces, and no
    ``sphere`` given) are first remeshed isotropically to ~``REFINE_TO`` faces
    (:func:`shapegen.geom.mesh.remesh_isotropic`, the same surface), because the
    sphere map needs a fine, well-shaped mesh. The code then encodes the
    *surface*, not its tessellation: a 12-triangle cube and a 12288-triangle one
    give the same decoded cube. ``res.sphere`` is still per input vertex (carried
    over from the remeshed map), so ``decode(res.coeffs, res.sphere)`` works as
    for any mesh. ``res.err`` is measured there. ``res.meta["err_surface"]`` is the
    error over the whole remeshed surface, the meaningful number when the input
    has only a few vertices. ``refine=True`` forces the remesh (use it for badly
    shaped inputs: slivers degrade the sphere map), ``False`` disables it.

    * ``"spectral"`` (default): the adjoint encoder through the mesh-free decoder
      (:func:`shapegen.geom.spectral_adjoint.encode_spectral`) -- L-BFGS on the
      Procrustes error of the decode, started from the round sphere (ρ̂ = 0; the
      start does not change the result) and stopped once the error stalls. On
      FaceScape heads at L=32: 0.0050 / 0.0045 in ~40 evaluations (~28 s).
    * ``"dirac"``: no optimiser -- the target's own spinor, and the ρ̂ whose Dirac
      operator best annihilates it, by two linear least squares
      (:func:`shapegen.geom.dirac_direct.encode_dirac`). 0.0069 / 0.0061 in ~1.6 s,
      3-4x better than ``direct``; deterministic.
    * ``"quotient"``: the Dirac equation read pointwise off the target's spinor,
      then corrected onto the decoder's truncated equation by three
      matrix-free passes (:func:`shapegen.geom.dirac_quotient.encode_quotient`).
      At L=32 it is 1.11x off ``dirac`` on faces and ~1.0x on cubes and bottles.
      It encodes in ~80 ms, faster than ``direct``; the closing solve and the
      scoring here add ~0.65 s, so call ``encode_quotient(..., close=False)``
      for bulk encoding. Below L ≈ 16 ``dirac`` is as fast and better on
      smooth shapes (1.6x on a bottle at L=8). Deterministic.
    * ``"polyhedral"`` (only when asked for): reads the mesh literally as a
      polyhedron and computes its *exact* conformal sphere map and spinor
      (:func:`shapegen.geom.poly_conformal.encode_polyhedral`). There is no remesh and
      no eigen sphere, so ``sphere`` and ``refine`` do not apply. The code depends
      only on the polyhedron: tessellations of one cube (12 → 12288 triangles,
      flipped diagonals) agree to the quadrature error (~1e-6), against
      5e-4 – 5e-3 for the remesh path. The mesh is used as given (no remesh, no
      decimation): the start is its discrete uniformization, the fit one
      Newton–Krylov solve over the edge periods, and the spinor quadrature
      closed-form. Small and CAD-like meshes take ~0.4–6 s (a 2562-vertex
      ellipsoid: 6 s). A raw 26k-vertex scan with sliver fans (a capped
      FaceScape head) converges too, to 1e-11, but takes ~20 min, so prefer
      ``"quotient"`` (80 ms) for scans in bulk.
      Codes use a cone-centroid Möbius gauge and a rotation fixed by the cones, so
      they are not interchangeable with the other methods'.
      ``res.meta["err_surface"]`` scores the decode over the whole surface.
    * ``"direct"``: ``ρ = H·e^u − 1`` measured off the geometry and projected --
      no optimisation, well under a second; what training uses (via
      :class:`shapegen.models.basis.RhoBasis`), since it is a measurement of a
      continuum quantity and comparable across meshes. ~5x less accurate.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "polyhedral":
        if sphere is not None:
            raise ValueError("method='polyhedral' computes its own exact sphere map; do not pass sphere")
        return _encode_polyhedral(mesh, L)
    coarse = len(mesh.faces) < REFINE_BELOW if refine == "auto" else bool(refine)
    if sphere is not None or not coarse:
        return _encode_on(mesh, L, sphere, maxiter, method)
    from .geom.mesh import Mesh, remesh_isotropic
    from .geom.resample import locate_on_mesh

    work = Mesh(*remesh_isotropic(mesh.vertices, mesh.faces, REFINE_TO))
    res = _encode_on(work, L, None, maxiter, method)
    # the input's vertices on the remeshed surface → their sphere positions
    idx, bary, _ = locate_on_mesh(work.vertices, work.faces, mesh.vertices)
    sph = (res.sphere[work.faces[idx]] * bary[:, :, None]).sum(1)
    sph /= np.linalg.norm(sph, axis=1, keepdims=True)
    res.meta.update(err_surface=res.err, refined_mesh=work, refined_sphere=res.sphere)
    res.sphere = sph
    res.err = shape_error(mesh.vertices, decode(res.coeffs, sph)) if len(sph) >= 4 else float("nan")
    return res


def _encode_on(mesh, L, sphere, maxiter, method) -> EncodeResult:
    """:func:`encode` on a mesh fine enough to map to the sphere as it is."""
    sph = base_sphere(mesh) if sphere is None else np.asarray(sphere, float)
    F = mesh.faces
    if method == "direct":
        from .geom import spin
        from .geom.operators import vertex_areas

        c0 = _enc.sh_project(spin.mesh_to_rho_direct(mesh, sph),
                             _enc.real_sh_basis(sph, L), mass=vertex_areas(sph, F))
        err = shape_error(mesh.vertices, decode(c0, sph))
        return EncodeResult(coeffs=c0, sphere=sph, L=L, err=err, err0=err,
                            niter=0, nfev=0, status="direct: measured, not optimised")
    lift, spinor, Ls = _spectral_context(L)
    if method == "dirac":
        from .geom.dirac_direct import encode_dirac
        o = encode_dirac(mesh.vertices, sph, F, L, lift=lift, Ls=Ls)
        # positions enter only here, to score the decode (from the closing solve's
        # own spinor: no second solve)
        err = shape_error(mesh.vertices, _eval_sh(sph, o["f_hat"], lift.Lf))
        return EncodeResult(coeffs=o["coeffs"], sphere=sph, L=L, err=err, err0=err, niter=0,
                            nfev=0, gamma=o["gamma"], status="dirac: linear solves, not optimised")
    if method == "quotient":
        from .geom.dirac_quotient import encode_quotient
        o = encode_quotient(mesh.vertices, sph, F, L, lift=lift, Ls=Ls)
        err = shape_error(mesh.vertices, _eval_sh(sph, o["f_hat"], lift.Lf))
        return EncodeResult(coeffs=o["coeffs"], sphere=sph, L=L, err=err, err0=err, niter=0,
                            nfev=0, gamma=o["gamma"],
                            status="quotient: pointwise Dirac quotient, not optimised")
    from .geom import spectral_adjoint as sa
    o = sa.encode_spectral(mesh.vertices, sph, F, L, maxiter=maxiter,
                           lift=lift, spinor=spinor, Ls=Ls)
    err = shape_error(mesh.vertices, decode(o["coeffs"], sph))
    return EncodeResult(coeffs=o["coeffs"], sphere=sph, L=L, err=err,
                        err0=o["err0"], niter=o["niter"], nfev=o["nfev"],
                        gamma=o["gamma"], status=o["status"],
                        meta={"sec_per_eval": o["sec_per_eval"], "loss_err": o["err"]})


def _encode_polyhedral(mesh, L) -> EncodeResult:
    """``method="polyhedral"``: the exact sphere map of the polyhedron itself."""
    from .geom.poly_conformal import encode_polyhedral

    lift, _spinor, Ls = _spectral_context(L)
    o = encode_polyhedral(mesh.vertices, mesh.faces, L, lift=lift, Ls=Ls)
    sph = o["sphere"]
    V = np.asarray(mesh.vertices, float)
    err = shape_error(V, _eval_sh(sph, o["f_hat"], lift.Lf)) if len(V) >= 4 else float("nan")
    # dense score: the decode at the quadrature samples against their exact surface points
    xs, wA, Xs = o["samples"]
    sel = np.flatnonzero(wA > 0)
    sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), 6000)).astype(int)]
    err_surface = shape_error(Xs[sel], _eval_sh(xs[sel], o["f_hat"], lift.Lf))
    info = o["info"]
    return EncodeResult(coeffs=o["coeffs"], sphere=sph, L=L, err=err, err0=err, niter=0, nfev=0,
                        gamma=o["gamma"], status="polyhedral: exact sphere map, not optimised",
                        meta={"err_surface": err_surface, "gauge": "cones", **info})


_SPECTRAL_CONTEXT: dict = {}


def _spectral_context(L):
    """The spectral lift and spinor precomputation depend only on ``L``: built
    once per band and reused across a dataset."""
    from .geom import dirac_spectral as ds
    from .geom import spectral_adjoint as sa

    if L not in _SPECTRAL_CONTEXT:
        Ls = ds.spinor_band(L, ds.potential_band_extra(L))
        _SPECTRAL_CONTEXT.clear()
        _SPECTRAL_CONTEXT[L] = (sa.SpectralLift(Ls), sa.SpectralSpinor(L, Ls), Ls)
    return _SPECTRAL_CONTEXT[L]


def decode(coeffs: np.ndarray, sphere: np.ndarray, faces=None) -> np.ndarray:
    """``ρ̂ → positions`` at the unit directions ``sphere`` ``(n, 3)``: the spectral
    spectral spinor and its surface solved in spherical harmonics
    (:func:`shapegen.quad.lift.lift_spectral`). No mesh is involved; ``faces`` is
    accepted and ignored. Defined up to a similarity -- use :func:`shape_error`."""
    from .quad.lift import lift_spectral

    coeffs = np.asarray(coeffs, float)
    L = int(round(np.sqrt(len(coeffs)))) - 1
    return lift_spectral(sphere, coeffs, L)[0]


def _eval_sh(P, f_hat, L, chunk: int = 4000):
    """Real-SH positions ``f_hat`` ``(K, 3)`` evaluated at unit directions ``P``,
    chunked so a band-79 basis never exists for all points at once."""
    X = np.empty((len(P), f_hat.shape[1]))
    for i in range(0, len(P), chunk):
        X[i:i + chunk] = _enc.real_sh_basis(P[i:i + chunk], L) @ f_hat
    return X


def shape_error(target: np.ndarray, X: np.ndarray) -> float:
    """Relative Procrustes error, invariant to rotation, translation and scale."""
    return float(np.sqrt(_enc.procrustes(target, X)[0]))


def encode_dataset(meshes, L: int, method: str = "spectral", sphere=None,
                   maxiter: int = 300, n_jobs: int = 1, verbose: bool = False):
    """Encode many meshes. Returns ``(coeffs (N, K), results)``.

    ``sphere=None``: each mesh on its own :func:`base_sphere` (the accuracy of
    ``spectral`` comes from each shape's own conformal map). ``sphere=<array>``:
    every mesh against that one sphere (meshes must share its vertex count).
    ``n_jobs > 1`` uses a process pool; results stay in input order.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    meshes = list(meshes)
    if n_jobs == 1:
        out = [encode(m, L, sphere=sphere, maxiter=maxiter, method=method) for m in meshes]
    else:
        import concurrent.futures as _cf
        import os
        workers = os.cpu_count() if n_jobs < 0 else n_jobs
        args = [(m.vertices, m.faces, L, sphere, maxiter, method) for m in meshes]
        with _cf.ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_encode_one, args))
    if verbose:
        for i, r in enumerate(out):
            print(f"[{i + 1}/{len(meshes)}] {r}", flush=True)
    return np.stack([r.coeffs for r in out]), out


def _encode_one(args):
    """Module-level so it pickles for ``ProcessPoolExecutor``."""
    from .geom.mesh import Mesh
    V, F, L, sphere, maxiter, method = args
    return encode(Mesh(V, F), L, sphere=sphere, maxiter=maxiter, method=method)
