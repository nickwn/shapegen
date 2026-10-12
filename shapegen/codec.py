"""The ρ codec: mesh ↔ band-limited spherical-harmonic coefficients of ρ.

One import for everything downstream::

    res = encode(mesh, L=32)                          # method="polyhedral" (default)
    X   = decode(res.coeffs, res.sphere)              # positions at res.sphere
    err = shape_error(mesh.vertices, X)               # == res.err

**ρ̂ is defined relative to a sphere parameterisation** of the shape, which
:class:`EncodeResult` carries. The default method computes the polyhedron's own
conformal sphere map; ``quotient`` uses a given sphere or the input's
eigenbasis sphere (:func:`base_sphere`). The decoder is mesh-free: the
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

METHODS = ("quotient", "polyhedral")


@dataclass
class EncodeResult:
    """An encoded shape, plus what you need to judge whether to trust it.

    ``err`` is what :func:`decode` gives back (relative Procrustes). ``gamma`` is
    the Dirac residual of the code: ``γ = 0`` means some immersion has exactly
    this ρ. ``quotient`` and ``polyhedral`` codes are closed (``ℓ = 0`` set so
    γ ~1e-15, see :func:`shapegen.geom.dirac_quotient.close_code`).
    """
    coeffs: np.ndarray
    sphere: np.ndarray
    L: int
    err: float
    gamma: float = float("nan")
    status: str = ""
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return f"EncodeResult(L={self.L}, err={self.err:.5f}, gamma={self.gamma:.2e})"


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
    conformal than the cMCF it replaced on scanned heads -- worth ~3x in encoder
    accuracy, and 9x faster in the FaceScape dataset build at equal fidelity."""
    from .geom.eigen_sphere import conformal_sphere
    return conformal_sphere(mesh.vertices, mesh.faces)


REFINE_BELOW = 10000      # faces: coarser inputs are remeshed before the sphere map
REFINE_TO = 12000


def encode(mesh, L: int, sphere=None, method: str = "polyhedral", refine="auto") -> EncodeResult:
    """``mesh → ρ̂ ∈ ℝ^((L+1)²)``. The default ``"polyhedral"`` computes its own sphere
    map (``sphere`` must be ``None``); ``"quotient"`` encodes relative to
    ``sphere`` (default :func:`base_sphere`, per mesh).

    **Coarse meshes** (``quotient`` only; ``refine="auto"``: under
    ``REFINE_BELOW`` faces, and no ``sphere`` given) are first remeshed isotropically to ~``REFINE_TO`` faces
    (:func:`shapegen.geom.mesh.remesh_isotropic`, the same surface), because the
    sphere map needs a fine, well-shaped mesh. The code then encodes the
    *surface*, not its tessellation: a 12-triangle cube and a 12288-triangle one
    give the same decoded cube. ``res.sphere`` is still per input vertex (carried
    over from the remeshed map), so ``decode(res.coeffs, res.sphere)`` works as
    for any mesh. ``res.err`` is measured there. ``res.meta["err_surface"]`` is the
    error over the whole remeshed surface, the meaningful number when the input
    has only a few vertices. ``refine=True`` forces the remesh (use it for badly
    shaped inputs: slivers degrade the sphere map), ``False`` disables it.

    * ``"quotient"``: the Dirac equation read pointwise off the
      target's spinor, then corrected onto the decoder's truncated equation by
      three matrix-free passes (:func:`shapegen.geom.dirac_quotient.encode_quotient`).
      FaceScape heads at L=32: ~0.0077 / 0.0065 (whole / face). It encodes in ~80 ms; the closing solve and the scoring here add
      ~0.65 s, so call ``encode_quotient(..., close=False)`` for bulk encoding.
      Weaker on smooth shapes at low L (see its module). Deterministic.
    * ``"polyhedral"`` (default): reads the mesh literally as a polyhedron (closed,
      genus 0) and computes its conformal sphere map and spinor
      (:func:`shapegen.geom.chord_map.encode_chord`). There is no remesh and
      no eigen sphere, so ``sphere`` and ``refine`` do not apply. The mesh is used
      as given (no remesh, no decimation), except that planar regions are
      retriangulated from their points and non-cone vertices in them dropped, so
      the code depends only on the polyhedron (a cube's 12-triangle, flipped
      192-triangle and flat-subdivided meshes: identical codes). The discrete
      conformal map on the
      mesh's triangulation corrected by edge periods (exact ones where the local
      estimate moves the map measurably: coarse meshes), and a piecewise-Möbius
      spinor with curved-edge corrections. On coarse meshes it is within −5 … +2 %
      of the exact encoder it replaced (removed 2026-10-11; L=16: cube, demo
      bottle and ellipsoid at 162–2562 vertices; a spiky 9-vertex hull +13 %) in
      0.3–2 s. Raw 26k-vertex FaceScape scans (capped, unsmoothed; 3 heads) encode
      in 6.4–6.8 s (L=16) / 7.6–8.1 s (L=32) at vertex error 0.075–0.084 / 0.059–0.069,
      within ±1.3 % of the two-path encoder before it (3.2–4.3 s); a smoothed
      40962-vertex icosphere-6 head in 7–8 s.
      Codes use a cone-centroid Möbius gauge and a rotation fixed by the cones, so
      they are not interchangeable with the other methods'.
      ``res.meta["err_surface"]`` scores the decode over the whole surface.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "polyhedral":
        if sphere is not None:
            raise ValueError("method='polyhedral' (the default) computes its own sphere map; to encode "
                             "against a given sphere pass method='quotient'")
        return _encode_polyhedral(mesh, L)
    coarse = len(mesh.faces) < REFINE_BELOW if refine == "auto" else bool(refine)
    if sphere is not None or not coarse:
        return _encode_on(mesh, L, sphere, method)
    from .geom.mesh import Mesh, remesh_isotropic
    from .geom.resample import locate_on_mesh

    work = Mesh(*remesh_isotropic(mesh.vertices, mesh.faces, REFINE_TO))
    res = _encode_on(work, L, None, method)
    # the input's vertices on the remeshed surface → their sphere positions
    idx, bary, _ = locate_on_mesh(work.vertices, work.faces, mesh.vertices)
    sph = (res.sphere[work.faces[idx]] * bary[:, :, None]).sum(1)
    sph /= np.linalg.norm(sph, axis=1, keepdims=True)
    res.meta.update(err_surface=res.err, refined_mesh=work, refined_sphere=res.sphere)
    res.sphere = sph
    res.err = shape_error(mesh.vertices, decode(res.coeffs, sph)) if len(sph) >= 4 else float("nan")
    return res


def _encode_on(mesh, L, sphere, method) -> EncodeResult:
    """:func:`encode` on a mesh fine enough to map to the sphere as it is."""
    sph = base_sphere(mesh) if sphere is None else np.asarray(sphere, float)
    F = mesh.faces
    from .geom.dirac_quotient import encode_quotient

    lift, Ls = _spectral_context(L)
    o = encode_quotient(mesh.vertices, sph, F, L, lift=lift, Ls=Ls)
    # positions enter only here, to score the decode (from the closing solve's own
    # spinor: no second solve)
    err = shape_error(mesh.vertices, _eval_sh(sph, o["f_hat"], lift.Lf))
    return EncodeResult(coeffs=o["coeffs"], sphere=sph, L=L, err=err, gamma=o["gamma"],
                        status="quotient: pointwise Dirac quotient, not optimised")


def _encode_polyhedral(mesh, L) -> EncodeResult:
    """``method="polyhedral"``: the sphere map of the polyhedron itself."""
    from .geom.chord_map import encode_chord

    lift, Ls = _spectral_context(L)
    o = encode_chord(mesh.vertices, mesh.faces, L, lift=lift, Ls=Ls)
    sph = o["sphere"]
    V = np.asarray(mesh.vertices, float)
    # dense score: the decode at the quadrature samples against their exact surface points
    xs, wA, Xs = o["samples"]
    sel = np.flatnonzero(wA > 0)
    sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), 6000)).astype(int)]
    err = shape_error(V, _eval_sh(sph, o["f_hat"], lift.Lf)) if len(V) >= 4 else float("nan")
    err_surface = shape_error(Xs[sel], _eval_sh(xs[sel], o["f_hat"], lift.Lf))
    info = o["info"]
    return EncodeResult(coeffs=o["coeffs"], sphere=sph, L=L, err=err, gamma=o["gamma"], status="polyhedral: chord sphere map, not optimised",
                        meta={"err_surface": err_surface, "gauge": "cones", **info})


_SPECTRAL_CONTEXT: dict = {}


def _spectral_context(L):
    """The decoder's spinor band ``Ls`` for ``L`` and its lift (which depends only on
    it): built once per band and reused across a dataset."""
    from .geom import dirac_spectral as ds
    from .quad.lift import SpectralLift

    if L not in _SPECTRAL_CONTEXT:
        Ls = ds.spinor_band(L, ds.potential_band_extra(L))
        _SPECTRAL_CONTEXT.clear()
        _SPECTRAL_CONTEXT[L] = (SpectralLift(Ls), Ls)
    return _SPECTRAL_CONTEXT[L]


def decode(coeffs: np.ndarray, sphere: np.ndarray, faces=None) -> np.ndarray:
    """``ρ̂ → positions`` at the unit directions ``sphere`` ``(n, 3)``: the
    spectral spinor and its surface solved in spherical harmonics
    (:func:`shapegen.quad.lift.lift_spectral`). No mesh is involved; ``faces`` is
    accepted and ignored. Defined up to a similarity -- use :func:`shape_error`."""
    from .quad.lift import lift_spectral

    coeffs = np.asarray(coeffs, float)
    L = int(round(np.sqrt(len(coeffs)))) - 1
    return lift_spectral(sphere, coeffs, L)[0]


def _eval_sh(P, f_hat, L):
    """Real-SH positions ``f_hat`` ``(K, 3)`` evaluated at unit directions ``P``
    (:func:`~shapegen.geom.encode.eval_real_sh`: no band-79 basis is built)."""
    return _enc.eval_real_sh(P, f_hat, L)


def shape_error(target: np.ndarray, X: np.ndarray) -> float:
    """Relative Procrustes error, invariant to rotation, translation and scale."""
    return float(np.sqrt(_enc.procrustes(target, X)))


def encode_dataset(meshes, L: int, method: str = "polyhedral", sphere=None,
                   n_jobs: int = 1, verbose: bool = False):
    """Encode many meshes. Returns ``(coeffs (N, K), results)``.

    ``sphere=None``: each mesh on its own sphere map (``polyhedral``'s, or
    :func:`base_sphere` for ``quotient``). ``sphere=<array>`` (with
    ``quotient``): every mesh against that one sphere (meshes must
    share its vertex count). ``n_jobs > 1`` uses a process pool; results stay in
    input order.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    meshes = list(meshes)
    if n_jobs == 1:
        out = [encode(m, L, sphere=sphere, method=method) for m in meshes]
    else:
        import concurrent.futures as _cf
        import os
        workers = os.cpu_count() if n_jobs < 0 else n_jobs
        args = [(m.vertices, m.faces, L, sphere, method) for m in meshes]
        with _cf.ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_encode_one, args))
    if verbose:
        for i, r in enumerate(out):
            print(f"[{i + 1}/{len(meshes)}] {r}", flush=True)
    return np.stack([r.coeffs for r in out]), out


def _encode_one(args):
    """Module-level so it pickles for ``ProcessPoolExecutor``."""
    from .geom.mesh import Mesh
    V, F, L, sphere, method = args
    return encode(Mesh(V, F), L, sphere=sphere, method=method)
