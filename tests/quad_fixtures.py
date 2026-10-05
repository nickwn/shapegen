"""Shared ρ fixtures for the quad-mesh port.

Not a pytest file itself — imported by ``test_quad_p0_p1.py`` and by the
``test_quad_*.py`` files P2 onward will add, so the fixtures are defined once.

``tests/data/quadmesh_ref/`` holds the bundle's own saved intermediates
(``~/Downloads/bundle/data/``, 1.1 MB) — the regression corpus P0 copies in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shapegen.geom.encode import n_coeffs, real_sh_basis
from shapegen.geom.mesh import icosphere

REF_DIR = Path(__file__).parent / "data" / "quadmesh_ref"


def load_ref(name: str) -> np.ndarray:
    """Load one saved array from the bundle's regression corpus."""
    return np.load(REF_DIR / f"{name}.npy")


def sphere_mesh(subdiv: int = 4):
    """The bundle's own mesh: ``icosphere(4)``, unit-normalised.

    Identical construction to the bundle's hand-rolled ``s1_mesh.icosphere``
    (same base 12 vertices, same face winding, same midpoint order) — verified
    to match ``tests/data/quadmesh_ref/V.npy`` / ``F.npy`` exactly (max vertex
    diff 2.2e-16, faces bit-identical). Vertices from ``geom.mesh.icosphere``
    line up 1:1 with the bundle's saved arrays; no re-indexing needed.
    """
    m = icosphere(subdiv)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    return V, m.faces


def bundle_style_coeffs(L: int = 32, lmin: int = 2, power: float = -2.0,
                         amp: float = 1.0, seed: int = 1) -> np.ndarray:
    """Real-SH coefficients reproducing the bundle's ``random_rho`` field, laid
    out in this repo's :func:`~shapegen.geom.encode.real_sh_basis` convention.

    ``s1_mesh.py``'s ``random_rho`` draws, per degree ``l``, a Gaussian ``c0``
    for ``m=0`` then a pair ``(a, b)`` per ``m=1..l`` (in that order, from one
    ``rng``), and builds::

        rho += c0 * Re(Y_l0)
        rho += sqrt2 * (a * Re(Y_lm) - b * Im(Y_lm))       for m = 1..l

    using scipy's ``sph_harm_y`` directly — no ``(-1)^m`` Condon-Shortley
    cancellation. ``real_sh_basis`` *does* carry that cancellation (the
    "one conversion pair" warning), so matching the two fields requires,
    for each ``m = 1..l``::

        coeffs[l, 0]  = c0
        coeffs[l, +m] = (-1)**m * a
        coeffs[l, -m] = -(-1)**m * b

    Verified against the saved ``rho.npy`` (L=32, seed=1) to 1.2e-15 — see
    ``test_quad_p0_p1.py::test_gate_p1_matches_bundle_rho``.
    """
    rng = np.random.default_rng(seed)
    coeffs = np.zeros(n_coeffs(L))

    def idx(l: int, m: int) -> int:
        return l * l + (m + l)

    for l in range(lmin, L + 1):
        sc = amp * (l ** power)
        c0 = rng.normal() * sc
        coeffs[idx(l, 0)] = c0
        for m in range(1, l + 1):
            a = rng.normal() * sc
            b = rng.normal() * sc
            coeffs[idx(l, m)] = (-1) ** m * a
            coeffs[idx(l, -m)] = -(-1) ** m * b
    return coeffs


def fixture_f0(subdiv: int = 4):
    """F0 — ρ ≡ 0 on ``icosphere(4)``. The identity; pins every convention."""
    V, F = sphere_mesh(subdiv)
    return V, F, np.zeros(len(V))


def fixture_f1(L: int = 32, seed: int = 1, subdiv: int = 4):
    """F1 — power-law ℓ⁻², ℓ = 2..32, seed 1. The bundle's own driver fixture;
    comparable to its saved ``rho.npy`` (Gate P1) and to every bundle-derived
    number that follows from it."""
    V, F = sphere_mesh(subdiv)
    coeffs = bundle_style_coeffs(L=L, seed=seed)
    rho = real_sh_basis(V, L) @ coeffs
    return V, F, rho, coeffs


def fixture_f2(L: int = 32, subdiv: int = 3, seed: int = 0):
    """F2 — a real code: ``codec.encode(bottle, L=32, method="spectral")``."""
    from shapegen.codec import encode
    from shapegen.data.dataset import demo_bottle_meshes

    mesh = demo_bottle_meshes(n=1, subdiv=subdiv, seed=seed)[0]
    return encode(mesh, L=L, method="spectral")


def fixture_f3(roots, L: int = 32, remesh: float = 0.8, t_smooth: float = 1e-4):
    """F3 — the hard case: a FaceScape head, ``γ_mesh ≈ 2.7e-2`` at L=32.

    Needs a local FaceScape ``models_reg`` root, so it is not exercised by the
    automated gate suite. ρ is ~99% unrepresentable on scanned faces, so
    development is driven on F1/F2; F3 is a stress test, not a target.
    """
    from shapegen.codec import encode
    from shapegen.data.faces import build_dataset
    from shapegen.geom.mesh import Mesh

    X, F = build_dataset(roots, t_smooth=t_smooth, align=False, cap=True,
                          remesh=remesh, limit=1)
    return encode(Mesh(X[0], F), L=L, method="spectral")


F4_SPECTRA = (-2.0, -1.5, -1.0, -0.5)
"""F4 — spectra ℓ⁻² … ℓ⁻⁰·⁵ for the umbilic-count sweep."""


def fixture_f4(power: float, L: int = 32, seed: int = 1, subdiv: int = 4):
    """One spectrum of F4: same construction as F1 at a different power law."""
    V, F = sphere_mesh(subdiv)
    coeffs = bundle_style_coeffs(L=L, power=power, seed=seed)
    rho = real_sh_basis(V, L) @ coeffs
    return V, F, rho, coeffs


# --------------------------------------------------------------------------- #
# Amplitude calibration: what the fixtures are worth against production        #
# --------------------------------------------------------------------------- #

FIXTURE_RMS = {1.0: 0.153, 2.0: 0.305, 4.0: 0.611,
               8.0: 1.222, 12.6: 1.924, 20.0: 3.055}
"""``rho_rms`` of ``bundle_style_coeffs(power=-2, seed=1)`` at each amplitude
multiplier, measured at L=24."""

PRODUCTION_RMS = 1.90
"""What real FaceScape codes measure (1.87-1.93 across four subjects, L=24).

**Every quality number in this repository before 2026-09-21 was taken at
``rms <= 0.61``** -- 3-12x below this. That is not a detail: it is the
regime where ``mu = 1 + 2rho`` is a valid linearisation, so the whole spectral-solve
comparison was run where the thing it replaces happens to work. The amplitude
at which it stops working (``rms ~ 1.22``, F5 below) was never tested.
"""

F5_AMPLITUDE = 12.6
"""The multiplier whose ``rho_rms`` matches :data:`PRODUCTION_RMS`."""


def fixture_f5(L: int = 32, seed: int = 1, subdiv: int = 4,
               amplitude: float = F5_AMPLITUDE):
    """F5 -- a **production-amplitude** band-limited field.

    F1-F4 are all at amplitude 1, which is 12x quieter than the data this
    pipeline ships on. F5 is F1 scaled so that :func:`shapegen.quad.field.rho_rms`
    matches :data:`PRODUCTION_RMS`, while staying band-limited -- so, unlike a
    real face, it can still be scored: :func:`shapegen.quad.field.area_distortion`
    converges in its referee subdivision here (drift 1.8-8.1%) and does not on
    a scanned face (20-35%, with the verdict changing sign).

    That combination is the point of this fixture. It is the only input in the
    corpus that is simultaneously at production amplitude *and* measurable.
    """
    V, F = sphere_mesh(subdiv)
    coeffs = bundle_style_coeffs(L=L, power=-2.0, seed=seed) * amplitude
    rho = real_sh_basis(V, L) @ coeffs
    return V, F, rho, coeffs
