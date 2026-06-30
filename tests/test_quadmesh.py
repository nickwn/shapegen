"""Session 11 acceptance tests: connection Laplacian, cross field, stripes.

Maps to Appendix B row 14:
  14. conn-Laplacian flat → λ≈0; index = χ         machine eps

A flat (planar) surface admits a parallel field (λ≈0, |ψ| const, machine
precision); the smoothest cross field's singularity indices sum to χ=2 on a
closed genus-0 surface and align toward principal directions away from umbilics;
the stripe field gives an alias-free band pattern.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from bottlegen.decode import quadmesh as qm
from bottlegen.geom.mesh import Mesh, ellipsoid, icosphere
from bottlegen.geom.operators import vertex_areas, vertex_normals
from bottlegen.geom.transport import (connection_laplacian, frames_from_normals,
                                      transport_edges)


def planar_grid(nx=12, ny=12):
    """Flat triangulated square patch (z=0): exactly flat, frames aligned."""
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    V = np.array([[x, y, 0.0] for y in ys for x in xs])
    F = []
    for r in range(ny - 1):
        for c in range(nx - 1):
            a = r * nx + c
            F += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    return Mesh(V, np.array(F))


# --------------------------------------------------------------------------- #
# Test 14a: flat surface -> parallel field, λ≈0, |ψ| const (machine eps)      #
# --------------------------------------------------------------------------- #

def test_flat_grid_parallel_field():
    g = planar_grid()
    psi, (E1, E2), (edges, w, rho), lam = qm.cross_field(g.vertices, g.faces)
    assert np.abs(rho).max() < 1e-12               # aligned frames -> no transport
    assert abs(lam) < 1e-9                          # parallel field exists
    assert np.std(np.abs(psi)) / np.mean(np.abs(psi)) < 1e-9   # |ψ| const


# --------------------------------------------------------------------------- #
# Connection Laplacian is Hermitian + PSD                                     #
# --------------------------------------------------------------------------- #

def test_connection_laplacian_hermitian_psd():
    s = icosphere(3)
    E1, E2, _ = frames_from_normals(s.vertices, s.faces)
    edges, w, rho = transport_edges(s.vertices, s.faces, E1, E2)
    L, M = connection_laplacian(len(s.vertices), edges, w, rho, 4,
                                vertex_areas(s.vertices, s.faces))
    assert np.abs((L - L.getH()).toarray()).max() < 1e-12
    vals = spla.eigsh(L, k=3, M=M, sigma=-1e-8, which="LM",
                      return_eigenvectors=False)
    assert vals.min() > -1e-8                       # PSD


# --------------------------------------------------------------------------- #
# Test 14b: singularity index sum = χ                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mesh", [icosphere(3), ellipsoid(4, axes=(1, .7, .5))])
def test_singularity_index_sum_is_chi(mesh):
    psi, (E1, E2), _, _ = qm.cross_field(mesh.vertices, mesh.faces, mult=4)
    idx = qm.singularity_indices(mesh.vertices, mesh.faces, psi, E1, E2, mult=4)
    print(f"[singularities] sum={idx.sum():.4f} chi={mesh.euler_number} "
          f"#sing={int((np.abs(idx) > 0.05).sum())}")
    assert abs(idx.sum() - mesh.euler_number) < 1e-6
    # indices are multiples of 1/4, localized at a few singular faces
    nz = idx[np.abs(idx) > 0.05]
    assert np.allclose(nz * 4, np.round(nz * 4), atol=1e-6)
    assert len(nz) < 0.1 * len(mesh.faces)


# --------------------------------------------------------------------------- #
# Test 14c: cross field aligns toward principal directions (ellipsoid)        #
# --------------------------------------------------------------------------- #

def test_cross_field_aligns_to_principal():
    # the q-guided field follows principal directions where the surface is
    # anisotropic (the smoothest field is gauge-ambiguous up to a global
    # rotation and would not).
    m = ellipsoid(4, axes=(1.0, 0.65, 0.5))
    psi, (E1, E2), _, _ = qm.cross_field(m.vertices, m.faces, mult=4,
                                         guide_weight=10.0)
    err = qm.principal_alignment_error(m.vertices, m.faces, psi, E1, E2, mult=4)
    print(f"[alignment] guided {err:.2f}° (random ≈ 22.5°)")
    assert err < 8.0
    # the guided field still resolves the forced singularities -> Σ index = χ
    idx = qm.singularity_indices(m.vertices, m.faces, psi, E1, E2, mult=4)
    assert abs(idx.sum() - m.euler_number) < 1e-6


def test_cross_directions_unit_tangent():
    s = icosphere(3)
    psi, (E1, E2), _, _ = qm.cross_field(s.vertices, s.faces, mult=4)
    cd = qm.cross_directions(psi, E1, E2, mult=4)
    n = vertex_normals(s.vertices, s.faces)
    assert np.allclose(np.linalg.norm(cd, axis=1), 1.0, atol=1e-9)
    assert np.abs((cd * n).sum(1)).max() < 1e-9     # tangent


# --------------------------------------------------------------------------- #
# Stripes: alias-free band pattern aligned to the cross field                 #
# --------------------------------------------------------------------------- #

def test_stripe_field_pattern():
    s = icosphere(4)
    V = s.vertices
    m = Mesh(V * (1 + 0.18 * (V[:, 2] ** 2 - 0.3))[:, None], s.faces)
    psi, (E1, E2), _, _ = qm.cross_field(m.vertices, m.faces, mult=4)
    cd = qm.cross_directions(psi, E1, E2, mult=4)
    sp, lam = qm.stripe_field(m.vertices, m.faces, cd, nu=12.0)
    ph = qm.stripe_phase(sp)
    assert np.isfinite(ph).all()
    assert ph.min() < -0.5 and ph.max() > 0.5       # spans bands (not constant)
    assert lam >= -1e-8
