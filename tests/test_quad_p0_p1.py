"""Gates P0 and P1 of PLAN.md — the regression corpus and the ρ fixtures.

P0 unblocks P5-P7: with ``cones_solved.npy``/``cell.npy`` in hand, later steps
can be written and tested out of order, exactly as the bundle's own
``04_extraction`` scripts run standalone.
"""

import numpy as np
import pytest

from shapegen.geom.encode import n_coeffs
from tests.quad_fixtures import (
    F4_SPECTRA,
    fixture_f0,
    fixture_f1,
    fixture_f4,
    load_ref,
    sphere_mesh,
)


# --------------------------------------------------------------------------- #
# P0 -- the regression corpus                                                 #
# --------------------------------------------------------------------------- #

def test_gate_p0_corpus_loads():
    """Every file in the corpus loads, and is not a truncated copy."""
    names = [
        "cones_solved", "cone_idx", "cell",
        "V", "F", "rho", "Vs", "u", "N", "k1", "k2", "d1",
        "umb_p", "umb_i",
        "Rm", "pole", "cones_plane",
        "q2_pos", "q2_P3", "q2_F", "q2_adj",
        "quad_keys", "quad_z", "quad_P3", "quad_F",
        "psiG", "psi0", "uM", "uL", "rho2",
    ]
    for name in names:
        arr = load_ref(name)
        assert arr.size > 0, f"{name}.npy loaded empty"


def test_gate_p0_cone_configuration():
    """8 cones, all index +1/4, index sum 2.00 -- the solved configuration
    every later step (P6+) is checked against."""
    cone_idx = load_ref("cone_idx")
    cones_solved = load_ref("cones_solved")
    assert cones_solved.shape == (8,)
    assert cone_idx.shape == (8,)
    assert np.allclose(cone_idx, 0.25)
    assert cone_idx.sum() == pytest.approx(2.00)


def test_gate_p0_cell_is_scalar():
    cell = load_ref("cell")
    assert np.asarray(cell).size == 1
    assert float(np.asarray(cell).item()) > 0.0


def test_gate_p0_umbilic_indices():
    """The 4 generic umbilics that split into the 8 cones above: index +1/2
    each, summing to the Poincare-Hopf total of 2 for a genus-0 surface."""
    umb_i = load_ref("umb_i")
    umb_p = load_ref("umb_p")
    assert umb_p.shape == (4, 3)
    assert np.allclose(umb_i, 0.5)
    assert umb_i.sum() == pytest.approx(2.00)


def test_gate_p0_mesh_matches_repo_icosphere():
    """The corpus's V/F are exactly ``geom.mesh.icosphere(4)`` -- so vertices
    line up 1:1 across every saved array and this repo's mesh, no re-indexing
    needed anywhere downstream."""
    V_ref, F_ref = load_ref("V"), load_ref("F")
    V, F = sphere_mesh(4)
    assert V_ref.shape == V.shape
    assert np.abs(V_ref - V).max() < 1e-14
    assert np.array_equal(F_ref, F)


# --------------------------------------------------------------------------- #
# P1 -- ρ fixtures                                                            #
# --------------------------------------------------------------------------- #

def test_gate_p1_matches_bundle_rho():
    """``real_sh_basis(V, 32) @ ρ̂`` reproduces the bundle's ``rho.npy`` to
    1e-12 (PLAN.md Gate P1), for the same seed and spectrum, after matching
    the m-sign and sqrt(2) conventions in ``bundle_style_coeffs``."""
    _, _, rho, _ = fixture_f1(L=32, seed=1)
    rho_ref = load_ref("rho")
    assert np.abs(rho - rho_ref).max() < 1e-12


def test_gate_p1_f0_is_the_identity():
    V, F, rho = fixture_f0()
    assert rho.shape == (len(V),)
    assert np.all(rho == 0.0)


def test_gate_p1_f4_spectra_are_distinct():
    """F4's four spectra (umbilic-count sweep, §P8.5) each build a valid,
    distinct band-limited field from the same machinery as F1."""
    prev_rms = None
    for power in F4_SPECTRA:
        _, _, rho, coeffs = fixture_f4(power)
        assert coeffs.shape[0] == n_coeffs(32)
        rms = float(rho.std())
        assert rms > 0.0
        if prev_rms is not None:
            assert rms != pytest.approx(prev_rms)
        prev_rms = rms


def test_gate_p1_ell0_ell1_absent():
    """The bundle's own sanity check, ported: F1's field has no ell=0/1
    content -- ``random_rho`` starts at ``lmin=2``. Checked by the mesh's own
    (approximate, unweighted) quadrature, so the tolerance is loose relative
    to Gate P1's coefficient-space 1e-12."""
    from shapegen.geom.encode import real_sh_basis

    V, F, rho, coeffs = fixture_f1()
    Y01 = real_sh_basis(V, 1)
    proj = Y01.T @ rho / len(V)
    assert np.abs(proj).max() < 1e-3
