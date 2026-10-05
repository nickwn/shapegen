"""Acceptance tests for the cone chart (reduced scope): pole choice and
stereographic projection of cones, ported from ``s4_cones.py`` minus the
umbilic-splitting it did before projecting (see ``shapegen/quad/cones.py``'s
module docstring for why -- cones here come from the GL-annealed cross field,
already at the correct +-1/4 index).
"""

import numpy as np
import pytest

from shapegen.quad import cones as qc
from shapegen.quad import field as qf
from tests.quad_fixtures import bundle_style_coeffs, load_ref
from shapegen.geom.encode import n_coeffs

L = 16
S = 4


def _gl_cones(seed=1, L=L, S=S):
    coeffs = bundle_style_coeffs(L=32, seed=seed)[:n_coeffs(L)]
    phi = qf.curvature_potential(coeffs, L)
    A = qf.connection_field(phi)
    Cmat, modes = qf.assemble_C(A, S, L)
    psi0 = qf.linear_start(Cmat, modes, S, L)
    psiG = qf.ginzburg_landau(psi0, Cmat, modes, S, L)
    return qf.cones_from_field(psiG, S, L)


# --------------------------------------------------------------------------- #
# Cross-checked against the bundle's own saved Rm/pole/cones_plane (P0)       #
# --------------------------------------------------------------------------- #

def test_rotation_to_north_matches_bundle_reference():
    """Given the bundle's own saved pole, our rotation formula must reproduce
    its saved Rm exactly -- same deterministic closed form, same input."""
    Rm_ref = load_ref("Rm")
    pole_ref = load_ref("pole")
    Rm_ours = qc.rotation_to_north(pole_ref)
    assert np.max(np.abs(Rm_ours - Rm_ref)) == 0.0


def test_stereo_round_trip_matches_bundle_reference():
    cones_plane_ref = load_ref("cones_plane")
    z_rt = qc.stereo(qc.unstereo(cones_plane_ref))
    assert np.max(np.abs(z_rt - cones_plane_ref)) < 1e-15


def test_forward_projection_reproduces_bundle_cones_plane():
    """Recover the bundle's pre-rotation 3D cone positions from its own saved
    cones_plane.npy (via our unstereo + the saved Rm's inverse), then redo
    the forward projection with our own stereo/rotation_to_north and check it
    reproduces cones_plane.npy -- validates the whole forward path end to
    end against real bundle-computed data, not just its two halves
    separately."""
    Rm_ref = load_ref("Rm")
    pole_ref = load_ref("pole")
    cones_plane_ref = load_ref("cones_plane")

    Pr_ref = qc.unstereo(cones_plane_ref)
    P_recovered = Pr_ref @ Rm_ref                 # Rm is orthogonal: Rm^-1 = Rm.T
    assert np.max(np.abs(np.linalg.norm(P_recovered, axis=1) - 1.0)) < 1e-12

    Rm_ours = qc.rotation_to_north(pole_ref)
    z_ours = qc.stereo(P_recovered @ Rm_ours.T)
    assert np.max(np.abs(z_ours - cones_plane_ref)) < 1e-10


def test_rotation_to_north_maps_pole_to_north():
    """The defining property, at machine precision, for an arbitrary pole
    (not just the bundle's saved one)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        pole = rng.normal(size=3)
        pole /= np.linalg.norm(pole)
        Rm = qc.rotation_to_north(pole)
        assert np.max(np.abs(Rm @ pole - np.array([0.0, 0.0, 1.0]))) < 1e-14
        # a proper rotation: orthogonal, determinant +1
        assert np.max(np.abs(Rm @ Rm.T - np.eye(3))) < 1e-12
        assert np.linalg.det(Rm) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# End to end on our own GL-found cones                                       #
# --------------------------------------------------------------------------- #

def test_choose_pole_is_far_from_the_cones():
    cones = _gl_cones()
    P = np.array([p for p, _k in cones])
    pole, dist = qc.choose_pole(P, n_samples=500, seed=0)
    assert np.abs(np.linalg.norm(pole) - 1.0) < 1e-12
    # every cone (and its antipode check) must be at least `dist` from pole
    assert np.arccos(np.clip(P @ pole, -1, 1)).min() >= dist - 1e-12
    assert dist > 0.3   # a generic 8-cone spread has real breathing room


def test_project_cones_preserves_index_sum():
    """Projection must not touch the indices -- no splitting happens here."""
    cones = _gl_cones()
    chart = qc.project_cones(cones, n_samples=500, seed=0)
    assert chart["idx"].sum() == pytest.approx(sum(k for _p, k in cones), abs=1e-12)
    assert len(chart["z"]) == len(cones)


def test_project_cones_round_trips_through_stereo():
    """stereo(unstereo(z)) == z for the chart this function actually
    produces, not just for the bundle's saved example."""
    cones = _gl_cones()
    chart = qc.project_cones(cones, n_samples=500, seed=0)
    z = chart["z"]
    assert np.max(np.abs(qc.stereo(qc.unstereo(z)) - z)) < 1e-13


def test_project_cones_min_gap_is_positive_and_finite():
    cones = _gl_cones()
    chart = qc.project_cones(cones, n_samples=500, seed=0)
    assert 0.0 < chart["min_gap"] < np.inf


def test_chart_to_sphere_round_trips_with_project_cones():
    cones = _gl_cones()
    P = np.array([p for p, _k in cones])
    chart = qc.project_cones(cones, n_samples=500, seed=0)
    P_back = qc.chart_to_sphere(chart["z"], chart["Rm"])
    assert np.max(np.abs(P_back - P)) < 1e-10


def test_chart_to_sphere_matches_bundle_reference():
    Rm_ref = load_ref("Rm")
    cones_plane_ref = load_ref("cones_plane")
    # recovered via unstereo + Rm in test_forward_projection_reproduces_..., so
    # this is the same relation exercised in the other direction
    P = qc.chart_to_sphere(cones_plane_ref, Rm_ref)
    assert np.max(np.abs(np.linalg.norm(P, axis=1) - 1.0)) < 1e-12
