"""Discrete connection Laplacian for n-direction (RoSy) fields on a surface.

A unit n-direction field is a per-vertex complex number ``z_i = e^{i n θ_i}`` in
the vertex tangent frame; the *smoothest* such field minimizes the Dirichlet
energy ``Σ_ij w_ij |z_i − r_ij^n z_j|²`` under parallel transport, where
``r_ij = e^{i ρ_ij}`` rotates frame ``j`` into frame ``i`` across the shared
edge (Knöppel et al., *Globally Optimal Direction Fields*). The minimizer is the
smallest eigenvector of the Hermitian connection Laplacian ``L`` (generalized
against the lumped mass ``M``).

``ρ_ij`` is the discrete Levi-Civita connection angle: the edge direction
measured in frame ``i`` minus the same direction measured in frame ``j``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .curvature import tangent_frames
from .operators import edge_cotan_weights, vertex_areas, vertex_normals

__all__ = ["transport_edges", "connection_laplacian", "smallest_eig",
           "frames_from_normals"]


def frames_from_normals(V, F, normals=None):
    """Per-vertex orthonormal tangent frames ``(E1, E2, n)`` (reuses the
    deterministic reference-axis construction)."""
    return tangent_frames(V, F, normals)


def transport_edges(V, F, E1, E2):
    """Per-undirected-edge cotan weight and Levi-Civita transport angle.

    Returns ``(edges (E,2), w (E,), rho (E,))``: ``rho_ij`` rotates frame ``j``
    onto frame ``i`` (the edge direction's angle in frame ``i`` minus its angle
    in frame ``j``).
    """
    edges, w = edge_cotan_weights(V, F)
    i, j = edges[:, 0], edges[:, 1]
    e = V[j] - V[i]
    ai = np.arctan2((e * E2[i]).sum(1), (e * E1[i]).sum(1))
    aj = np.arctan2((e * E2[j]).sum(1), (e * E1[j]).sum(1))
    rho = ai - aj
    return edges, w, rho


def connection_laplacian(n_verts, edges, w, rho, mult, mass=None):
    """Assemble the Hermitian connection Laplacian for an ``mult``-direction
    field. Returns ``(L, M)`` (complex CSR, real diagonal mass).

    ``L_ij = −w_ij e^{i·mult·ρ_ij}`` (off-diagonal), ``L_ii = Σ_j w_ij``; PSD,
    energy ``Σ w_ij |z_i − e^{i·mult·ρ_ij} z_j|²``.
    """
    i, j = edges[:, 0], edges[:, 1]
    r = np.exp(1j * mult * rho)
    I = np.concatenate([i, j])
    J = np.concatenate([j, i])
    vals = np.concatenate([-w * r, -w * np.conj(r)])
    L = sp.coo_matrix((vals, (I, J)), shape=(n_verts, n_verts)).tocsr()
    diag = np.zeros(n_verts)
    np.add.at(diag, i, w)
    np.add.at(diag, j, w)
    L = (L + sp.diags(diag)).tocsr()
    if mass is None:
        mass = np.ones(n_verts)
    return L, sp.diags(mass)


def smallest_eig(L, M, sigma=-1e-8):
    """Smallest generalized eigenpair ``(λ, ψ)`` of the Hermitian ``L`` against
    SPD ``M`` (shift-invert just below 0)."""
    vals, vecs = spla.eigsh(L, k=1, M=M, sigma=sigma, which="LM")
    return float(vals[0]), vecs[:, 0]
