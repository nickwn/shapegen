"""Generative modelling over shape codes.

``basis.py``  meshes <-> coefficient vectors (PCA over positions, or rho-hat).
``flow.py``   rectified-flow generative model over those vectors.

The split is the point: ``flow.py`` only ever sees an ``(N, K)`` array, so the
representation is a swap of one object rather than a rewrite.
"""
