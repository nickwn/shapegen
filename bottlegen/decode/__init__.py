"""Decode: robust σ̂ → mesh and curvature-adaptive surface sampling.

``reconstruct.decode`` wraps the Session-3 Dirac spin transform with an
amplitude **trust region** that guarantees a non-degenerate genus-0 mesh: the
linearized decode amplifies high-ℓ content by ``(ℓ(ℓ+1))²`` and can fold the
surface, so we take the largest amplitude that stays valid (no flipped faces,
bounded Willmore). ``sample_points`` draws curvature-adaptive surface samples
(dense where the Hopf differential ``|q|`` is large).
"""
