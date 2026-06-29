"""Forward/reverse processes on the SWSH state σ̂ (NumPy float64).

The forward process is the analytic spectral *blur* — conformal Willmore
curvature flow linearized on the sphere, ``σ̂_{ℓm}(t) = σ̂_{ℓm}(0) e^{-κ_ℓ t}``
with ``κ_ℓ = (ℓ−1)ℓ(ℓ+1)(ℓ+2)`` — which smooths any shape toward the round
sphere. Generation learns to reverse it (cold diffusion / inverse heat
dissipation).
"""
