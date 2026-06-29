"""Dataset over the cached bottle σ̂ spectra, with free rotation augmentation.

A global rotation acts on σ̂ by the Wigner-D matrices (σ is spin-0), so a random
``R ∈ SO(3)`` per sample is an exact, label-preserving augmentation — no
geometry recomputation — turning a few hundred bottles into an effectively
unbounded training set.
"""
