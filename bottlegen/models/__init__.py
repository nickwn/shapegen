"""Generative models over the SWSH state σ̂ (torch, float32 on MPS).

The Simple model (Session 7) is a degree-tokenized cold-diffusion DiT: it learns
to reverse the analytic spectral blur on σ̂ coefficients. Packing/blur live in
``pack.py``; the network in ``blocks.py`` / ``sh_dit.py``; the cold-diffusion
training step, EMA, terminal prior, and sampler in ``cold_diffusion.py``.
"""
