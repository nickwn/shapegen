# PROGRESS

Re-run `pytest` at the end of every session; a green Appendix B is the
definition of "still correct."

## Session 1 — Spectral core: quaternions, SWSH, ð ✅

**Files:** `bottlegen/spectral/quaternion.py`, `bottlegen/spectral/grids.py`,
`bottlegen/spectral/swsh.py`, `tests/test_swsh.py`, `tests/test_quaternion.py`.

- Quaternion algebra (`[w,x,y,z]`, `ij=k`): mul/conj/inv/norm, ℝ³↔Im ℍ, and the
  spin sandwich `λ̄ v λ`.
- GL×equiangular grid (`Nθ=L+1`, `Nφ=2L+1`) with exact solid-angle weights.
- SWSH via Wigner small-d (sum formula), self-contained NumPy/SciPy (float64, no
  heavy runtime deps): `synth`/`analy`, the ð/ð̄ ladder, spin-s Laplacian
  `½(ðð̄+ð̄ð)`, and the Hopf scaling `q=ð²σ`.
- Convention verified against the **`spherical`** oracle (tests only) at every
  spin, and scipy `sph_harm_y` at spin 0.

**DoD met.** 29 tests green. Max round-trip error per spin (L=16):
spin 0 = 2.8e-12, spin 2 = 2.2e-12, spin 4 = 2.0e-12. (Accuracy holds through
L≈24; beyond that switch to a recursive backend such as `s2fft`.)

Appendix B coverage: row 1 (round trip s∈{0,2,4}, 1e-10) ✅,
row 2 (ð ladder; Δ_s; q=ð²σ scaling, 1e-9) ✅.

## Session 2 — Discrete DG, curvature, Hopf — TODO
## Session 3 — Quaternionic Dirac & spin transform — TODO
## Session 4 — Spherical conformal param + Möbius normalization — TODO
## Session 5 — State, forward processes, terminal prior — TODO
## Session 6 — Dataset object & augmentation — TODO
## Session 7 — Simple model: SH cold-diffusion DiT (MVP) — TODO
## Session 8 — Decode, adaptive sampling, eval — TODO
## Session 9 — DiffusionNet set-latent VAE — TODO
## Session 10 — Rectified flow (vanilla → physics-matched) — TODO
## Session 11 — Quad meshing from cross field — TODO
## Session 12 — Physics-prior losses, full eval, ablations — TODO
