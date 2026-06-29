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

## Session 2 — Discrete DG, curvature, Hopf ✅

**Files:** `bottlegen/geom/mesh.py`, `operators.py`, `curvature.py`,
`tests/test_curvature.py`.

- Light `Mesh` container (trimesh-like `area`/`center_mass`/`euler_number`/
  `genus`) + analytic generators (`icosphere`, `ellipsoid`) with closed-form
  curvature.
- Cotan stiffness `L = D − W` (PSD, ≈ −Δ) + lumped barycentric mass `M`;
  area-weighted vertex normals; per-undirected-edge cotan weights; angle defect.
- Curvature: `H = |½M⁻¹LV|` (convex-positive), `K` via angle defect, per-vertex
  shape-operator fit (principal curvatures), Hopf `q = ½(II₁₁−II₂₂) − i II₁₂`.

**DoD met.** 11 tests green. Sphere median errors drop ~4×/subdivision
(med|H−1|: 9.9e-3→5.9e-4); ellipsoid vs closed-form H,K median rel err 2e-4/3e-4;
Gauss–Bonnet `ΣKᵢAᵢ = 2πχ` to 1e-6; q→0 at umbilics; `|q| = ½|κ₁−κ₂|` to 10%.
Appendix B row 3 ✅.

## Session 3 — Quaternionic Dirac & spin transform ✅

**Files:** `bottlegen/geom/dirac.py`, `tests/test_dirac.py`.

- Intrinsic Dirac `X_{pq} = −(1/4A_f) e_p e_q` (face-assembled), represented via
  the real 4×4 left-mult matrices → real symmetric `4|V|×4|V|` sparse operator;
  `(X − M·diag(ρ))λ = γMλ` smallest-|γ| via shift-invert; gauge-fixed by dividing
  out the mean spinor.
- Edge transform `ẽ_{i→j} = ∫₀¹ λ̄(t) e λ(t) dt` (⅓,⅙,⅙,⅓) + cotan-Poisson
  integration (pinned exact solve). `reconstruct(σ̂)` with the calibrated linear
  drive `ρ̇ = −(ℓ(ℓ+1))² σ̂` and a damped encode-residual loop.

**DoD met.** 8 tests green. **ρ=0 ⇒ λ≡1, identity reconstruction to ~1e-14**
(the machine-precision gate); X symmetric & row-sum-zero to 1e-10; constant/
Möbius conformal maps reproduced to 1e-14; σ̂ decodes to a finite-W genus-0 mesh.
Appendix B row 4 ✅.

## Session 4 — Spherical conformal param + Möbius normalization ✅

**Files:** `bottlegen/geom/conformal.py`, `tests/test_conformal.py`.

- cMCF `(M+dt·L0)f' = M·f` (fixed original `L0`, evolving mass) → unit sphere;
  ball-automorphism Möbius centering (conformal barycenter → 0, kills ℓ=1);
  conformal Hopf field transported to the sphere θ̂/φ̂ frame; spin-2 least-squares
  SHT; `σ̂ = q̂/√((ℓ−1)ℓ(ℓ+1)(ℓ+2))`, ℓ≤1 zeroed.
- Fixed the principal-line angle (`α = −½ arg q`) so the Hopf field is rotation
  covariant.

**DoD met.** 11 tests green. cMCF ellipsoid→sphere (radii std 1e-17); round &
Möbius spheres → σ̂≈0 (~2e-4); **rotation equivariance exact vs the Wigner-D
oracle (3e-15)**; per-degree norm rotation-invariant; Möbius gauge invariance of
σ̂ (0.0035 rel); encode∘decode spectral rel err 0.34, Chamfer 0.057 < feature
scale (approximate inverse). Appendix B rows 5–6 ✅.

## Session 5 — State, forward processes, terminal prior ✅

**Files:** `bottlegen/flow/schedule.py`, `willmore.py`, `state.py`,
`tests/test_flow.py`, `tests/test_state.py`.

- `kappa`, analytic `blur`/`reverse_velocity`, `TerminalPrior` (diagonal
  Gaussian over surviving low-ℓ, Hermitian-symmetrized).
- Spectral Willmore flow `σ̂(t)=σ̂e^{−κ_ℓ t}` + `W = 4π + Σκ_ℓ|σ̂|²` and mesh
  `∫H²dA`; state packing (flat + degree-tokenized), gauge fix, Sobolev weights.

**DoD met.** 13 tests green. Blur leaves only low-ℓ; 1-step reverse first-order;
W monotone↓ to 4π with q→0; fitted decay rate = κ_ℓ to 1e-6; **‖q‖² two ways
(spectral vs mesh W−4π) agree to 4.7%**; state round trips; gauge fix idempotent.
Appendix B row 8 ✅.

## Session 6 — Dataset object & augmentation ✅

**Files:** `bottlegen/data/dataset.py`, `classprior.py`, `tests/test_dataset.py`.

- Self-contained Wigner-D (`spectral/swsh.py`: `euler_zyz`, `wigner_D`,
  `rotate_coeffs`, reusing the verified small-d) → `wigner_rotate_sigma` free
  rotation augmentation; uniform `random_so3`; `BottleSigma` map-style dataset
  (lazy torch); `ClassPrior` empirical σ̂ distribution.

**DoD met.** 18 tests green (2 torch `__getitem__` tests skipped — torch not
installed). Rotation preserves per-degree ‖σ̂‖ to 1e-9; Wigner-D matches the
grid-rotation ground truth (5e-15) and the `spherical` oracle (1e-9);
composition `D(R₂)D(R₁)=D(R₂R₁)`; augmented σ̂ decodes to a recognizably rotated
bottle (aligned residual 0.14 vs 0.66 baseline); class-prior power spectrum
rotation-invariant to 6e-16. Appendix B row 9 ✅.

> **Total: 80 passed, 2 skipped.** Geometry/spectral/flow are self-contained
> NumPy/SciPy float64 (no torch needed); `spherical`/`quaternionic` are
> test-only oracles. The Dirac decode is the gated machine-precision kernel; the
> encode∘decode round trip is an approximate inverse (~0.34 spectral) as the
> plan anticipates.

## Session 7 — Simple model: SH cold-diffusion DiT (MVP) — TODO
## Session 8 — Decode, adaptive sampling, eval — TODO
## Session 9 — DiffusionNet set-latent VAE — TODO
## Session 10 — Rectified flow (vanilla → physics-matched) — TODO
## Session 11 — Quad meshing from cross field — TODO
## Session 12 — Physics-prior losses, full eval, ablations — TODO
