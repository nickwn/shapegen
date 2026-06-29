# CONVENTIONS.md — authoritative; never drift

These are the fixed conventions for `bottlegen`. Every kernel must conform; tests
assert them. If a library disagrees, wrap it and verify against these.

## Algebra & immersion

- **Quaternions** ℍ, basis `1,i,j,k`, `ij=k`. Storage order is **`[w,x,y,z]`**.
  Identify ℝ³ ≅ `Im ℍ`: `(x,y,z) ↔ xi+yj+zk`. Immersion `f: M → Im ℍ`.
- **Spin transform** (Crane–Pinkall–Schröder, *Spin Transformations of Discrete
  Surfaces*): a new immersion satisfies `df̃ = λ̄ (df) λ`, `λ: M → ℍ`; it is
  conformal, `|df̃| = |λ|²|df|`; integrable iff the quaternionic **Dirac equation**
  `Dλ = ρλ` holds (`ρ: M→ℝ` = change in mean-curvature half-density). Discrete:
  solve `(D − ρ)λ = γλ` for the smallest-|γ| eigenpair; transformed half-density
  `µ̃ = µ + (ρ−γ)|df|`, with `µ = H|df|`. Reconstruct positions by integrating
  `df̃` (per-coordinate Poisson solve; translation is the gauge).
  The sandwich helper computes `λ̄ v λ` for `v ∈ Im ℍ`.

## Orientation

- Outward normal; convex-positive curvature. Sphere of radius R: `H = +1/R`,
  `K = +1/R²`, `q = 0`.

## Spin-weighted spherical harmonics (SWSH)

- `₍s₎Y_{ℓm}` defined for ℓ ≥ |s|, m ∈ [−ℓ, ℓ].
- **Definition used here** (single internally-consistent convention; anchored to
  scipy `sph_harm_y` at s=0):
  `₍s₎Y_{ℓm}(θ,φ) = √((2ℓ+1)/4π) d^ℓ_{m,−s}(θ) e^{imφ}`,
  with `d` the Wigner small-d function (Wigner sum formula); the Condon–Shortley
  phase already lives inside `d^ℓ_{m,−s}`, so spin 0 reproduces scipy
  `sph_harm_y` exactly. Orthonormal: `∫ ₍s₎Y_{ℓm} ₍s₎Ȳ_{ℓ'm'} dΩ = δ_{ℓℓ'} δ_{mm'}`.
- **Ladder:** `ð ₍s₎Y = +√((ℓ−s)(ℓ+s+1)) ₍s+1₎Y`,
  `ð̄ ₍s₎Y = −√((ℓ+s)(ℓ−s+1)) ₍s−1₎Y`.
- **Spin-s Laplacian** `Δ_s = ½(ðð̄ + ð̄ð)`, eigenvalue `−[ℓ(ℓ+1)−s²]`.
- **SH normalization:** orthonormal, Condon–Shortley phase included; asserted in
  tests.

## Coefficient layout

- Dense complex array `C[ℓ, ℓ+m]` of shape `(L+1, 2L+1)`, for ℓ ∈ [0..L],
  m ∈ [−ℓ..ℓ] (column index `ℓ+m` ∈ [0, 2ℓ]). Unused entries are zero.
- Grid: Gauss–Legendre in θ (`Nθ = L+1` nodes in `x=cosθ`), equiangular in φ
  (`Nφ = 2L+1`). This makes `analy∘synth = id` exactly for band-limited spin
  fields (the relevant products are degree-≤2L polynomials in cosθ).

## State algebra

- σ spin-0; Hopf `q = ð²σ` spin-2 ⟹
  `q̂_{ℓm} = √((ℓ−1)ℓ(ℓ+1)(ℓ+2)) · σ̂_{ℓm}`; cross-field representative spin-4.

## Linearization (radius R)

- `σ̂_{ℓm} = (2R²/(ℓ(ℓ+1))) ρ̇_{ℓm}`; mean-curvature response multiplier
  `m_ℓ = (ℓ−1)(ℓ+2)/(ℓ(ℓ+1))`. Used only to *initialize* ρ for the Dirac decode;
  iterate to exactness.

## Gauge fix

- Drop ℓ≤1 (ℓ=0 scale, ℓ=1 translation/Möbius). State lives in ℓ≥2; `q=ð²σ`
  already annihilates ℓ≤1.

## Willmore & structure equations

- `W = ∫H²dA ≥ 4π`; `W − 4π = ¼∫(κ₁−κ₂)²dA = ‖q‖²`. Forward blur rate
  `κ_ℓ = (ℓ−1)ℓ(ℓ+1)(ℓ+2)`.
- Metric `e^{2u}|dz|²`. Physics-prior residuals: Gauss
  `−Δ₀u = e^{2u}H² − e^{−2u}|q|²`; Codazzi `∂_z̄ q = ½ e^{2u} ∂_z H`.

## Precision split (M3 / MPS)

- MPS has no float64. Run all of `geom/`, `flow/willmore.py`, conformal param,
  and the Dirac solve in **NumPy/SciPy float64 on CPU**. Networks run
  **float32 on MPS**. The spectral core (`spectral/`) is float64 NumPy.
