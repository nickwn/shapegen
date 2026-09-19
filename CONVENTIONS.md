# CONVENTIONS.md — authoritative; never drift

These are the fixed conventions for `shapegen`, a codec mapping genus-0 surfaces
to band-limited spherical-harmonic coefficients of ρ. Every kernel must conform;
tests assert them. If a library disagrees, wrap it and verify against these.

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

## Base sphere and gauge

- The codec decodes from a **base sphere** carrying the mesh's own connectivity.
  `codec.base_sphere` produces it (cMCF) and Möbius-normalizes it against the
  mesh's conformal density.
- Möbius normalization kills the ℓ=1 freedom; the unit-sphere projection kills
  ℓ=0 scale. **Rotation is not fixed by anything** and remains an open gauge —
  two shapes differing by a rotation encode to different ρ̂.
- ρ̂ is meaningless without its base sphere; `EncodeResult` carries both.

## Real spherical harmonics

- Real orthonormal basis, ordered ℓ ascending and m ∈ [−ℓ, ℓ], so the first
  `(L'+1)²` columns *are* the bandwidth-L' basis. Slicing that way is relied on
  throughout; asserted in `tests/test_encode.py`.

## Precision split (M3 / MPS)

- MPS has no float64. Run all of `geom/` — conformal param and the Dirac solve
  included — in **NumPy/SciPy float64 on CPU**. The spectral core (`spectral/`)
  is float64 NumPy. The float32/MPS half applies to networks, of which this
  package currently has none.
