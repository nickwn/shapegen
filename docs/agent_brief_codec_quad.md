# Brief: ρ codec (quotient encode, spectral decode) and quad-mesh generation

For an agent picking up work in this repo. It says what each stage computes, where
the code is, what must stay true, and the traps already paid for. Numbers are
measured (2026-09/10) unless marked otherwise.

---

## 0. The representation in one paragraph

A genus-0 surface is stored as **ρ̂**: real spherical-harmonic (SH) coefficients,
band `L` (so `(L+1)²` numbers), of a scalar **ρ** on the unit sphere. ρ is the
*curvature potential* of a **spin transformation** of the round sphere.

- **Spin transformation.** The surface is a conformal immersion `f: S² → ℝ³` with
  `df = λ̄ df₀ λ`. `df₀` is the round sphere's differential, and `λ` is a quaternion
  field (the **spinor**: `|λ|²` is the local length scale, its direction is the
  local rotation).
- **The Dirac equation.** λ is a solution of `(D − ρ) λ = 0`.
- **Convention.** `ρ_bundle = 1 + c·ρ_repo` with `c = 1`, so `ρ̂ = 0` is the round
  sphere. The `+1` is applied **only** in `dirac_spectral.rho_on_grid`.
- **What ρ̂ means.** It is defined *relative to a conformal sphere map of the shape*
  (`EncodeResult.sphere`). Always decode against the sphere the code was encoded
  with.
- **Any code is valid.** Every ρ̂ decodes to a valid closed surface. That property
  is the main reason to use ρ.

Public API (`shapegen/codec.py`):

```python
res = codec.encode(mesh, L=32, method="quotient")   # or "dirac", "spectral", "direct"
X   = codec.decode(res.coeffs, res.sphere)           # positions at res.sphere points
err = codec.shape_error(mesh.vertices, X)            # == res.err (similarity-invariant)
```

---

## 1. The base sphere (shared by every encoder)

`codec.base_sphere(mesh)` → `geom/eigen_sphere.conformal_sphere(V, F)`. About 0.7 s on
10k vertices. Gives a conformal map of the mesh onto the unit sphere, one point per
vertex.

**How it works.**
- Dirichlet energy is conformally invariant, so the mesh's cotan stiffness `L0` is
  also the stiffness of its conformal image. Only the mass differs.
- Solve `L0 φ = λ N φ` with `N` = the round sphere's area pulled back. Its ℓ=1
  eigenvectors are the sphere coordinates.
- The unknown is `N`. It is iterated in a K=100 eigenbasis using the triple products
  `C_jkl = ∫φ_jφ_kφ_l`, with a first-order Möbius boost each step and Anderson
  acceleration. About 16 steps.
- The result is then mirrored if needed and passed through `mobius_normalize`.

**Traps.**
- The **rotation is not fixed and not reproducible** (eigsh starts from a random
  vector). Never decode against a freshly recomputed sphere.
- It needs N > K+1 vertices and well-shaped triangles. `codec.encode(refine="auto")`
  remeshes inputs under 10k faces to ~12k, then carries the sphere back to the
  original vertices.

---

## 2. Spectral decode (ρ̂ → surface)

Code: `geom/dirac_spectral.py` (spinor) and `geom/spectral_adjoint.SpectralLift`
(positions). Entry point: `quad/lift.lift_spectral` / `codec.decode`. No mesh is
involved anywhere.

### 2a. Spinor solve, `dirac_spectral.solve_spinor(rho_hat, L, Ls=...)`

Split `λ = ψ1 + ψ2 j`. ψ1 is spin-weight +½, ψ2 is spin −½, each expanded in
spin-weighted SH up to the half-integer band **Ls**. Use:

    Ls = spinor_band(L, potential_band_extra(L))    # 39.5 at L=32 → n = 1640 modes per spin

The Kamberov–Pedit–Pinkall pair becomes the linear operator

    G(ψ1, ψ2) = ( ð ψ2 + M₊(ρ) ψ1 ,  ð̄ ψ1 − M₋(ρ) ψ2 )
    G = [[Mp, E], [Ē, −Mm]],   E, Ē diagonal: ±(ℓ+½)

- **The coupling blocks** `M±(ρ)` are Galerkin products `⟨Y_a, ρ Y_b⟩`. They are
  built *by transform*: FFT ρ in φ, then one batched product per Δm with a cached
  Wigner-d table (`SpinorTransform.coupling`). This is exact on `coupling_grid(Ls, L)`.
- **Trap:** a grid sized for two spinor modes only (`half_grid`) aliases the top ~30
  degrees of the coupling.
- **The decoded spinor** is the kernel of G: the right singular pair with the
  smallest singular value γ.
- **How it is solved.**
  - `DiracOperator` factors G through the n×n Schur complement `S = Ē + Mm E⁻¹ Mp`.
    E is diagonal with |E| ≥ 1, so this is 3.3× cheaper than an LU of the full
    2n×2n G.
  - `lu_kernel_pair` then runs block inverse iteration on G*G.
  - About 0.4 s at L=32. Cost grows as n³: L=64 (n=6480) takes ~13 s and ~5 GB.
- **Hermitian form.** `σ₃G = K + ρ` is Hermitian, with `K = [[0, E], [−Ē, 0]]` the
  round Dirac operator.
  - Adding a constant to ρ shifts every eigenvalue and leaves every eigenvector
    unchanged. So **ℓ=0 of ρ̂ does not change the decoded surface.**
  - `close_potential` sets ℓ=0 so that γ = 0 exactly (the code is then "closed").
  - At γ = 0 exactly, G is singular. The adjoint in the optimiser needs a small
    nonzero γ while fitting.
- **γ meaning.** γ is the residual. The realised curvature is `ρ − γ`.

### 2b. Spinor → quaternion field (the dictionary), `quad/lift.frame_quaternion`

    λ(x) = R(θ,φ) · (ψ1 + ψ̄2 j),     R = q_z(φ) · q_y(θ − π/2)

The frame R goes on the **left**, and ψ2 appears **conjugated**.

- **Trap:** a right-multiplied frame passes every round-sphere (ρ=0) test to 1e-15
  and is wrong at first order in ρ.
- The checks that actually discriminate (in `tests/test_quad_lift.py`):
  - the kernel's quaternionic gauge must act as a **right** constant;
  - `λ_spec⁻¹ λ_mesh` must be constant.
- ψ has half-integer m, so `e^{imφ}` and R are both antiperiodic in φ. Only their
  product is single-valued. Use the same φ branch (`arctan2`) for both.

### 2c. Lift to positions, `SpectralLift(Ls).forward(v) → f̂`

Solve `min ∫_S² |∇f − ω|² dA` with `ω = Im(λ̄ df₀ λ)` (an ℝ³-valued 1-form). Round
SH diagonalise the left side (`∫∇Y·∇Y = ℓ(ℓ+1)`), so this is one weighted sum per
coefficient.

- `f` is **exactly** band `Lf = 2·Ls` (79 at L=32), because ω is quadratic in λ.
- The grid must integrate degree `Lf + 2Ls`. A grid sized for λ alone aliases (~1%
  error).
- Evaluate positions at any sphere points with `codec._eval_sh(P, f̂, Lf)`. The
  result is the same however the points are sampled or connected.
- The decode is defined up to a similarity (rotation, scale, translation), so
  compare decodes with `shape_error` or Procrustes.

---

## 3. Quotient encode (mesh → ρ̂), `method="quotient"`

Code: `geom/dirac_quotient.py` (`rho_quotient`, `encode_quotient`) and
`geom/dirac_direct.py` (`mesh_spinor`, `_project_faces`, `close_code`). Tests:
`tests/test_dirac_quotient.py`. There is no optimiser and no dense matrix.

**Idea.** The decoder solves `Dψ = ρψ` for ψ given ρ. Encoding runs that backwards:
read the target's ψ off the mesh, then solve the same equation for ρ.

### Step 1: the target's spinor, `mesh_spinor(S, T, F, Lm)` with `Lm = L + ½`

1. **Per-face λ** (`spin.face_lambdas`). The affine map from sphere triangle to
   target triangle, `B A⁻¹`, is `s·R` when the map is conformal.
   `λ = √s · conj(quat(R))`, with `s = √(area_T / area_S)`. Shear is discarded.
2. **Consistent signs** (`spin.align_signs`). λ and −λ are the same map. Signs are
   propagated along a **minimum spanning tree on `2 − |cos∠(λ_f, λ_g)|`** of the dual
   graph, with parity computed by pointer jumping.
   - **Trap:** a breadth-first tree runs a sign seam across the surface around folded
     triangles. On FaceScape head 10 of `trainset_001_100` that made `dirac` decode
     at 0.48 against ~0.007.
3. **Remove the decoder frame** at each face centroid: `raw = R⁻¹ λ` gives
   `ψ1 = raw0 + i·raw1` and `ψ2 = raw2 − i·raw3`.
4. **Project onto spin ±½ SH by quadrature**, weighted by sphere-triangle area
   (`_project_faces`). It is separable:
   - the φ factor `e^{−imφ}` is exact per face, built by recurrence;
   - the Wigner-d θ factor is 4-point Lagrange interpolation from a cached table of
     `16(Lm+1)` nodes.

   60 ms on 20k faces, matching the per-point ladders to 7e-7. Cost is linear in
   face count.

**Use `Lm = L + ½`, not `Ls`.** This was measured as more accurate as well as
cheaper: 1.20× vs 1.32× off `dirac` before step 3.

### Step 2: pointwise quotient (first pass)

Zero-pad v to the decoder band `Ls`. `half_modes` orders by ℓ first, so a band-Lm
vector is a prefix of each spin's band-Ls modes. Then:

    ρ(x) − 1 = −Re⟨ψ(x), σ₃ G(1)ψ(x)⟩ / |ψ(x)|²       on coupling_grid(Ls, L)
    ρ̂ = Yᵀ (w · ρ)                                     (YᵀWY = I: no solve)

- Use **no weight**. Weighting by `|ψ|^p` gets worse as p rises. The textbook least
  squares `min ‖G(ρ̂)v‖` is also worse (0.0137 vs 0.0085 on one head).
- This pass alone is 1.23× off `dirac` (whole head) and 1.15× (face).

### Step 3: correct onto the decoder's truncated equation (passes 2–3)

The decoder realises `P_Ls[ρψ] = Dψ` (Galerkin, truncated), not the pointwise
equation. It falls ~7% short at high ℓ. Iterate:

    ρ̂ ← ρ̂ − Q[ G(ρ̂) v ]

- `G(ρ̂)v` is evaluated matrix-free: synthesise ψ, multiply by ρ on the grid, analyse
  to Ls, add `G(1)v`.
- `Q` is step 2's quotient formula. It is the exact inverse when there is no
  truncation, so it works as a preconditioner. About 4 orders of magnitude per pass;
  `iters=3` is the default.
- **Gate:** a closed code is recovered from its own decoded spinor to machine
  precision.

### Step 4: ℓ = 0

`rho_quotient` sets ℓ=0 from the σ₃-Rayleigh quotient of v. It does not change the
surface; it only selects the right kernel.

`encode_quotient(close=True)` (what the codec calls) spends one `solve_spinor`,
warm-started at v, to close exactly (`dirac_direct.close_code`, γ ≈ 1e-14). It
reuses that solve for `f̂`, which is used for scoring. `close=False` skips the solve.

### Numbers (L=32, ico5 FaceScape heads, same sphere for every method)

| method | error vs `dirac` (whole / face) | encode time |
|---|---|---|
| `direct` (mesh curvature `H·e^u − 1`, projected) | 3.41× / 3.62× | 170 ms |
| **`quotient`** | **1.11× / 1.06×** (worst head 1.13×) | **80 ms** encode (`close=False`); ~0.74 s in `codec.encode` |
| `dirac` (reweighted Dirac least squares) | 1 | ~1.0 s |
| `spectral` (L-BFGS fit through the decoder) | ~0.72× | ~28 s |

- Codes move 3% between ico5 and ico6 meshings, the same as `dirac` (`direct`: 4.4%).
  The quotient is a deterministic *measurement*.
- **Where it is weaker:** smooth shapes at low L. The demo bottle is 1.6–1.75× off at
  L=8 and 1.2× at L=16–24. Use `dirac` at L ≤ 16; it costs about the same there.
- **At L=64** (head 339): quotient 1.4 s encode, `dirac` 44 s / 6.6 GB, decode
  ~13 s. `spectral` from ρ̂=0 **breaks**: 53 min, shape error 0.0055 vs `dirac`'s
  0.0025, with visible ringing.

---

## 4. Quad-mesh generation (ρ̂ → closed quad mesh in ℝ³)

Code: `shapegen/quad/`. Entry point: `quad/pipeline.quad_mesh(rho_hat, L,
target_faces=...)`. Demo and renders: `scripts/quad_demo.py`. Cones come only from a
spectral cross field; there are no mesh umbilics and no triangle mesh anywhere.

```python
out = quad_mesh(rho_hat, L=32, target_faces=20000)
out["ok"], out["P3"], out["faces"], out["grid"].euler, out["grid"].coverage
```

**P8: cross field and cones** (`quad/field.py`, `pipeline.solve_field`)

1. `solve_potential_spinor` solves the spinor once, at the padded band. It is reused
   by the potential and by the lift.
   - **Trap:** a spinor at plain `spinor_band(L)` is accepted silently and gets φ's
     top degree 6–9× wrong.
2. `curvature_potential` gives the conformal potential `φ = −log|λ|²`.
   `connection_field` then gives `A = i·ðφ`.
3. The field is a spin-4 cross field ψ (a cross raised to the 4th power, so
   single-valued), with connection operator
   `C ψ = −Δ₀ψ + s²|A|²ψ + 2is(A·∇₀ψ)`, assembled densely (`assemble_C`).
4. `linear_start` takes C's smallest eigenvector. `ginzburg_landau` then runs a
   semi-implicit GL anneal towards |ψ| = 1, one factorisation per ε level.
   - ε must stay **below `epsilon_ceiling = 1/√λ_min(C)`**. Above it the flow
     silently returns the linear start; `check=True` raises instead.
   - ε must stay above ~0.5·π/L, or the vortex cores fall under the grid resolution.
5. `cones_from_field` finds cones as plaquette windings of ψ on a fine grid. Each has
   index ±k/4, and the indices **must sum to +2**.
   - **Trap:** a sign flip gives Euler characteristic −3240 downstream.
   - The round sphere gives 8 cones of +¼.

Optional: `quad/refine.refine_cones` moves cone positions (count fixed) to lower an
area-distortion energy. On traced meshes it gains ~1.08–1.18×; the sampled energy
overstates that ~3×.

**P4: chart** (`quad/cones.project_cones`)

Choose a pole far from every cone, rotate it to north, and project stereographically
to complex positions `z_k` with indices `idx_k`.

- `min_gap` sets the period solve's motion budget (`0.25 × gap`).
- Freeze `pole` for anything that sweeps cone positions.

**P5: period solve** (`quad/periods.solve_lattice`)

The flat structure has developing-map derivative `W(z) = Π (z − C_k)^(−idx_k)`.

- A square grid closes without seams only if every loop period of `∫W dz` (one loop
  per edge of a cone spanning tree) lies on the lattice `ℤ[i]·s`.
- Newton perturbs the cones within budget to round the periods onto the lattice
  (naive rounding, not LLL).
- Cone-to-cone integrals have branch-point endpoints. Gauss–Jacobi quadrature with
  the exact endpoint exponents is machine-precise at 12–16 nodes.
- It takes the **largest** cell size in `cell_sizes` that solves. For a target face
  count, pass `cell_ladder(cell_for_faces(z, idx, n))`, a descending ladder.
  - **Trap:** never pass `cell_for_faces` straight to `quad_grid`. Only solved cells
    close; on a cube an unsolved cell gave Euler characteristic −5430.
- `sublattice="(1+i)Z[i]"` puts every cone on a grid vertex (pure quads) at ~4–8×
  the faces. The default lands half the cones on cell centres, where they become
  triangular or pentagonal faces.

**P6: trace the grid** (`quad/grid.quad_grid`)

- **Two charts.** `z` below the equator and `ζ = 1/z` above. With Σidx = 2, ω is
  regular and nonzero at ζ = 0, so the chart pole has no hidden singularity.
- **Trace.** A wavefront BFS from a seed vertex steps one cell in all four developed
  directions with adaptive RK4, the substep capped near cones. Arrivals are merged
  with existing vertices, and vertices that land on a cone are snapped to it.
- **Connectivity** comes from a rotation system: faces are the orbits of `σ∘α` on
  darts. The surface is closed by construction, so `V − E + F` is a *measurement*.
- **Trust the result only if** `euler == 2`, `coverage == 1.0` and `merge_residual`
  is tiny (the cube fixture gives ~1e-8 of a cell). `converged=False` means the trace
  hit `max_vertices`.
- Ground truth is the cube: exactly 8 cones, and the known mesh at cell `edge/m`.

**P7: lift** (`quad/lift.lift_spectral`)

The traced grid points are directions on the unit sphere. Positions come from the
*same* spinor (§2c's spectral least squares, evaluated at those directions). About
1.8 s at L=32 for 20k faces. Per-face anisotropy (`aniso`) and `quad_isotropy`
score the result.

**What decides quality.** Cone placement (GL) and cell size. Face area varies with
the conformal factor: cells are uniform in the *flat* metric, not on the surface.
Report the traced-mesh metrics, not the sampled energy.

---

## 5. Invariants to keep (tests enforce most of them)

- `ρ_bundle = 1 + ρ_repo` is applied once (`rho_on_grid`).
- A decode uses the encoder's own `sphere`.
- Spinor work feeding the potential or lift uses `spinor_band(L, potential_band_extra(L))`.
- Coupling uses `coupling_grid(Ls, L)`; the lift uses its own grid at degree
  `Lf + 2Ls`.
- Cone indices sum to +2; the quad grid needs `euler == 2` and `coverage == 1`.
- Sign alignment uses the spanning tree (`spin.align_signs`), not breadth-first.
- Measurement hygiene:
  - Compare codes only after rotation alignment, and excluding ℓ=0.
  - Compare surfaces with exact point-to-surface distance (`igl.point_mesh_squared_distance`).
    Point-to-point chamfer between two 200k samplings has a ~0.004 floor on an
    RMS-normalised head.
  - Score the face separately. `metrics.nose_tip` can pick the occiput on
    icosphere-resampled heads; on FaceScape the face is the +z end.

## 6. Open threads (not in the repo)

- **Adaptive resolution.** A Möbius zoom of the sphere toward the face gives the face
  2.5× (position SH) / 1.8× (ρ) at ×1.5, at no whole-head cost.
- **Adapted-metric ρ decoder.** A background metric `g = e^{2w}·round` with eigenbases
  of g, using conformal covariance `D_g = e^{−3w/2} D e^{w/2}`. A curvature-driven w
  beats the Möbius zoom, reaching the same face error with 9% better rest-of-head.
  Prototype only, one head.
- **Storage comparison.** A quadric-decimated mesh beats both position SH and ρ above
  ~1000 stored floats. ρ is most compact only below that.
