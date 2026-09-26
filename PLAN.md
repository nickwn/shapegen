# PLAN.md — porting the quad-meshing prototype into shapegen

Take a band-limited `ρ̂` and emit a **quadrilateral** mesh. The algorithms exist
already, as research scripts in `~/Downloads/bundle` (from
`quadmesh-codebase.zip`). This document is the ordered port of that code into
this package — what moves, what is replaced by something this repository already
does better, what is rejected, and the gate on each step.

It is a port plan, not a design document. Where the source already answers a
question, the answer is quoted with its measurement rather than re-derived.

---

## 0. Provenance, and how numbers are marked

The source bundle is research code: scripts that run top-to-bottom, print
measurements, and `exec()` each other to reuse setup. Its own README says so.
`PLAN.md` inside it (the document this one replaces) carries the reasoning; the
scripts carry the evidence.

Three markers are used throughout, and they are not interchangeable:

- **[verified]** — re-run on this machine, today. Five headline checks
  reproduce via the bundle's `run_checks.sh`:

  | check | script | bundle claim | re-run here |
  |---|---|---|---|
  | single-valued `w'` identity | `04_extraction/polyfield.py` | `9.9e-16` | **`7.87e-16`** |
  | tracing throughput, 6561 vertices | `04_extraction/wavefront.py` | `0.129 s` | **`0.098 s`** |
  | one RK4 step per cell is enough | `04_extraction/simpler2.py` | `8.9e-07` | **`8.94e-07`** |
  | cone layout on the lattice | `02_periods/conegate.py` | `7.5e-14` | **`1.37e-13`** |
  | global solve is exact, not `O(h²)` | `05_experiments/poisson2.py` | `1.7e-15` | **`2.48e-15`** |

  Same order of magnitude throughout; the small differences are machine and
  seed. Treat the re-run column as the number to reproduce.

- **[bundle]** — printed by a source script that was **not** re-run here. Real
  measurements from working code, but not independently confirmed. Where a port
  step has a [bundle] number as its gate, running the source script first is the
  cheapest way to get a trustworthy target.

- **[unbuilt]** — no implementation exists, in the bundle or here. Exactly one
  thing is in this category, and it matters: **the full quaternionic Dirac
  solve**. Everything in the bundle uses the linearisation `μ = 1 + 2ρ`.

### The bundle is not fully runnable

Several scripts `exec()` a sibling to reuse setup, and **four of those siblings
are not in the bundle**:

| script | needs | status |
|---|---|---|
| `02_periods/cvp.py` | `fastfail.py` | **missing** |
| `02_periods/cached.py` | `cvpmod.py` (holds `lll()`) | **missing** |
| `03_spectral/gl.py` | `subface.py` | **missing** |
| `02_periods/build.py` | `template.html`, `img/`, `solve.pkl` | **missing** |

So the LLL/Babai target selection (§P5.3) has to be **reconstructed from its
description**, not transcribed — `choose_target` and `target_cached` are present
but the `lll()` they call is not. `02_periods/build.py` is an HTML report
builder, not the "EMST, loop construction, condition selection" its README entry
claims; that code is actually inside `01_pipeline/s5_grid.py`, which *is*
self-contained.

---

## 1. The source, file by file

### 01_pipeline — mesh-based, runs end to end

| file | what it does | port verdict |
|---|---|---|
| `s1_mesh.py` | hand-rolled icosphere(4) = 2562 vertices; `random_rho` builds a real band-limited field by hand | **delete** — both exist here, §P1 |
| `s2_surface.py` | `ρ → surface` by the *linearised* transform, radial graph `Vs = V(1+εu)` | **replace** with `codec.decode`, §P2 — the single biggest upgrade in this port |
| `s3_umbilics.py` | umbilics as singularities of the principal-direction **line** field; index by parallel transport per face | **port**, §P3 |
| `s4_cones.py` | split each index-`+½` umbilic into two `+¼` cones; pick a pole; stereographic projection | **port**, §P4 |
| `s5_grid.py` | the whole period solver, self-contained: Gauss–Jacobi cut integrals with gradients, EMST, loop set, rank selection, Newton + homotopy, cell-size sweep | **port**, §P5 — the most valuable single file in the bundle |
| `s6b_extract.py` | march, **geometric** vertex merge, quads as 4-cycles, closure-defect filter, lift to ℝ³ | **port with changes**, §P6/§P7 |

### 02_periods

| file | what it does | port verdict |
|---|---|---|
| `core.py` | developing map, holonomy, period/rotation of a loop, clearance, Gauss–Bonnet — clean and self-contained | **port** as diagnostics, §P5 |
| `conegate.py` | the cheap gate: accumulate `w` at every cone along the spanning tree; compares four-way branch selection against unwrapped `arg(ψ)` and shows they agree | **port**, §P5.4 — **[verified] 1.37e-13** |
| `cvp.py` | greedy lattice-target selection to minimise *predicted cone motion*, not period distance | **reconstruct**, §P5.3 — needs the missing `fastfail.py` |
| `cached.py` | Babai on a once-LLL-reduced basis; scale-invariant in `s` | **reconstruct**, §P5.3 — needs the missing `cvpmod.py` |
| `build.py` | HTML report builder | **reject** |

### 03_spectral — mesh-free, where the plan is headed

| file | what it does | port verdict |
|---|---|---|
| `sht.py` | `Grid(L, s, over=2.4)`: spin-weighted transforms, per-`m` FFT in φ, GL in θ | **merge** with the repo's restored `swsh.py`, §P8.1 |
| `halfspin.py` | half-integer degrees via `Γ(x+1)`; orthonormality and ð eigenvalues | **port**, §P8.2 |
| `solver.py` | connection operator `C` from the curvature density; `ð̄ð` check; dense assembly column by column | **port**, §P8.3 |
| `gl_spec.py` | Ginzburg–Landau anneal; cones by cell winding on a fine grid | **port**, §P8.4 |
| `tune.py` | the `ε` schedule sweep — the pitfall that cost the most | **port as a test**, §P8.4 |
| `hopf.py` | `Q = ð²u` as a genuine spin-2 field; winding; umbilic count vs spectrum | **port**, §P8.5 |
| `blend.py` | the smoothness ↔ alignment control `t` | **port**, §P8.6 |
| `count.py`, `pairs.py` | why the cone count stays at 8; whether extra pairs survive the flow | **port as tests** |
| `crossfield.py` | cone placement and smooth cross fields solve the *same* Poisson equation; the norms differ (`H⁻¹` vs `H⁻²`) | **read, do not port** — it is the argument behind §3.5 |
| `frameop.py`, `gl.py` | the same solver on a triangle mesh, for comparison (`gl.py` needs the missing `subface.py`) | **oracle only** |
| `coneopt2.py`, `distortion2.py`, `whyfixed.py`, `emerge.py`, `score.py` | point-based area-distortion objective and why point relaxation cannot make cones emerge | **read, do not port** — §3.5 |
| `curved.py` | the mesh comparison on a varying-curvature surface | **oracle only** |

### 04_extraction

| file | what it does | port verdict |
|---|---|---|
| `polyfield.py` | verifies `(w')⁻⁴ = ∏(z−pᵢ)^{±1}` and single-valuedness | **port as a test** — **[verified] 7.87e-16** |
| `simpler2.py` | substep sweep; **and the `log|ψ|`-as-a-sum version of `psi_arg`** | **port this one's `psi_arg`**, §P6.1 |
| `wavefront.py` | batch across lines — but uses `np.prod(np.abs(D))`, which overflows | **port the structure, not its `psi_arg`** |
| `phase5.py` | RK4 accuracy, and Newton correction against a Gauss–Legendre segment integral | **port**, §P6.3 |
| `speed5.py`, `analytic5.py` | the optimisation ladder; Taylor + series reversion | **read, do not port** |

### 05_experiments — measured and rejected, with one exception

| file | approach | outcome |
|---|---|---|
| `arch5.py`, `arch3.py` | disc-based series tracing | 2.6 ms/vertex — *slower* than the ODE — and 10% of steps wrong **[bundle]** |
| `fftcoef.py` | Taylor coefficients by FFT on a circle | branch continuation did not close (4.9% jump) **[bundle]** |
| `cutsolve.py` | cut along the EMST, solve with transitions | edge classification near cones never made correct; residual `1.4e-02` **[bundle]** |
| `sparse_w.py` | how sparse can the samples be | exact at 40 points; binding constraint is triangle size vs cell size **[bundle]** |
| `poisson2.py`, `poisson_w.py` | global least-squares solve on a simply-connected region | **this one works** — **[verified] 2.48e-15**, independent of point count |

> **`poisson2.py` is filed under "rejected" and is not rejected.** Both the
> bundle's README and the re-run say it is *exact*, not `O(h²)`, because `w′dz`
> is closed and the periods are integral — "the mesh is scaffolding for linear
> algebra, not a discretisation". It is directly relevant to §P7, where a global
> solve is one of the two candidate lifts. Do not skip it because of the folder
> it sits in.

---

## 2. What the port changes, and why

Five decisions that are not transcription. Each is a place where this repository
already knows something the prototype did not.

### 2.1 `s2_surface.py` is deleted, not ported — `codec.decode` replaces it

The prototype builds its surface as a **radial graph**:

```python
u_lm = 2*rho_lm/(l*(l+1)-2)          # linearised spin transform
Vs   = V*(1 + EPS*u)[:,None]         # eps = 0.35
```

Two costs, both recorded in the bundle's own README. The divisor `ℓ(ℓ+1)−2`
means a `ℓ⁻²` curvature spectrum yields a surface effectively band-limited near
**`ℓ = 7`** regardless of the nominal `L = 32` — "the surface looks far smoother
than `ρ`". And a radial graph cannot represent a non-star-shaped shape at all.

`shapegen/geom/dirac.py` solves the real discrete spin transform — `A(ρ)λ = γMλ`
with `ρ` coupled *inside* the Dirac energy, then edge transform and a
cotan-Poisson integration — and its docstring states outright that this is what
buys non-star-shaped genus-0 shapes. It is tested (`tests/test_dirac.py`) and it
is what every number in the README was measured with.

**So the mesh pipeline's stage 2 becomes one call to `codec.decode`.** This
removes the bundle's "Caveat on s2" entirely, and it is available immediately —
it does not wait on §P9.

It does **not** fix the mesh-free path: `03_spectral/solver.py` builds its
connection from `μ = 2ρ` (linearised) too, and replacing *that* is §P9.

### 2.2 `s1_mesh.py` is deleted — both halves already exist here

`icosphere(4)` (2562 vertices) is `shapegen.geom.mesh.icosphere(4)`.
`random_rho`'s hand-rolled real-SH sum, with its `√2` and `(−1)^m` bookkeeping,
is `shapegen.geom.encode.real_sh_basis` — already orthonormal, already in a
fixed layout, already asserted in `tests/test_encode.py`. A power-law spectrum is
then `band_index(L)` and one `**power`. The fixture is three lines, not a module.

### 2.3 Two spin-weighted transform implementations — merge, do not pick

`03_spectral/sht.py` has the padded grid (`nth = int(2.4*L)+8`, `nph = 2*nth`)
that products need, and a per-`m` FFT in φ.

This repository had its own, deleted at `5f84d24` and recoverable:

```bash
git show e2e7ab1:bottlegen/spectral/swsh.py      # 279 lines
git show e2e7ab1:bottlegen/spectral/grids.py     #  49 lines
git show e2e7ab1:tests/test_swsh.py              # 182 lines
```

The repository's version is **tested against an independent oracle** — the
`spherical` package's `Wigner(L).sYlm` via `quaternionic`, cross-checked against
scipy `sph_harm_y` at spin 0, at spins 0/2/4, tolerance 1e-10 (round trip) and
1e-9 (ð ladder). Both packages are installed and declared in `pyproject.toml`.
`sht.py` has no oracle at all.

**Keep the oracle-tested module as the base; take `sht.py`'s `over=` padded grid
and `halfspin.py`'s `Γ(x+1)` extension into it.** The repo's `grids.py` uses the
tight `Nθ = L+1, Nφ = 2L+1` that makes `analyse ∘ synth = id` *exactly*; keep
that as the default and add the padded grid as a separate constructor, because
several existing tests rely on the exact identity.

### 2.4 The ρ convention differs by a constant

The bundle's `ρ` is the mean-curvature half-density; the unit sphere has `ρ ≡ 1`,
which is what its `PLAN.md` Gate 1.A asserts. This repository's `ρ` is the
**change** in that density (`CONVENTIONS.md`): `ρ = H·e^u − 1`, round sphere
`ρ ≡ 0`.

```
    ρ_bundle = c·ρ_repo + 1          c ∈ {1, 2}, PINNED EMPIRICALLY, not assumed
```

`c` is left open deliberately: the bundle's own linearisation is `μ = 1 + 2ρ`,
which puts a factor of 2 somewhere in its conventions, and its `PLAN.md` says not
to trust any reference on the point including itself. Pin `c` against the
round-sphere gate, then state it once in the module docstring.

In the real orthonormal basis, `Y₀₀ = 1/√(4π)`, so adding the constant 1 is
`coeffs[0] += sqrt(4*pi)` — `3.5449077…`, not `1.0`.

> **Pitfall.** Apply that offset **exactly once**, in the one boundary function
> where a `codec` `ρ̂` enters bundle-derived code. Twice, or again inside a
> re-measurement pass, shifts the realised curvature by a constant and the
> failure reads as a scale error in the output mesh, not as a bug.

### 2.5 γ is not the same object in the two formulations

| | `geom/dirac.py` (here) | the bundle's `PLAN.md` Phase 1 |
|---|---|---|
| equation | `A(ρ)λ = γMλ`, `A = (D−ρ)*M_F(D−ρ)` | `(D − ρ)λ = γλ` |
| sign | `γ ≥ 0` always, PSD | real, either sign |
| meaning | **squared** Dirac residual | spectral shift; realised curvature `ρ − γ` |
| multiplicity | 4 (the right ℍ action) | 2 per quaternionic mode |

Because `A` is the Gram operator of `D − ρ`, the two are *predicted* to satisfy
`γ_mesh ≈ γ_spec²` up to the mass normalisation. That is a prediction, gated in
§P9, not a derivation done here. `EncodeResult.gamma` carries `γ_mesh`; do not
let a number cross between the two without the square.

---

## 3. Conventions the port must obey

`CONVENTIONS.md` is authoritative and is not amended by this plan.

- Quaternions `[w,x,y,z]`, `ij=k`, ℝ³ ≅ Im ℍ. Spin transform `df̃ = λ̄ df λ`.
- Outward normal, convex-positive: sphere of radius `R` has `H = +1/R`.
- Real SH layout: ℓ ascending, `m ∈ [−ℓ, ℓ]`, first `(L'+1)²` entries **are** the
  band-`L'` code. The bundle's complex `C[ℓ, ℓ+m]` needs one conversion pair,
  gated once (§P8.1), not ad-hoc conversions at each use.
- **float64 NumPy/SciPy on CPU** for all of `geom/`, `spectral/` and the new
  `quad/`. No torch, no float32, no MPS. `shapegen/util.py` says why.
- λ is fixed up to a global right ℍ factor; `dirac._gauge_fix` divides by the
  mean spinor so a constant field becomes `λ ≡ 1`. Any new solver uses the same
  convention or `φ = 2log|λ|` differs by a constant between paths.
- Möbius normalisation kills ℓ=1, the unit-sphere projection kills ℓ=0,
  **rotation is fixed by nothing**. Cone positions are therefore defined relative
  to the base sphere's frame. On a shared template `ρ̂` is rotation-invariant to
  4e-15, so cones are reproducible there and not otherwise.

---

## 4. Destination layout

```
shapegen/spectral/
    grids.py            restore from e2e7ab1; + sht.py's padded grid
    swsh.py             restore from e2e7ab1; + halfspin.py's half-integer degrees
    products.py         NEW  dealiased products (from solver.py's dealias)
shapegen/geom/
    dirac_spectral.py   NEW  P9 only; named to sit beside dirac.py on purpose
shapegen/quad/
    __init__.py         NEW  public surface: quad_mesh() -> QuadResult
    umbilics.py         NEW  P3   line-field winding                  <- s3
    cones.py            NEW  P4   splitting, pole choice, stereographic <- s4
    periods.py          NEW  P5   cut integrals, EMST, lattice solve  <- s5_grid, conegate
    extract.py          NEW  P6   marching, merging, quads            <- s6b, 04_extraction
    lift.py             NEW  P7   chart -> R^3
    field.py            NEW  P8   spectral cross field, GL, blend     <- solver, gl_spec, blend
tests/
    data/quadmesh_ref/  NEW  P0   the bundle's saved intermediates
    test_swsh.py        restore from e2e7ab1; extend
    test_quad_umbilics.py  test_quad_cones.py  test_quad_periods.py
    test_quad_extract.py   test_quad_field.py  test_dirac_spectral.py
```

Public surface mirrors `codec.py` — one import, diagnostics that cannot be
silently dropped:

```python
@dataclass
class QuadResult:
    vertices: np.ndarray      # (n,3)
    quads:    np.ndarray      # (m,4)
    cones:    np.ndarray      # (k,2) chart positions with indices
    cell:     float           # the solved cell size s
    index_sum: float          # must be 2
    coverage: float           # fraction of sphere area inside the chart -- §9
    anisotropy: dict          # median / p90
    rejected: int             # quads dropped by the closure-defect filter
    gamma:    float           # from the decode that produced the surface
```

`EncodeResult` exists in that shape because an iteration-capped encode is an
upper bound, not an answer. Same reasoning: `coverage` and `rejected` are the two
numbers that make a plausible-looking quad mesh wrong, so they travel with it.

---

## 5. The port, step by step

Ordered so that each step is testable when it lands, and so the highest-value
replacement (§2.1) happens early rather than after a faithful transcription of
the thing it replaces.

### P0 — The regression corpus (do this first; it unblocks P5–P7)

`bundle/data/` is **1.1 MB** of saved intermediates. Copy into
`tests/data/quadmesh_ref/`:

| file | what | used by |
|---|---|---|
| `cones_solved.npy`, `cone_idx.npy`, `cell.npy` | the solved 8-cone configuration and its cell size | P5, P6 |
| `V.npy`, `F.npy`, `rho.npy`, `Vs.npy`, `u.npy`, `N.npy`, `k1.npy`, `k2.npy`, `d1.npy` | the mesh pipeline's stage outputs | P1–P3 |
| `umb_p.npy`, `umb_i.npy` | umbilic positions and indices | P3 |
| `Rm.npy`, `pole.npy`, `cones_plane.npy` | the projection and the unsolved cones | P4 |
| `q2_pos.npy`, `q2_P3.npy`, `q2_F.npy`, `q2_adj.npy`, `quad_*.npy` | the extracted quad mesh | P6, P7 |
| `psiG.npy`, `psi0.npy`, `uM.npy`, `uL.npy` | spectral field intermediates | P8 |

**This is the whole reason the port can be done out of order.** With
`cones_solved.npy` and `cell.npy` in hand, P6's marching can be written and
tested against `q2_*.npy` before P3, P4 or P5 exist — which is exactly how the
bundle's own `04_extraction` scripts run standalone.

**Gate P0** — load every file; assert 8 cones, all index `+¼`, index sum `2.00`,
and that `cell.npy` is a scalar. Cheap, and it catches a truncated copy.

### P1 — ρ fixtures (replaces `s1_mesh.py`)

`icosphere(4)` from `geom.mesh`; a power-law `ρ̂` from `real_sh_basis` and
`band_index`. Plus the fixtures that the bundle does not have and this repository
does:

| # | input | why |
|---|---|---|
| F0 | `ρ ≡ 0` on `icosphere(4)` | the identity; pins every convention |
| F1 | power-law `ℓ⁻²`, `ℓ = 2…32`, seed 1 | the bundle's own driver; comparable numbers |
| F2 | `codec.encode(bottle, L=32, method="residual")` | a real code, `demo_bottle_meshes(n=1, subdiv=3)` |
| F3 | FaceScape head, `remesh=0.8`, `t_smooth=1e-4`, `L=32` | the hard case, `γ_mesh = 2.7e-2` |
| F4 | spectra `ℓ⁻²`…`ℓ⁻⁰·⁵` | umbilic-count sweep, P8.5 |

**Gate P1** — `real_sh_basis(V,32) @ ρ̂` reproduces the bundle's `rho.npy` to
`1e-12` for the same seed and spectrum, after matching the `m`-sign and `√2`
conventions. If it cannot be matched exactly, record the conversion explicitly;
do not leave two real-SH conventions live in one repository.

Two warnings about F3, both already measured here: `γ` is **not** small on real
codes (2.7e-2 at L=32), so "the realised curvature is `ρ − γ`" is a first-class
statement about the output; and ρ is ~99% unrepresentable on scanned faces, where
position-SH is 6.2× better at equal cost. **Drive development on F1/F2. F3 is a
stress test, not a target.**

### P2 — Surface by the real spin transform (replaces `s2_surface.py`)

`codec.decode(ρ̂, sphere, faces)`. Recover the curvature machinery the bundle's
stage 2 computes by least-squares normal fitting:

```bash
git show e2e7ab1:bottlegen/geom/curvature.py   # principal_curvatures, hopf_q,
                                               # second_fundamental_form, tangent_frames
```

**Gate P2.A — the swap is a strict improvement, and is measured as one.** On F1
at small amplitude, `codec.decode` and the bundle's `Vs = V(1+εu)` must agree to
first order in `ε`; the gap must grow with `ε` and with the high-ℓ content of
`ρ`. This is the linearisation check in reverse, and it is what licenses
discarding stage 2.

**Gate P2.B — the band-limit caveat is real and is quantified.** Compute the
effective bandwidth of the bundle's surface (its own README says `ℓ ≈ 7` for a
`ℓ⁻²` spectrum at nominal `L = 32`) and of `codec.decode`'s. Record both. If the
decode's is not materially higher, the swap bought nothing and the reason must be
found before P3.

### P3 — Umbilics by line-field winding (`s3_umbilics.py`)

Per-face index by parallel transport of the principal direction, matched **mod
π** because it is a line field, not a vector field. Merge within `0.12` and drop
anything with `|index| ≤ 0.25`.

**Gate P3** — index sum `2.000000` **[bundle]**, and the merged umbilic list
matches `umb_p.npy` / `umb_i.npy` from P0 when run on the bundle's own `Vs.npy`.
Run it on the bundle's surface *first*, then on `codec.decode`'s; the two inputs
isolate a port error from a genuine difference in the surface.

### P4 — Cones (`s4_cones.py`)

A generic umbilic has index `+½`; a quad mesh needs `+¼`, so each umbilic splits
into **two** valence-3 cones. Pole chosen by maximising angular distance to the
nearest umbilic over 4000 random directions; rotate pole→north; stereographic
from the south.

> **Port note.** `s4_cones.py` computes its Rodrigues rotation `Rm` three times,
> the first two being dead stores (one of them a wrong formula left in place).
> Port the third expression only, and test it: `Rm @ pole == [0,0,1]` to `1e-14`.

**Gate P4** — index sum `2.00`; stereographic round trip `1e-16`; min cone
separation recorded (the displacement budget in P5 is `0.25 × gap`).

### P5 — Periods and the lattice solve (`s5_grid.py`, `core.py`, `conegate.py`)

The most valuable file in the bundle, and self-contained.

**P5.1 — cut integrals.** Gauss–Jacobi with both endpoint exponents
(`roots_jacobi(n, a[j], a[i])`), value **and gradient in one pass**
(`cut_and_grad`). Machine precision at 12–16 nodes; ordinary quadrature never
gets there. Cache the nodes by `(aᵢ, aⱼ, n)` as the source does.

**P5.2 — the condition set.** EMST over the cones; one loop per edge; same index
→ difference loop, opposite → sum loop. Rank-select until `n−2` independent
conditions (four cone DOF are the affine kernel). Newton with backtracking,
homotopy in `τ`, `lstsq` not a square solve.

> **Proved, do not re-derive.** The `n−1` edge loops are a complete condition
> set: `t(K)` is the `ℤ[i]`-span of the edge periods and `Λ` is a `ℤ[i]`-module.
> No block loops.

> **Pitfall.** Freeze the MST edge list at the start. Recomputing it per
> iteration and rejecting when it changes throws away valid solutions — the loop
> set is defined by cone *pairs* and the proof holds for any spanning tree.

> **Pitfall.** Quadrature nodes and the work cap must scale with the
> configuration. Defaults tuned on evenly spread cones (16 nodes, cap 40) failed
> at a 16:1 spread in cone separations; it needed 48 nodes and a cap of 250.
> `s5_grid.py` hard-codes `n=16` for solving and `n=48` for the final check —
> port that split, and derive the node count from the worst segment clearance.

**P5.3 — target selection, which must be reconstructed.** `s5_grid.py` uses
naive componentwise rounding (`np.round(t0/s)`) and is complete. `cvp.py` and
`cached.py` improve on it — greedy trading of cells between components to
minimise *predicted cone motion*, then Babai on a once-LLL-reduced basis, which
is scale-invariant in `s` because every LLL condition is a ratio of lengths — but
**`lll()` itself is in the missing `cvpmod.py`**.

Port naive rounding first and ship it. Treat LLL/Babai as a separate, later
improvement with its own measurement, not as part of this step.

**P5.4 — the cheap gate, and it is nearly free.** `conegate.py`: accumulate `w`
at every cone along the spanning tree, one cut integral per edge. Every developed
cone must land on an integer or half-integer cell coordinate. **[verified]
`max|2w/s − round| = 1.37e-13`**, 17 edges, ~1 ms. It also demonstrates that
unwrapping `arg(ψ)` and choosing nearest-of-four agree — which is the evidence
for §P6.1.

**Run it before P6 every time.** It catches an upstream error for a thousandth of
extraction's cost.

**Gate P5** — all periods integral to `1e-8` including derived loops never shown
to the solver; cone motion within `0.25 × gap`; P5.4 passing; and the solved
configuration reproduces `cones_solved.npy` and `cell.npy` from P0.

### P6 — Extraction (`s6b_extract.py` + `04_extraction/*`)

> **Superseded by `shapegen/quad/grid.py`.** Everything below is the faithful
> port, kept because `shapegen/quad/extract.py` still reproduces the P0 corpus
> and because §P6.1/§P6.2 (the single-valued identity, log-sum magnitude,
> unwrap `arg ψ` at *its* period not `arg w'` at a four-times-coarser one) are
> load-bearing in the replacement too. What changed: two charts instead of one
> (§9), an adaptive substep instead of a tuned `nsub` (§P6.3 below), and
> connectivity from a rotation system instead of 4-cycles plus a
> closure-defect filter (§P6.5). Result on the same P0 configuration:
> coverage `1.000000000` and `χ = 2`, against `0.911` with 98 valence-2 /
> 275 valence-3 / 77 valence-5 vertices.

**P6.1 — the representation, and why there is nothing to track.**

```
(w')^-4 = prod (z - p_i)^(±1)        <-- single-valued
```

`|w'| = |ψ|^(−¼)` and the four-fold direction set are **global**; only which of
the four is local, and that is the grid's own 90° symmetry. **[verified]** the
identity holds to `7.87e-16` over 2000 points, and ψ returns to itself around a
cone while `w'` comes back rotated.

> **This is the single most useful fact in the bundle.** Its README: "Almost
> every failure in this session came from not knowing this." There was never a
> global branch to handle.

**P6.2 — port `simpler2.py`'s `psi_arg`, not `wavefront.py`'s.** They differ in
one line and it matters:

```python
# wavefront.py -- OVERFLOWS in the far field, a product of n factors
mag = np.prod(np.abs(D), axis=1)
# simpler2.py  -- port THIS
lm  = np.sum(np.log(np.abs(D)), axis=1)
```

Likewise port the **unwrapped argument**, not `s6b_extract.py`'s and
`phase5.py`'s nearest-of-four search: unwrapping is faster *and* more accurate
(`9.5e-12` against `2.4e-11` **[bundle]**) because it never makes a wrong
discrete choice.

**P6.3 — accuracy budget.** The cone solver already moves cones by ~2% of the
cell spacing, so vertex errors below ~1e-3 of a cell are invisible downstream.
**[verified]** substep sweep:

| order | nsub | evals/cell | time (25×25) | median step error |
|---|---|---|---|---|
| RK4 | 8 | 32 | 0.012 s | `1.85e-10` |
| RK4 | 4 | 16 | 0.006 s | `3.01e-09` |
| RK4 | 2 | 8 | 0.003 s | `5.49e-08` |
| **RK4** | **1** | **4** | **0.002 s** | **`8.94e-07`** |
| RK2 | 2 | 4 | 0.002 s | `7.87e-05` — too coarse |
| RK2 | 1 | 2 | 0.001 s | `3.68e-04` — too coarse |
| RK1 | 4 | 4 | 0.002 s | `1.17e-03` — too coarse |

**Default `nsub = 1`.** Note `s6b_extract.py` ships `nsub=14` — 14× the work for
error the pipeline cannot see. Do not port that default.

> **Do not economise on the order instead.** RK2 at the *same* four evaluations
> is ~90× worse than RK4, and Euler worse again. The simplification is "one step,
> not eight" — never "a simpler integrator".

Where clearance to a cone drops below a few cells, raise `nsub` or correct with
Newton against a Gauss–Legendre segment integral (`phase5.py`): two substeps plus
three Newton steps reaches `2.5e-17` and beats 48 substeps at lower cost
**[bundle]**.

**Correction, real faces at L=32:** the sweep above was measured at one fixed
cell size and cone layout, and doesn't generalise the other direction either.
`s` is arc length in the *developed* plane; a chart-space step is `s / |w'|`,
so far from every cone (`|w'|` small) the same `s` is a much bigger step in
raw chart coordinates than near one — the accuracy table's error figures are
for whatever step that particular layout happened to produce, not a property
of `nsub=1` in general. Confirmed on a real face (not the P0 corpus): `nsub=1`
left a visibly self-overlapping (bowtie) quad region far from the cones and
~2.6% degenerate quads; `nsub=8` cleaned it up. `scripts/quad_demo.py`'s
`build_quad_mesh` now passes `nsub=8`; the library default stays `1` (the P0
reproduction test pins it there).

**P6.4 — batch across lines.** A grid line is sequential; different lines are
independent. March one spine, then advance the whole perpendicular wavefront in
one vectorised call. **[verified] 6561 vertices in 0.098 s**, 0.0149 ms/vertex,
of which the spine is 32 ms — the sequential fraction shrinks as the grid grows.

> **Pitfall.** Bound the traced region. The grid marches off the chart, where
> `log|ψ|` grows without limit and `exp` overflows; the prototype produced NaNs
> across an entire run before the region was constrained. Clipping the exponent
> is a patch; the fix is to stop at a radius — and that radius **is** the chart,
> so report it as `coverage` (§4) rather than discovering it later.

**P6.5 — merge geometrically, never by index.**

> **Pitfall, and a real bug in the source's own history.** Integer `(i,j)` labels
> assume trivial holonomy. Walking around a cone returns you rotated a quarter
> turn, so one point of the plane acquires different labels by different routes;
> the symptom is patches of grid, each internally square, meeting at seams.
> `s6b_extract.py` merges by position (within `0.30·s`) and detected **7448**
> path reconvergences **[bundle]**. Quads are then 4-cycles in the adjacency
> graph, not index quadruples.

Port the closure-defect filter with it, and **report what it drops** —
`QuadResult.rejected`. A silently filtered quad mesh looks better than it is.

**Gate P6** — on P0's `cones_solved.npy`: the identity to `1e-15`; 6561 vertices
under 0.1 s with median step error below `1e-5` of a cell; valence histogram
matching the cone list; median quad anisotropy `< 1.2` (**[bundle]** 1.136 over
2464 quads); and the vertex set reproducing `q2_pos.npy`.

### P7 — Lifting the chart to ℝ³

`s6b_extract.py` ends with

```python
Sph = unstereo(pos) @ Rm
uu  = <KD-tree interpolation of u onto Sph>
P3  = Sph*(1 + EPS*uu)[:,None]
```

**That last line is the radial-graph assumption, and it is the only place it
survives into extraction.** With §2.1's replacement there is no scalar `u` to
interpolate. Two candidate lifts:

1. **Interpolate the decoded positions.** `codec.decode` gives positions at the
   base sphere's vertices; the traced points are directions on that same sphere.
   `geom/resample.py` already does exactly this lookup — `locate_on_sphere` /
   `resample_to`, with libigl AABB queries — and is tested. Cheapest, and reuses
   tested code.
2. **Integrate `df̃ = λ̄ df₀ λ` along the traced grid lines.** Exact where the
   Dirac equation holds. But `df̃` is closed only at `γ = 0`; at `γ ≠ 0` the
   integral is **path-dependent**, so a vertex's position depends on the route to
   it. The mesh decode hides this — `integrate_edges` is a least-squares
   cotan-Poisson solve, which projects the inconsistency away. `poisson2.py`
   (§1, and **[verified]** exact to `2.48e-15`) is the same idea and says the
   global least-squares solve is exact rather than `O(h²)`.

**Gate P7.A — holonomy.** Sum `df̃` around traced quad loops. Zero to integration
accuracy on F0; scaling with `γ` on F1–F3. Report it.

**Gate P7.B — agreement with the reference decode.** Chamfer between
`quad_mesh(ρ̂).vertices` and `codec.decode(ρ̂, sphere, faces)`, **restricted to the
charted region**, at or below the codec's own mesh-dependence at that γ (0.057 at
γ≈2e-2, ~0 at γ≈1e-13 — `codec.py`). Unmasked, this number is meaningless; §9.

**Decide between the two lifts on P7.A's number, not in advance.** Route 1 is
available immediately and is probably right for F1–F3; route 2 becomes
interesting only once §P9 lands.

### P7a — route 2, built and measured (2026-09-21)

§P9 has landed and so has route 2: `quad/lift.lift_route2`, plus
`--lift {route1,route2}` on `scripts/quad_demo.py`. **Route 1 remains the
default.** Full numbers and method live in `lift_route2`'s docstring; the
decision and the two gates:

**The blocker was never the integration — it was the spinor-to-quaternion
dictionary**, which `quad/lift.frame_quaternion` now pins exactly:

```
λ(x) = [Re ψ₁, Im ψ₁, Re ψ₂, Im ψ₂] · F(θ, φ),   F = q_y(θ) q_x(φ)
```

right multiplication, no conjugation, up to the global `λ → λq` gauge.
Recovered rather than guessed: `F` is pure sphere geometry, so it is read off
the **round sphere**, where `spin_transform` gives `λ ≡ 1` and `γ_spec = 5.5e-16`
— agreement with the closed form **1.67e-15** over 4000 directions, and
`swsh`'s spin convention cancels in the product. A rotation search cannot find
this: `ℍ = ℂ ⊕ ℂj` quantises spin about **i** while `(θ, φ)` are measured
about **k**, so `F` is a mixed-axis product and not ZYZ-expressible at all
(fitting a fixed axis with `R(x)α = x̂` leaves residual 0.998).

**Gate P7.A — holonomy: measured, and it does not vanish at production γ.**
Summed around the traced quad faces *before* the least-squares solve. At
L=16 on the `ℓ^-2` fixture over 2.6 decades of γ it is monotone and `≈ 0.28√γ`
(1.32e-3 → 3.36e-2 as γ goes 3.08e-5 → 1.19e-2); at γ = 0 it is < 1e-7 and
route 2 returns the round sphere to 3.9e-9. **The solve does not hide it**: the
post-solve edge residual is *larger* than the holonomy at every amplitude, and
on two FaceScape heads at L=32 (γ ≈ 1.9e-2 / 2.6e-2) it is **6.2e-2 holonomy
and 2.5e-1 edge residual** — a quarter of every edge vector is inconsistent.

**Gate P7.B — agreement:** 0.176 / 0.188 Procrustes against route 1, versus
route 1's own mesh dependence of 0.017 / 0.019. Per PLAN's own framing this is
*not* the adjudicator, since route 2 needs only the spectral spinor to be
right; the quality columns are.

**Verdict — opt-in, not default.** Route 2 is **4.3x faster** (4.7s against
route 1's 20.1s at L=32/20k faces) and genuinely **mesh-free**, which is what
M5 asks for. But at production γ it is also measurably worse geometry:
log-area variance 0.064 → 0.107 and 0.065 → 0.147, median quad isotropy
1.008 → 1.158 (p95 1.04 → 1.65-1.69). That is Gate P7.A showing up in the
output, so route 2 becomes the right default when γ falls — not before.

> The 4.3x **requires reusing the spinor** the pipeline already solved for
> `curvature_potential(method="dirac")`; at L=32 that solve is 32.5s against
> the lift's own 3.6s. Re-solving makes route 2 measure 1.8x *slower* than
> route 1. `quad/field.curvature_potential` grew `spinor=`/`sink=` and
> `quad_demo.solve_field` now returns it for exactly this.

> **Correction, 2026-09-26 — the P7a verdict above was the dictionary, not γ.**
> The map above is right on the round sphere and wrong at first order in ρ.
> The correct one is
>
> ```
> λ(x) = R(θ, φ) · (ψ₁ + ψ̄₂ j),   R = q_z(φ) q_y(θ − π/2)
> ```
>
> with the frame on the **left** and ψ₂ **conjugated**; `R` carries the
> quaternion `i` axis to the outward normal. Why: Crane's `D` multiplies by
> frame vectors on the left, so `λ = R ψ`; with the normal along `i`, left
> multiplication by the tangent units is complex-*antilinear* on `(a, b)`, so
> the complex-linear unknowns are `(ψ₁, ψ̄₂)`. The round sphere cannot pin
> this, because any `A(x)·raw·B(x)` with `A raw₀ B = 1` agrees there.
>
> **What gave it away.** (1) Split test on head #339: the spectral λ integrated
> with route 1's own cotan-Poisson reproduces route 2 exactly (0.1760), and the
> mesh λ integrated with route 2's arcs + *unweighted* least squares
> reproduces route 1 to 1e-4 (holonomy 8e-4). So the integration was never
> the problem. (2) Under the old map the spectral kernel's quaternionic gauge
> acted on the **left** (`λ₂λ₁⁻¹` constant to 1e-14); a rigid motion is a
> *right* constant. (3) Old-map λ matched the mesh in `|λ|²` (corr 0.998)
> and in rotation angle, but not in rotation axis: a pointwise conjugation.
> The old test compared `λ_mesh λ_spec⁻¹`, the wrong side, where old and new
> maps score the same.
>
> **Ruled out on the way, same two heads, L=32** (measured, not argued):
> edge weighting of the least squares (`1/|ẽ|²`, `1/arc`, `1/arc²`: within
> ±0.02 of 0.18, never better than 0.1757); spinor band (`extra` 8 → 16: γ
> 0.0189 → 0.0188, Procrustes unchanged); mode selection (next singular value
> 0.83, 44x gap); coupling-quadrature aliasing (real — `half_grid` is exact
> for degree 2Ls, the coupling needs 2Ls+L — but worth only 0.18 → 0.17).
>
> **After the fix** (numbers in `lift_route2`'s docstring): Procrustes against
> route 1 **0.0062 / 0.0067** (was 0.176 / 0.188); against the *scanned head*
> route 2 is 0.0460 / 0.0389 and route 1 0.0472 / 0.0414; log-area variance
> and isotropy equal to route 1's; lift 4.7s vs 20.1s. Gate P7.A holonomy
> 6.7e-3 / 6.2e-3 on faces. On the fixture it is now **second order**
> (≈ 2e-3·γ, 1.2e-7 → 9.3e-5 over amp 0.1 → 4), not `0.28·√γ`; route 2
> matches the decode to 1.3e-3 even at γ = 4.6e-2.
>
> **Untapered** (#339, t = 0): 0.020 vs route 1 (≈ route 1's own mesh
> dependence), holonomy 2.2e-2 at γ 6.6e-3; vs scan still 0.0465 against 0.0482.
>
> **Regime:** L=32, `band_taper` t = 1e-3, 20k faces, two heads, spinor
> reused from `potential="dirac"`. Route 1 stays the default until route 2 has
> been run across more heads. Without a reusable spinor, route 2 pays the
> spinor solve (15-32s at L=32) and is no faster than route 1.
>
> **Default switched, 2026-09-26.** Eight more random heads (10, 31, 70,
> 144, 147, 185, 252, 334) at taper 1e-3, and four of them untapered, against
> a gate fixed before the run (route 2 within 1.25x + 0.005 of route 1 vs the
> scan; p95 isotropy within 0.05): **12/12 pass**, and route 2 is closer to
> the scan on 10/12. The loser is head 252 (+7% tapered, +19% untapered at
> γ = 4.1e-2), where route 1 is itself 3-5x worse than on any other head.
> `quad_demo --lift auto` (the new default) takes route 2 when the spinor is
> in hand (`--potential dirac`) and route 1 otherwise; table in
> the table below.

#### P7a record — route 2 measurements after the dictionary fix (2026-09-26)

Moved here from `lift_route2`'s docstring when it was trimmed.

**Wider check, 2026-09-26 — the basis for making route 2 the default**
(``quad_demo --lift auto``). Eight more heads, drawn at random, same
protocol; "scan" is Procrustes to the scanned head. The gate was fixed
before the run: route 2 within 1.25x + 0.005 of route 1 against the
scan, and p95 isotropy within 0.05. All 12 cases pass::

    head  taper   γ        hol      r1↔r2   scan r1  scan r2  p95 iso r1→r2
     10   1e-3  1.56e-2  6.6e-3   0.0065  0.0382   0.0372   1.039→1.041
     31   1e-3  1.57e-2  6.6e-3   0.0075  0.0390   0.0375   1.040→1.042
     70   1e-3  2.26e-2  6.6e-3   0.0065  0.0452   0.0428   1.039→1.041
    144   1e-3  1.57e-2  6.0e-3   0.0063  0.0296   0.0270   1.039→1.040
    147   1e-3  2.12e-2  6.6e-3   0.0069  0.0394   0.0373   1.039→1.041
    185   1e-3  1.54e-2  6.7e-3   0.0066  0.0311   0.0304   1.039→1.041
    252   1e-3  1.86e-2  6.8e-3   0.0119  0.1320   0.1412   1.044→1.044
    334   1e-3  2.28e-2  6.6e-3   0.0066  0.0332   0.0319   1.039→1.041
     10   0     2.82e-3  2.2e-2   0.0213  0.0382   0.0378   1.045→1.054
     70   0     1.04e-2  2.3e-2   0.0220  0.0456   0.0394   1.048→1.057
    147   0     8.62e-3  2.2e-2   0.0222  0.0363   0.0327   1.047→1.055
    252   0     4.10e-2  2.3e-2   0.0451  0.1987   0.2358   1.068→1.062

Route 2 is closer to the scan on 10 of 12. **The exception is head 252**,
where *both* routes are poor (route 1 is 3-5x further from the scan than
on any other head, i.e. ``ρ̂`` itself does not represent it) and route 2
is worse by 7% tapered and 19% untapered — the latter the highest γ
measured (4.1e-2, 10 cones). That is the regime to watch: route 2 loses
ground where the code is already failing, not where it works. Lift time
with four jobs in parallel: 25s → 5.3s tapered, 48s → 8.7s untapered.

**Gate P7.A is second order in amplitude, as closedness requires.**
``ℓ^-2`` fixture, L=16, icosphere(4), against ``codec.decode`` on the same
mesh; the "old" columns are the pre-fix dictionary::

    amp    γ_spec    | old hol   old resid  old vs r1 | hol       resid     vs r1
    0.1    3.08e-05  | 6.37e-04  4.90e-03   0.0037    | 1.22e-07  6.67e-08  0.0000
    0.5    7.64e-04  | 3.28e-03  2.54e-02   0.0138    | 1.43e-06  1.37e-06  0.0001
    1.0    3.03e-03  | 7.85e-03  6.28e-02   0.0428    | 5.44e-06  5.48e-06  0.0003
    2.0    1.19e-02  | 1.66e-02  1.33e-01   0.1189    | 2.19e-05  2.24e-05  0.0005
    4.0    4.61e-02  | 4.46e-02  3.36e-01   0.2119    | 9.31e-05  9.58e-05  0.0013

Holonomy ``≈ 2e-3·γ`` here. The old ``≈ 0.28·√γ`` law — first order in
amplitude — was never the Dirac residual: it was the wrong dictionary
failing closedness at first order. On faces the holonomy/γ ratio is
larger (≈ 0.35), because the features sit at the band limit.

### P8 — The mesh-free spectral path

This is where the bundle's `03_spectral` goes, and where the pipeline stops
needing a triangle mesh for anything but the encode.

> **P8.1 DONE, gate met.** The decision this step demanded is taken: a stable
> three-term recurrence in ℓ (`swsh.wigner_d_ladder`), not a `ducc0`/`shtns`
> dependency. Round trip at band 32 is **4.3e-14** against the 1e-12 gate, and
> still 1.3e-13 at band 48; the sum formula it replaced measured 5.3e-8 at
> band 32 and overflowed at 64. Bonus: `eval_basis` went 9.5s → 0.53s, because
> the ladder walks all ℓ per m instead of summing each (ℓ,m) independently.
>
> **P8.2 DONE.** Half-integer degrees fall out of the same recurrence with no
> special casing (`a−b` stays integral). `swsh.half_*` adds the container and
> quadrature; round trip 4.8e-14 at L=31.5, 1056 modes per spin.

**P8.1 — transforms.** §2.3's merge. Add the real↔complex coefficient pair.
*Gate:* round trip `1e-12` at band 32 both ways; a field synthesised from
converted coefficients matches `real_sh_basis(V,L) @ ρ̂` to `1e-12`.

> **Accuracy ceiling, and it is already known from both sides.** The restored
> `swsh.py`'s own docstring: ~1e-12 at L=16, inside tolerance "through L≈24,
> beyond which cancellation in the factorial sum starts to matter". The bundle's
> `PLAN.md`: `1.2e-7` at band 32 **[bundle]**. **The target is L=32 and neither
> implementation reaches it.** Either replace the Wigner sum with a stable
> three-term recursion in ℓ, or adopt `ducc0`/`shtns`/`spinsfast` for integer
> spin and keep the hand-rolled path for half-integer. Neither is installed
> (`numpy`, `scipy`, `spherical`, `quaternionic`, `torch`, `pymeshlab`, `libigl`
> are), so option (b) is a `pyproject.toml` change, in a new extra beside
> `spectral`, with the hand-rolled path kept as the reference it is verified
> against. **Take this decision in P8.1, not later** — it is upstream of
> everything in P8 and P9.

**P8.2 — half-integer degrees** (`halfspin.py`). `fac(x) = gamma(x+1.0)`; `ℓ`
ranges become `np.arange(0.5, L+0.5, 1.0)`.
*Gates:* orthonormality to `1e-8` at `(s,ℓ,m) ∈ {(½,½,½), (½,3/2,−½), (−½,5/2,3/2), (½,11/2,7/2)}`; ð eigenvalue ratio constant to `1e-9` **[bundle]**.

> **There is no oracle for this half, and it fails late rather than early.**
> `spherical.Wigner(2.5)` *constructs* without complaint; `sYlm(0.5, R)` then
> dies inside its numba kernel (probed here, `spherical` 1.1.3). These two
> internal identities are the only protection the half-integer path has. Do not
> weaken them, and do not assume that passing at integer spin says anything about
> `ℓ = 11/2`.

**P8.3 — the connection operator** (`solver.py`). `Δ₀Φ = μ − 1` diagonally
(`φ_lm = −μ_lm/(ℓ(ℓ+1))`), `A = ∗dΦ = i·ðφ`, then

```
C ψ = -Lap_0 psi + s^2|A|^2 psi + 2 i s (A . grad psi)
```

with `A·∇ψ = ½(conj(A)·ðψ + A·ð̄ψ)`. The `div A` term vanishes because `A` is a
rotated gradient. *Gate:* `ð̄ð` on spin 0 returns exactly `−ℓ(ℓ+1)` — the source
prints `−6.000, −30.000, −90.000`.

> **Port note.** `solver.py` assembles `C` densely, column by column, one
> transform per column, then `eigh`. At `LPSI=14` that is fine and it is how the
> measurements were taken. It will not scale; leave it dense, and note the
> matrix-free route rather than taking it now.

**P8.4 — Ginzburg–Landau** (`gl_spec.py`, `tune.py`). Semi-implicit flow, one
factorisation per `ε` level, cones read off as cell windings on a fine
(`over=7.0`) grid.

> **Pitfall, the one that cost the most.** `ε` scales to the **field's**
> resolution `π/L_ψ`, not to the fine grid used to read off zeros. Scaling to the
> output grid (spacing 0.026 against the field's 0.224) left the cones
> essentially unmoved — area energy `1.94e-1`, *worse than a plain cube* — and
> rescaling alone took it to `1.22e-1`. Above about `4π/L` the penalty does
> nothing at all. `tune.py` has the sweep; port it as a test, not as a comment.

> **Pitfall.** Do not drive `ε → 0`. Once the vortex core is narrower than the
> resolution the solution degrades. Measured optimum around `1–1.5·π/L`.

*Gates:* index sum `2` to `1e-10`; GL beats the linear eigenproblem on
area-distortion energy by `> 2×` (**[bundle]** 3.15×); on a round sphere it
scores at least as well as an evenly spread cube.

> **P8.4 DONE — and the first schedule shipped was silently not an anneal
> at all (2026-09-21).** The flow has a hard stability ceiling. Linearising
> the semi-implicit step about `ψ = 0`, each eigenmode of `C` is multiplied by
> `(1+c)/(1+c·ε²λ)`, which exceeds 1 only when `ε²λ < 1` — *independently of
> the step size*. So above `ε_crit = 1/√(λ_min(C))` (`field.epsilon_ceiling`,
> ~0.5–0.65 absolute) **every** mode decays, `ψ → 0`, and the per-level
> rescale leaves the slowest-decaying eigendirection: `field.linear_start`.
> The anneal is replaced by the eigenproblem it exists to beat.
>
> The old default `(6.0, 3.0, 2.0)` was entirely above that ceiling at
> `L = 12` and `L = 16`, which is where every test in this repo runs. Measured
> overlap `|⟨ψ_GL, ψ_lin⟩|`: **0.999992** (L=12 amp 1), **0.999643** (L=16
> amp 1), **1.000000** on the round sphere. The replacement is
> `field.DEFAULT_LEVELS = (2.0, 1.74, 1.52, 1.32, 1.15, 1.0)` in units of
> `π/L`, clamped by `field.anneal_schedule`.
>
> **The lesson is not the schedule, it is that topology cannot referee this.**
> On the round sphere at L=16 the collapsed flow reports 8 cones of index
> `+1/4` summing to `+2` to 1e-10 — a flawless Poincaré–Hopf reading — while
> scoring `area_distortion` **2.7525** against the cube's **0.0719**, i.e.
> **38× the known optimum**. Traced all the way to a quad mesh it still gives
> Euler characteristic 2, coverage `1.000000000`, valences and face degrees in
> `{3,4}`, and a median `quad_isotropy` of 1.0783 against the good flow's
> 1.0385 — every structural check passes, and the traced log-area variance is
> 0.2739 against 0.0742. Index sum is *forced* by Poincaré–Hopf and has no
> power to discriminate; isotropy is forced by the developing map. This is the
> third defect here invisible to topology, after the cone index sign error and
> the 91% coverage ceiling.
>
> **Both of this step's quality gates now bite, and both are demonstrated
> failing on the collapse** (`test_quad_field.py`), scored as a geometric mean
> over three in-window schedules because varying only the schedule moves the
> score 1.08–1.21× per case:
>
> | gate | shipped | collapsed |
> |---|---|---|
> | GL beats `linear_start` (`> 2×`) | **3.95×** (L=16; 3.83× at L=12) | **1.031×** |
> | round sphere vs cube (`< 1.25×`) | **1.068×** | **38.3×** |
> | traced log-area variance (`< 0.12`) | **0.0742** | **0.2739** |
>
> **It was a small-`L` failure, which is why it survived.** `ε_crit` in `π/L`
> units is 1.91 at L=12, 2.55 at L=16, 3.82 at L=24, 5.09 at L=32 (round
> sphere), so the old schedule is entirely above it at L≤16 and mostly below
> it at L=32, where `scripts/quad_demo.py` runs. At L=24 the old schedule
> recovers to 3.63× on the first gate — so a quality gate on this step only
> discriminates if it is run at `L ≤ 16`. Two further escapes worth knowing
> about: at L=16 *amplitude 2* the collapsed flow already scores 2.15× on
> the first gate, which would pass a `> 2×` bound; and on the round sphere
> whether topology notices at all depends on `cones_from_field`'s `over` —
> 0 cones at 4.0, a flawless 8 at 7.0.
>
> **Audit of everything else measured under it (2026-09-21).** Every
> downstream claim derived from a `L ≤ 24` GL run was re-measured on both
> schedules. **All of them still hold, and none of them can see the
> difference** — which is the finding, not a reprieve:
>
> | claim | collapsed | shipped | verdict |
> |---|---|---|---|
> | `extract_quad_mesh` coverage (§P6, §9) | 0.910964 | 0.910909 | holds — analytic, `r²/(r²+1)` at `refuse_radius = 3.2` |
> | `adaptive_tol` changes nothing | equal | equal | holds |
> | median quad anisotropy (gate `< 1.5`) | 1.1099 | 1.0449 | holds, number corrected |
> | lift chamfer ratio, subdiv 4→5→6 (gate `< 0.7`) | 0.497 / 0.508 | 0.505 / 0.493 | holds — measures the icosphere interpolation, not the cones |
> | `quad_grid` χ / coverage / valences | 2 / 1.000000000 / {3,4} | identical | holds |
> | median `quad_isotropy` (gate `< 1.15`) | 1.0783 | 1.0385 | holds, number corrected |
> | `quad_demo` stock-sweep face counts at L=32 | 2137/540/292 | 1112/507/590 | not reproducible as quoted (RNG-dependent sample); the 2–7× span it illustrates is |
> | `quad_demo` `--faces 20000` at L=32 | 19935/20028/19942 | 20063/19987/20048 | holds |
>
> The *only* quantities that moved were `area_distortion` (4.0×) and the
> traced mesh's log-area variance (3.7×). Cell **shape** is forced by the
> developing map and cell **count** by Poincaré–Hopf; cell **area** is the
> one thing cones control, and it was the one thing nothing asserted.

**P8.5 — the Hopf differential** (`hopf.py`). `u_lm = 2ρ_lm/((ℓ−1)(ℓ+2))`, then
`Q = ð²u` — spin 0 → 1 → 2, both diagonal.

> **Pitfall, and a bug that survived several attempts.** Do **not** build `Q` as
> a "real combination" of `±m` spin-2 harmonics. Conjugation flips the spin, so
> the spin-0 recipe does not apply and the result is not a section at all — total
> winding came out `0.0000` where it must be `4`. `ð²` of a **real scalar** gives
> the right complex object automatically.

*Gates:* total winding `±4` **[bundle]**; umbilic count grows as the spectrum
flattens (14 / 36 / 72 / 117 at `ℓ⁻²` / `ℓ⁻¹·⁵` / `ℓ⁻¹` / `ℓ⁻⁰·⁵`, net `+4`
throughout **[bundle]**), fixture F4. Cross-check the umbilic *positions* against
`hopf_q` from `e2e7ab1` on the decoded triangle mesh — two discretisations
agreeing on where the umbilics are is much harder to fake than a winding number.

**P8.6 — the blend control** (`blend.py`). `q = (Q/|Q|)²` is spin-4, exactly the
type of ψ, so the spin weights line up with no conversion. One extra term in the
flow: `psi = solve(psi + tau*(nl + t*q))`.

Measured sweep at `ℓ⁻²` **[bundle]**: `t = 0…15` → 8 cones; `t = 40` → 10, one
valence-5; `t = 100` → 14, three valence-5. Net index `+8` throughout, following
the `n = 8 + 2n₋` ladder.

> **Pitfall.** `t` is a **threshold**, not a smooth dial. Nothing happens below
> `t ≈ 15`: an extra vortex pair costs two cores and alignment must outbid that.
> Sweep it; expect jumps, not drift.

*Gates:* exactly 8 cones at `t = 0`; strictly more at large `t`; net index
invariant at **every** value of `t`, not just the ends.

**P8.7 — do not attempt.** Replacing the mass matrix with the scalar Laplacian to
"target area distortion directly" looks like a one-line improvement. Measured
**2.1× worse** on the objective it was designed for, and worse on frame energy
too **[bundle]**. `crossfield.py` has the reason: cone placement and smooth cross
fields solve the *same* Poisson equation and differ only in the norm (`H⁻¹` vs
`H⁻²`); the error lives in the **relaxation**, not the norm. Ginzburg–Landau
fixes the relaxation and leaves the norm alone. `whyfixed.py` and `emerge.py`
carry the companion result: point relaxation cannot make cones *emerge* at all.

### P9 — The quaternionic Dirac solve ~~[unbuilt]~~ **BUILT**

> `shapegen/geom/dirac_spectral.py`, gates in `tests/test_dirac_spectral.py`.
> Four things this step settled that the plan below had open or wrong:
>
> 1. **`c = 1`**, not 2 (§2.4 left it open and said to pin it empirically).
>    Measured by amplitude ratio of the conformal factor against
>    `geom/dirac.py`: **1.0003**, flat across a 10× amplitude sweep.
> 2. **§2.5 is wrong.** It predicts `γ_mesh ≈ γ_spec²`. Measured,
>    `γ_mesh/γ_spec` is constant to 3% across an 8× amplitude sweep while
>    `γ_mesh/γ_spec²` swings 60×. The relation is **linear**, with a
>    mesh-dependent constant scaling like 1/N (0.0128 at subdiv 4, 0.0033 at 5)
>    — which is the mass normalisation §2.5 hedged about, but the *power* was
>    the error.
> 3. **A singular value, not an eigenvalue.** The operator is not Hermitian and
>    its eigenvalues go imaginary once `ℓ+½ > ρ`; the smallest singular value
>    is the right object, and it also matches `dirac.py`'s Gram formulation.
> 4. **It does not place cones better — and that is not a verdict on P9.**
>    By `field.area_distortion` (a metric that never forms `μ`, validated
>    against the traced mesh to 2.5% and against the cube's known optimum),
>    the exact potential is worse in 5 of 7 cases. **But the control shows
>    that says nothing about the Dirac solve.** Feed the flow `φ_ref =
>    ½·log(dA_X/dA_S)` — the conformal factor of `codec.decode`'s *own*
>    surface, the referee's own geometry, with no spectral Dirac anywhere in
>    it — and it scores *identically* to `method="dirac"` (0.30451, 0.33301,
>    0.15758, 0.17620 to five digits) and loses to the linearisation in
>    exactly the same cases. Ground truth scores the same. The pipeline is
>    **not monotone in the accuracy of the potential**: a 50/50 blend beats
>    *both* endpoints by 1.61× at L=12 amp=4.
>
> 5. **And the measurement lacks the resolution for a 1.17× claim anyway.**
>    Within-method spread across anneal schedules exceeds the linear-vs-dirac
>    gap in 5 of 7 cases (up to 4.2×); jittering the cones 1° moves the score
>    9% and *improves* the median, so the Ginzburg–Landau answer is not even
>    a local optimum of the metric; the `cones_from_field` readout grid alone
>    moves it 6%.
>
> 6. **A real bug, found and fixed: `Ls = L − ½` under-resolved the Galerkin
>    system.** The spinor is fine there (power above the band is 2.8e-7), but
>    `ρ` couples degree ℓ to ℓ±L, so the top rows lose their partners and the
>    truncation residual lands on the highest spinor degrees — which `|λ|²`
>    then deposits at degree exactly `L`, the top of the connection's band and
>    the mode `ð` weights hardest. The ℓ=L coefficient came back 6–9× too
>    large. `POTENTIAL_BAND_EXTRA = 4` fixes it: relative error of `A = i·ðφ`
>    0.0869 → 0.0046 at L=12. It did **not** change the verdict (<1%), which
>    is itself the evidence for point 4.
>
> **CORRECTION, 2026-09-21 — points 4 and 5 above were measured under a
> collapsed anneal, and the verdict they record is wrong.** Both were taken
> at `levels = (6.0, 3.0, 2.0)`, which sits entirely above
> `field.epsilon_ceiling` at `L ≤ 16` (see the P8.4 note above): the flow
> returned `field.linear_start`, so the potential entered only through `C`'s
> smallest eigenvector and "5 of 7 cases" was a comparison of two
> eigenproblems, not of two anneals. The wording above is left as written.
> What replaces it:
>
> **Method.** 12 runs per potential per case — 8 in-window anneal schedules
> plus 4 perturbed starts — scored by `field.area_distortion` on the
> geometric mean, before and after `quad/refine.py`. The noise floor quoted
> with every number is the standard error of that geometric mean, as a ratio
> factor; the raw schedule-to-schedule spread is 1.03–1.39×, which
> refinement cuts to 1.02–1.10×. 13 cases at L = 12/16/24/32.
>
> **`dirac` ÷ `linear`, geometric mean of 12 runs (raw / after refinement):**
>
> | amplitude | L=12 | L=16 | L=24 | L=32 |
> |---|---|---|---|---|
> | 1 | 0.991 / 1.000 | 0.977 / 0.997 | 1.026 / 0.997 | 0.985 / 0.983 |
> | 2 | 0.958 / 0.970 | 1.000 / 1.002 | 1.014 / 1.009 | — |
> | 4 | 1.121 / 1.157 | 1.164 / 1.034 | 0.983 / 0.993 | 1.006 / 0.992 |
> | 8 | **0.477 / 0.420** | **0.532 / 0.420** | **0.380 / 0.443** | — |
>
> Floors: `se` 1.003–1.03 on almost every cell, 1.07–1.08 on the two
> amplitude-8 `dirac` entries, whose own spread is genuinely wider.
>
> * **At amplitude ≤ 2 there is nothing to resolve.** Every entry is within
>   4% of 1.0 and the *sign is not consistent across bandlimit* — `dirac`
>   ahead at L=12/16, behind at L=24 — so even where a 1–3% gap exceeds the
>   standard error it is not a property of the potential.
> * **The amplitude-4 loss is an `L ≤ 16` artefact.** It reverses at L=24
>   (0.983 raw, 0.993 refined), vanishes at L=32 (1.006 / 0.992), and
>   refinement alone halves it at L=16 (1.164 → 1.034). Nothing about "the
>   linearisation wins at amplitude 4" survives raising the bandlimit —
>   which matters, because L=32 is the target and nothing in this area had
>   ever been tested above L=16.
> * **At amplitude 8 the exact potential wins decisively, and it is the
>   largest effect in this whole comparison.** 2.1–2.6× on the raw flow and
>   2.3–2.4× after refinement, consistent at all three bandlimits, tens of
>   standard errors clear of the floor.
> * **The traced mesh agrees with the sampled metric everywhere**, which is
>   the headline this should be read on (log-area variance of the quad mesh
>   the cones actually produce, at pinned cell size ≈ 6000 faces, 4
>   schedules, after refinement): `dirac`/`linear` = **1.004** (L16 amp 1),
>   **1.088** (L16 amp 4), **0.998** (L24 amp 4), **0.464** (L12 amp 8),
>   **0.464** (L16 amp 8), **0.472** (L24 amp 8). Traced floor `se`
>   1.003–1.14. Every one of the six tracks its sampled counterpart to
>   within 5%, including the amplitude-4 reversal between L=16 and L=24.
> * **The control holds throughout, which is what says the effect is real.**
>   `φ_ref = ½·log(dA_X/dA_S)` from `codec.decode`'s own surface — ground
>   truth, no spectral Dirac in it — tracks `dirac` to within 0.1–6% in
>   every cell of the table above (0.477/0.477, 0.532/0.539, 0.380/0.404 at
>   amplitude 8), and to 0.2% on the traced mesh. Wherever the exact
>   potential moves, ground truth moves with it by the same margin.
>
> * **On real faces the question cannot be answered with this apparatus, and
>   finding that out is the more useful result.** Four FaceScape cases (two
>   heads, L=24 and L=32) first read as "no difference" (0.969×, 1.018× at
>   L=24) and then as "`dirac` 2.2× worse" (2.205×, 2.270× at L=32) — at
>   which point **the control stopped tracking** (1.284×, 1.139× against
>   `dirac`'s 2.2×), which is the tripwire the control exists to be. Four
>   candidate causes, separated by measurement:
>
>   1. **Branch point (Gate P9.C) — ruled out.** `min_spinor_modulus` is
>      0.844/0.850 at L=24 and 0.822/0.837 at L=32 on these heads, against
>      0.784 on the `ℓ⁻²` fixture at amplitude 8 where everything behaves.
>      `min|λ|²` is 5.4e-2 and `γ` *falls* from 2.0e-2 at L=24 to 9.1e-3 at
>      L=32 — the spinor solve is better resolved at L=32, not worse.
>   2. **The band — real, and a defect in a shipped constant.**
>      `POTENTIAL_BAND_EXTRA = 4` was tuned at L=12/16 on a smooth fixture.
>      Sweeping it at L=32 on these heads: score 0.1183 / 0.1109 / 0.0738 /
>      0.0746 at `extra` = 2 / 4 / 8 / 12 on one, 0.0991 / 0.1157 / 0.0900 /
>      0.0917 on the other. **8 and 12 agree to 1–2%; 4 does not**, and
>      costs 1.28–1.50×. Documented in `dirac_spectral.spinor_band`, with
>      `max(4, L // 4)` as the indicated rule; not changed, because the new
>      value is characterised at one bandlimit on two faces.
>   3. **The referee — the dominant cause, and it matters well beyond P9.**
>      `area_distortion` decodes on `icosphere(subdiv)`, default 5. On a
>      smooth field that is converged (0.9% at L=16, 7.0% at L=32 between
>      subdiv 5 and 6). On a face it is **21.7% / 23.7% at L=24 and 29.7% /
>      35.1% at L=32**, and not monotone in the mesh (0.158 → 0.057 → 0.081).
>      Nor is the bias common mode: scoring the *same* cone sets against
>      subdiv 6 instead of 5 moves `dirac`/`linear` from 0.983 → 0.831 and
>      1.016 → 0.711 at L=24, and from 2.247 → 0.946 and 2.271 → 0.737 at
>      L=32 — a **sign change**, while the same test on the fixture moves
>      0.432 → 0.453. The 2.2× was an artefact of the referee, and so was
>      the control's failure to track it.
>   4. **Genuine divergence of the two solves — cannot be claimed** while (2)
>      and (3) are live, and (3) alone is larger than the effect.
>
>   **So `area_distortion` at its default `subdiv` does not referee a scanned
>   face**, neither in absolute value nor in the ratios it is used for. That
>   is a limit on the metric, not on P9, and it applies to anything in this
>   repository that scores a face with it. It does not touch the band-limited
>   fixtures the tests use, where the metric is converged and was validated
>   against the traced mesh and the cube's known optimum.
>
> **Verdict.** The original point 5 ("the measurement lacks the resolution")
> was right about the measurement it described and is obsolete as a statement
> about the question: averaging 12 runs puts the floor at 0.3–3%. Point 4's
> control argument is the part that survived, and it got sharper — it now
> certifies a *win* on the fixtures rather than excusing a loss everywhere,
> and when it finally stopped tracking it correctly flagged a broken
> measurement rather than a broken solve.
>
> On **band-limited fixtures**, where the metric is converged, the answer is
> settled: no effect below amplitude 4, and a **2.1–2.6× win for the exact
> potential at amplitude 8**, confirmed on the traced mesh at three
> bandlimits. On **real faces the answer is not determined**, and cannot be
> until the referee is resolved (cause 3) and the band constant is fixed
> (cause 2); both readings taken so far were artefacts. `method="dirac"`
> therefore stays off by default because it is expensive (a dense
> `2632 × 2632` SVD at L=32) and unproven on production inputs — **not**
> because it is worse, which is what the previous two verdicts claimed and
> neither established.
>
> P9 is *vindicated* by the h-refinement row of P9.E, done frame-free:
> the spectral potential's error against the decoded surface falls
> 0.0436 → 0.0109 → 0.0027 → 0.00068 at subdivision 3/4/5/6 — exactly 4.00×
> per refinement, textbook O(h²) — while the linearisation plateaus at 0.084
> and 0.368, i.e. converges to a *different answer*.
>
> P9.E's λ-vs-λ rows are answered only through the conformal factor `|λ|²`,
> which is frame-free. `ψ₁`/`ψ₂` are spin ±½, so comparing them to a
> per-vertex quaternion field needs frame matching that is not done here.

The one thing the bundle does not have. Its README: "Not implemented: the full
quaternionic Dirac solve. Everything here uses the linearisation `μ = 1 + 2ρ`,
which is the largest modelling error in the chain and sits upstream of all of
it."

Split `λ = a + bj`; in the spinor formulation (Kamberov–Pedit–Pinkall;
Konopelchenko–Taimanov),

```
    ethbar psi_1 =  rho psi_2          psi_1 spin +1/2
    eth    psi_2 = -rho psi_1          psi_2 spin -1/2
```

`ð`, `ð̄` diagonal; multiplication by `ρ` a sparse Gaunt coupling; build the
coupling **by transform, not by 3j symbols**. At `L=32` that is 2112 complex
unknowns — dense Hermitian eigensolve, a few seconds. `geom/dirac.py` already
uses shift-invert `eigsh` at `σ=1e-6` with `k > 4` for the mesh problem; the same
recipe applies with multiplicity 2 rather than 4.

Solve `(D − ρ)λ = γλ` for smallest `|γ|`; the realised curvature is `ρ − γ`.

**Gate P9.A — the round sphere.** `ρ_repo ≡ 0` (equivalently `ρ_bundle ≡ 1`) must
give `λ = const`, `γ = 0`, both to `1e-10`, after the same gauge fix.
*This single test pins `c` from §2.4 and every sign and normalisation. Do not
proceed past it.* Its mesh-side twin already exists and passes —
`tests/test_dirac.py::test_constant_spinor_in_kernel` — so a failure here is in
the new code, not in the convention.

**Gate P9.B** — `D − ρ` Hermitian to `1e-10`; catches conjugation errors in the
coupling.

**Gate P9.C — branch points.** `min|λ|` on a fine grid; approaching zero means no
surface exists for that `ρ`. Fail with the location. Expect this on F3 before F1.

**Gate P9.D — convergence in band.** L = 16, 24, 32; spectral for smooth `ρ`.

**Gate P9.E — agreement with `geom/dirac.py`.** Two discretisations of one
continuum problem, on F0/F1/F2:

| quantity | mesh path | spectral path |
|---|---|---|
| residual | `γ_mesh` | `γ_spec²` (§2.5) |
| conformal factor | `\|λ\|²` per vertex | `e^{φ}` sampled there |
| `ρ` recovered | `dirac.rho_from_lambda(V,F,λ)` | `ρ − γ` |

The third row is the strongest: `rho_from_lambda` already inverts the mesh
transform and is tested; feeding it the *spectral* λ must return the ρ that went
in. Run at two mesh resolutions — the gap should shrink as `O(h²)`; flat in `h`
is a bug in the spectral path, growing is a bug in the sampling.

**Fallback.** If P9.A cannot be passed within budget, keep `μ = 1 + 2ρ` and
record it as the dominant modelling error, exactly as the bundle does. Do **not**
ship a Dirac solve that fails P9.A; a wrong spinor corrupts everything downstream
silently. Note that P0–P7 do not wait on this — the mesh pipeline with §2.1's
decode is strictly ahead of the bundle already.


> **Linear potential removed, 2026-09-26.** `quad.field.curvature_potential`
> is now the Dirac solve only; `method="linear"` (`μ = 1 + 2ρ`,
> `φ_lm = −μ_lm/(l(l+1))`), `method="auto"` and `AUTO_RMS_THRESHOLD` are gone,
> and `sink=` is replaced by `field.solve_potential_spinor` + `spinor=`. The
> evidence that retired it, condensed from the deleted docstring:
>
> - **Production amplitude.** Real FaceScape codes sit at `rho_rms` 1.87-1.93;
>   every fixture the linearisation had been judged on sat at 0.15-0.61. At
>   `|ρ| ≈ 12`, `μ = 1 + 2ρ` is not an approximation of anything.
> - **Fixtures, unconfounded.** At amp 8 (rms 1.22), on a subdivision-6
>   referee: linear 12 cones E=0.235 vs dirac 8 cones E=0.100 at L=16, and
>   0.239 vs 0.106 at L=24. That is 2.3x better with a third fewer cones. At amp
>   ≤ 2 the two were a wash (every cell within 4% of 1.0).
> - **Real faces.** Four heads at L=24, taper 1e-3: dirac/linear 0.833 (254),
>   0.866 (338), 0.777 (204), 0.820 (339). With the taper at 2e-3 the margin
>   shrinks (0.990/0.883/0.994/0.902). 8/8 wins, 8 cones in every case.
> - **Convergence.** Against the decoded surface's own conformal factor the
>   Dirac potential converges O(h²): 0.0436/0.0109/0.0027/0.00068 at
>   subdivision 3-6. `linear` plateaus at 0.084 (amp 1) and 0.368 (amp 4),
>   converging to a different answer.
> - **Small amplitude.** The two agree mode by mode up to sign
>   (`φ_dirac/φ_linear` → −1: −0.988, −1.015, −0.965, −1.001 at ℓ = 2,3,4,6).
>
> Tests removed with it: the linear-vs-dirac comparisons and the `auto`
> switch tests in `test_quad_field.py` / `test_dirac_spectral.py`, and
> `test_quad_extract.py::test_adaptive_tol_no_longer_changes_coverage_and_why`.
> That test's exact-equality claim held for linear-potential cones only (1.3e-4
> apart with Dirac cones) and concerned the superseded single-chart
> extractor. Round-sphere operator tests now build the exact zero potential
> directly: the Dirac solve returns φ ≈ 2.5e-14 at ρ = 0, which splits the
> sphere's degenerate eigenspaces.

> **Coupling quadrature made exact, 2026-09-26.** `coupling_matrix` built
> `B* diag(ρw) Bᵀ` on `half_grid(Ls)`, which is exact to degree `2Ls`. The
> integrand `Ȳ_ℓ' ρ Y_ℓ` has degree `2Ls + L`, so at L=32 / Ls=39.5 the top
> ~30 degrees of the coupling aliased. It now runs on
> `dirac_spectral.coupling_grid(Ls, L)`: `⌈(2Ls+L+1)/2⌉` GL nodes and
> `2Ls+L+2` φ samples. `rho_on_grid` samples there, and `coupling_matrix` /
> `dirac_matrix` take `L`. Guarded by
> `test_coupling_quadrature_is_exact`, which also asserts the old grid does
> alias. Solve cost is unchanged (extra 8: 28s / 21s before, 25s / 19s
> after, heads 339 / 254).
>
> Measured on heads 339 / 254 at L=32, `band_taper` t=1e-3. Each head was
> traced once at 20k faces and lifted every way on that one mesh. Route 1 is
> pinned at icosphere(6), whose own 5↔6 dependence is 0.017 / 0.019.
>
>     potential A-error vs decode(ico6)'s conformal factor (ℓ(ℓ+1)-weighted)
>     extra              4       6       8       10      12      16
>     339  aliased     0.253     –     0.089     –     0.036     –
>          exact       0.106   0.067   0.041   0.035   0.033   0.035
>     254  aliased     0.240     –     0.097     –     0.036     –
>          exact       0.100   0.068   0.043   0.036   0.033   0.035
>
>     route 2 lift                 hol       edge res   vs r1(ico6)  p95 iso
>     339  aliased, extra 8     6.66e-3   3.96e-3    0.0062      1.041
>          exact,   extra 8     5.95e-3   3.39e-3    0.0062      1.041
>          exact,   extra 12    3.86e-3   2.08e-3    0.0063      1.040
>          exact,   extra 16    2.29e-3   1.20e-3    0.0063      1.039
>     254  aliased, extra 8     6.23e-3   3.80e-3    0.0067      1.041
>          exact,   extra 8     5.51e-3   3.23e-3    0.0068      1.041
>          exact,   extra 12    3.67e-3   2.02e-3    0.0068      1.040
>          exact,   extra 16    2.26e-3   1.18e-3    0.0069      1.039
>
> Four findings:
>
> 1. **At the shipped `extra = 8`, the exact grid halves the potential error**
>    (0.089 → 0.041, 0.097 → 0.043). The cones do not move: the same 8, at
>    identical positions on the readout grid. γ moves by < 0.3%.
> 2. **`potential_band_extra` was partly compensating for aliasing.** The
>    aliasing residue lands at the top of the *spinor* band, so extra padding
>    pushed it away from ℓ ≤ L. With the exact grid, extra 4 is already 2.4x
>    better than it was. The ~0.033 floor at extra 10–16 is the ico6
>    referee's own error, not the solve. `max(4, L//4)` is left unchanged: on
>    the exact grid, extra 8 sits 25% above that floor, and extra 12 reaches
>    it at 1.8x the solve time (21s → 38s).
> 3. **The remaining P7.A holonomy is spinor-band truncation.** On the exact
>    grid it falls steadily with the band: 5.9e-3 → 3.9e-3 → 2.3e-3 at extra
>    8 / 12 / 16. It did not before, because aliasing at the band top put a
>    floor under it. That explains "tracks band-limit content, not γ".
> 4. **Procrustes vs route 1 does not move** (0.0062 / 0.0068 at every
>    setting). That is route 1's own mesh error, 0.017 between subdivisions
>    5 and 6, not route 2's.

---

## 6. Milestones

| # | deliverable | steps | gate summary |
|---|---|---|---|
| M0 | corpus in place, fixtures build | P0, P1 | 8 cones, index sum 2, ρ̂ matches `rho.npy` |
| M1 | **mesh pipeline end to end, on the real decode** | P2–P7 route 1 | quads out; anisotropy < 1.2; chamfer vs `codec.decode` |
| M2 | transforms, integer + half-integer, at L=32 | P8.1, P8.2 | round trip `1e-10` at band 32 |
| M3 | spectral cones, emergent count and indices, blend | P8.3–P8.6 | index sum 2 at every `t` |
| M4 | **Dirac solve passing the round-sphere test** | P9 | P9.A, then P9.E |
| M5 | mesh-free end to end | P8 + P9 + P4–P7 | no triangulation downstream of the encode |

> **M5, 2026-09-21: the mesh-free path exists and is reachable, but is not the
> default.** `lift_route2` removes the last triangulation downstream of the
> encode (§P7a) and is 4.3x faster than route 1, so the *capability* gate is
> met. It is opt-in because at production γ (≈ 2e-2 on FaceScape heads at
> L=32) Gate P7.A's holonomy is 6.2e-2 and the least-squares solve absorbs a
> 2.5e-1 edge residual, costing 1.7-2.3x the log-area variance and 20x the
> excess median anisotropy against route 1. **M5 is gated on γ, not on the
> lift**: nothing further is needed in §P7: what would move it is a smaller γ
> (a better `rho_hat`, or a re-measurement pass), since at γ ≈ 3e-5 route 2
> already agrees with route 1 to 0.4%.
>
> **Correction, 2026-09-26:** "gated on γ" was wrong. The γ-dependence was a
> first-order dictionary error (§P7a, correction block). With the fixed map,
> route 2 matches route 1 to 0.6% at production γ, so M5's quality gate is met
> on the two heads measured. What remains is validation breadth, not γ.

**M1 is reachable now and is the first thing to ship.** It is the bundle's own
end-to-end pipeline with its weakest stage replaced by this repository's
strongest one, and every gate in it has a number already.

M2's transform decision (§P8.1) and M4 are the two places the port can stall;
neither blocks M1.

---

## 7. Risks

| risk | likelihood | mitigation |
|---|---|---|
| missing `lll()` / `fastfail.py` / `cvpmod.py` | **certain** | naive rounding is complete in `s5_grid.py`; treat LLL as a later improvement, §P5.3 |
| naive transform cannot reach L=32 | **high** — measured from both sides | §P8.1 is a decision to take, not a risk to watch |
| half-integer path has no oracle | **high** | Gates in P8.2 are the only protection; do not weaken them |
| Dirac sign conventions, `c` of §2.4 | **high** | P9.A has a known answer; the mesh-side twin already passes |
| `γ` large on real codes | **certain** on F3 (2.7e-2) | report it; P7.A quantifies the consequence |
| branch points (λ vanishes) | medium | P9.C; fail with the location |
| chart leaves the far pole unmeshed | **certain** | report `coverage`; mask every comparison, §9 |
| `Q` built as a "real combination" | **high** | winding must be 4, not 0 — P8.5 |
| quads silently dropped by the defect filter | medium | `QuadResult.rejected` |
| `nsub=14` ported by transcription | medium | P6.3 — the source's default is 14× the needed work |
| `wavefront.py`'s `np.prod` overflow ported | medium | P6.2 — take `simpler2.py`'s version |
| zero extraction under-resolves dense umbilics | medium | at `ℓ⁻⁰·⁵`, 117 umbilics gave net `+5` not `+4` **[bundle]**; refine until exact |

---

## 8. Do not re-attempt

Each of these was measured in the bundle and each looks plausible. The numbers
are in §1's tables.

- Disc-based series tracing (`arch5.py`) — slower than the ODE, 10% of steps
  wrong.
- Taylor coefficients by FFT on a circle (`fftcoef.py`) — branch continuation
  does not close.
- Cut-and-transition global solve (`cutsolve.py`) — edge classification near
  cones never made correct.
- Swapping the mass matrix for the scalar Laplacian (§P8.7) — 2.1× worse on its
  own objective.
- Integer `(i,j)` vertex keys (§P6.5) — assumes trivial holonomy.
- Any branch tracking of `w'` (§P6.1) — there is no global branch to track.
- **Decoding on the quad mesh's own triangulation** instead of an icosphere.
  Tempting — 2.3× faster (27.2s vs 62.6s at 20k faces) and it deletes the
  interpolation, the base-mesh sizing, and the coupling between mesh density
  and lift resolution. But a conformal mesh cannot be uniform (§10), and the
  Dirac solve pays for it. Gate P9.E's ρ round-trip, which is the
  resolution-normalised measure (raw γ scales like 1/N and cannot compare
  meshes of different size): `icosphere(5)` 0.0314 at 10242 v, `icosphere(6)`
  0.0100 at 40962 v, **quad mesh 0.0515 at 20037 v** — twice ico5's vertices
  and worse than it. That also settles the 16% divergence between the two
  routes: the quad mesh is the one that is wrong. Pinned in
  `test_dirac_spectral.py`.

---

## 9. Chart coverage — CLOSED, and the reasoning below was wrong

> **Resolved by `shapegen/quad/grid.py`.** Coverage is `1.000000000` and
> `V − E + F = 2` on every configuration tried — the cube fixture at four cell
> sizes, the P0 corpus, and the Ginzburg–Landau cones at three spectra. The
> output is a closed quad mesh of the whole sphere, so the masking this section
> demands is no longer needed and `eval/metrics.py`'s closed-surface metrics
> apply again.
>
> **The two things this section got wrong**, both worth keeping because they
> cost a lot of effort in the wrong direction:
>
> 1. *"A chart's excluded pole always carries an implicit extra singularity
>    (`Σaₖ ≠ 0` over any finite cone set forces one there)."* It does not.
>    `ω = W dz` with `W = ∏(z−Cₖ)^{−idxₖ}` has order `−m−2` at infinity for
>    `W ~ z^m`, and Gauss–Bonnet fixes `Σ idxₖ = 2` on a genus-0 field, so
>    `m = −2` and the order is **0** — regular *and* non-vanishing. Verified:
>    `|W|` in the `ζ = 1/z` chart is `1.0` at `ζ = 0` and stays above `0.44`
>    over the whole disk. What blows up towards the pole is the chart
>    *coordinate*, so a fixed developed step becomes an ever larger step in
>    `z` — a coordinate artifact, removed by using each chart only on its own
>    hemisphere.
>
> 2. *"0.832 → 0.911 as vertices are spent … coverage only climbs
>    asymptotically toward 100%, never quite arriving."* Those numbers are real
>    but they are not asymptotic. `r²/(r²+1)` at `extract.py`'s own hard-coded
>    `refuse_radius = 3.2` is `0.9110`, which is the measured ceiling to three
>    decimals. The trace was stopping at the radius it was told to stop at.
>
> **Why the symptom looked real anyway.** `cones_from_field` returned every
> index negated, so the mesh-free path's cones summed to `−2`, not `+2`. At
> `−2` the form genuinely *does* acquire a pole of order 4 at the far pole and
> the local cell really does collapse there. The extra singularity was
> self-inflicted, and the general claim was inferred from it. With the sign
> corrected the same configuration traces to `χ = 2`; with it negated
> `grid.py` returns `χ = −3240`. Both numbers are pinned in
> `tests/test_quad_grid.py`.
>
> The rest of this section is left as written, and is now historical.

## 9a. Chart coverage was a regression, not a footnote (historical)

The bundle lists chart coverage as open: one stereographic chart leaves the far
pole unmeshed. In a standalone prototype that is a fair scoping call. **Here it
is a loss of parity** — `codec.decode` returns a closed genus-0 surface;
`quad_mesh` returns a disc.

1. **Every comparison against the reference decode must be masked** by the
   charted region, or the chamfer reports the missing pole rather than the
   quality of the quads. Unmasked it will look catastrophic and mean nothing.
2. **`coverage` is a first-class output**, measured where the trace is bounded
   (§P6.4), not estimated afterwards.
3. **Closed-surface metrics do not apply.** `eval/metrics.py`'s `genus` and
   `flip_fraction` assume a closed triangle mesh. Extend them or leave them out;
   do not run them and read the output.

Until coverage is closed, this is an **additional** decode path, not a
replacement, and the README should say so if it mentions it at all.

**Substantially narrowed, not closed, by `extract_quad_mesh(..., adaptive_tol=True)`.**
The ~80-85% single-chart ceiling wasn't `refuse_radius` (raising it from 3.2 to
20 didn't move it) — a chart's excluded pole always carries an implicit extra
singularity (`Σaₖ ≠ 0` over any finite cone set forces one there, wherever the
pole is put — a standard fact about meromorphic functions on the sphere, not a
property of this specific cone layout), so the local step size shrinks without
bound approaching it. The fixed `tol = tol_factor * s` merge tolerance
eventually exceeds that shrinking step and starts misreading real progress as
"arrived back at the start" (measured directly: steps at the reached frontier
were already the same order of magnitude as `tol`, not orders away).
`adaptive_tol` scales the threshold to each step's own size instead. Measured
on a real face at L=32: 0.832 -> 0.846/0.876/0.896/0.911 coverage at
max_vertices 2000/5000/10000/20000, with *zero* closure-defect rejections at
every budget (vs 11% at the fixed tolerance) — the resulting cap is a smooth
converging pattern of shrinking quads, not the self-overlapping mess the fixed
tolerance produced trying to occupy the same space. It does not reach 100%: as
cells keep shrinking approaching the pole, coverage only climbs asymptotically
toward it, never quite arriving, so `converged` reads `False` by design at any
practical vertex budget — a real diminishing-returns curve, not a bug to chase
away. A two-chart atlas (build a second chart from the antipodal pole, trim to
a hemisphere split, weld the seam) was tried first and measured to gain far
less (82.6% for two charts, both capped by the *same* mechanism at their own,
different poles) for real added complexity (a new module, non-trivial
vertex-welding, seam quality bookkeeping) — abandoned in favor of this fix,
which is a few lines in the existing single-chart extraction and needs no
stitching at all. Masking against `coverage` (point 1, above) still applies:
this raises the ceiling, it doesn't remove the need for it.

---

## 10. Deliberately out of scope

- **Chart coverage**, as above — recorded as a known regression, not a neutral
  limitation.
- **Equal-area quads.** Conformal maps cannot preserve shape and size both, so
  this is inherent, not a defect to fix; grading cell size across the surface is
  the remedy. Now measured over the *whole* sphere rather than a trimmed chart:
  63× interdecile solid-angle spread on the P0 corpus (the bundle's 26× was on
  its ~85% patch). What *is* testable, and is tested, is that each face's area
  equals `s²` times the conformal factor at its centre —
  `grid.developed_cell_area`, median ratio `1.000000` on the cube by m=24 — and
  that the spread vanishes where symmetry forces it to: the 2×2-per-face cube
  has 24 faces of exactly equal area.
- **Automatic choice of `t`.** Exposed as a parameter; choosing it from a
  distortion budget is a separate optimisation.
- **Choosing the cone count directly.** Set indirectly by `t` and by `ρ`'s
  spectrum. Prescribing `n` and solving for a field with exactly that many
  singularities is a different, harder problem.
- **Faces.** ρ is ~99% unrepresentable on scanned faces and position-SH is 6.2×
  better at equal cost; F3 is a stress test, not a target.
- **Any torch, float32 or MPS involvement.** `CONVENTIONS.md`'s precision split
  puts all of this path in float64 NumPy/SciPy on the CPU.
