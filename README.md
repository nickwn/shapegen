# shapegen

A **generic 3D shape generator**: turn any genus-0 surface into a fixed-length
vector of numbers, learn a distribution over those vectors, and sample new
shapes. Currently targeted at **faces**; nothing in the representation is
face-specific.

```
mesh  ──encode──▶  ρ̂ ∈ ℝ^((L+1)²)  ──generative model──▶  ρ̂'  ──decode──▶  mesh
```

---

## The idea

Most 3D generative models fight the fact that meshes have no canonical
representation — different vertex counts, different connectivity, no natural
ordering. The usual escapes (voxels, point clouds, neural fields) each give
something up.

This takes a different route, from **discrete conformal geometry**. A theorem of
Crane–Pinkall–Schröder says that deforming a surface by a quaternion field `λ`,

```
df̃ = λ̄ · df · λ
```

produces a valid new immersion exactly when `λ` solves a Dirac equation
`Dλ = ρλ`, where **ρ is a single scalar function** — the change in mean-curvature
half-density, `ρ = H·e^u − 1`.

So a shape is one scalar field on the sphere. Band-limit it to spherical
harmonics and it is a **fixed-length real vector**: 169 numbers at L=12. That is
an easy generative target — no graph structure, no permutation ambiguity, just a
vector.

## Why this representation

- **No correspondence needed.** Works on unregistered meshes of any
  triangulation. This is the whole point, and what distinguishes it from a 3DMM.
- **Every vector decodes to a surface.** The eigensolve always returns a `λ`, so
  the model *cannot* emit something that isn't a mesh. Most 3D representations
  can.
- **A free validity score.** `γ`, the Dirac residual, falls out of every decode.
  `γ ≈ 0` means the code is exactly realisable as an immersion; large `γ` means
  it is off-manifold. You do not have to invent a quality metric.
- **Rotation-invariant** on a shared template — measured at 4e-15. `H` is
  rotation-invariant and `e^u` comes from edge lengths, so rotating the input
  leaves ρ̂ untouched.

## Status

Honest, and measured. Relative Procrustes shape error, round trip:

| shape | encoder | coefficients | error |
|---|---|---:|---:|
| bottle, subdiv 5 | direct, own sphere | 169 (L=12) | **0.0128** |
| bottle, subdiv 3 | adjoint, own sphere | 1089 (L=32) | **0.0041** |
| bottle, subdiv 3 | adjoint, shared template | 1089 (L=32) | 0.0078 |
| smoothed head | adjoint, own sphere | 1089 (L=32) | 0.0209 |
| smoothed head | *PCA over positions* | 64 | *0.0058* |

**The codec works well on smooth genus-0 shapes and is mediocre on faces.** For
registered face data specifically, PCA over vertex positions beats it — 3.6×
better from 17× fewer numbers. That comparison is fair and it is in this README on
purpose. The counter-argument is that PCA needs a solved correspondence and ρ̂
does not, so ρ̂ is the path that generalises off this dataset.

The generative half is **not built yet**. See [SH_FLOW_PLAN.md](SH_FLOW_PLAN.md).

## Quickstart

```python
from shapegen import codec
from shapegen.data.dataset import demo_bottle_meshes

mesh   = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
sphere = codec.base_sphere(mesh)                          # cMCF + Möbius gauge fix
res    = codec.encode(mesh, L=12, sphere=sphere, method="direct")
X      = codec.decode(res.coeffs, sphere, mesh.faces)

print(res.coeffs.shape)                                   # (169,)
print(codec.shape_error(mesh.vertices, X))                # 0.0644
```

Two encoders:

- **`direct`** (default) — `ρ = H·e^u − 1` read straight off the geometry. Closed
  form, 0.81 s at 2 562 vertices. Its codes are comparable across meshes, which
  is what a dataset encoder needs.
- **`adjoint`** — optimises decoded shape error through the adjoint of the decode.
  Several times more accurate on a *fixed* mesh (0.0127 against 0.0644 at L=12)
  and ~26× the cost. Its fit absorbs the discretisation it was fitted against, so
  its codes are **not** portable — per-band coefficients differ 46% at L=12
  between two triangulations of one surface, against `direct`'s 12%.

**`L` is a regularisation parameter for `direct`, not a fidelity knob** — and this
is a fact about that estimator, *not* about ρ̂. Measured on a bottle: 0.115 (L=4),
**0.054 (L=8)**, 0.064 (L=12), 0.0641 (L=24), and decoding with no band limit at
all also gives 0.0641. The curve converges to the error of reproducing ρ_direct
faithfully; low L wins by accidentally truncating the estimator's bias, which lives
in the high bands.

**With the adjoint, error falls monotonically with bandwidth**, as you would
expect: on a FaceScape head against a shared template, 0.056 (L=8) → 0.043 (L=12)
→ 0.035 (L=16), against `direct`'s 0.092 / 0.090 / 0.088. So if accuracy matters
more than encode cost, use `adjoint` **and** raise L; with `direct`, choose L by
validation and expect an optimum around 8-16.

For a dataset, `codec.encode_dataset(meshes, L, sphere=template)` encodes
everything against **one shared sphere**, which is required for the codes to mean
the same thing — and required outright for generation, where there is no mesh to
recover a sphere from.

## How it works

```
mesh ──cMCF──▶ sphere ──Möbius normalise──▶ base sphere      parameterise
     ρ = H·e^u − 1  on that sphere                           measure
     project onto real SH (Galerkin, area-weighted)          band-limit
     ─────────────────────────────────────────────────────
     A(ρ)λ = γMλ, smallest γ                                 Dirac eigensolve
     ẽ = λ̄ e λ                                               transform edges
     cotan-Poisson solve                                     integrate
```

The adjoint encoder differentiates that entire chain — including the eigensolve,
whose adjoint is a singular solve requiring deflation of a 4-dimensional null
space — and optimises ρ̂ against decoded shape error directly.

## Layout

| path | what |
|---|---|
| `shapegen/codec.py` | the public API — `encode`, `decode`, `base_sphere`, `shape_error` |
| `shapegen/geom/dirac.py` | Dirac operator, spin transform, edge integration |
| `shapegen/geom/dirac_adjoint.py` | adjoint of the decode |
| `shapegen/geom/encode.py` | SH basis, projection, the optimising encoder |
| `shapegen/geom/conformal.py` | cMCF, Möbius normalisation, Gu–Yau |
| `shapegen/data/` | demo bottles, registered-dataset loaders (FaceScape) |
| `shapegen/models/` | flow matching over code vectors *(new, unlanded)* |
| `scripts/` | evaluation sweeps, Willmore flow, dataset builds |

## Things that are true and unobvious

Each of these was measured after being assumed otherwise, and each is recorded in
the code where it matters.

- **Quasi-conformal distortion does not bound reconstruction error.** A shared
  template with qc = 0.0665 reconstructs at 0.0078 — 8.6× *below* it. qc is a
  pointwise anisotropy average; reconstruction error is optimised over all
  conformal reparameterisations, and by uniformisation that family is every
  genus-0 surface. Do not tune a parameterisation on qc.
- **On scanned data, `H` is noise.** Two independent curvature estimators
  disagree by **92.7%** on a raw FaceScape head and by **2.36%** on a procedural
  bottle. `H` is a second derivative, so it amplifies scan noise by ~ℓ². Smooth
  before encoding.
- **ρ's flat high-ℓ spectrum is real signal, not estimator noise.** Swapping in
  three alternative curvature estimators — including one 578× more accurate
  pointwise on a sphere — changed the round trip by ≤2%.
- **Check `(L+1)²` against your vertex count** before reading any spectrum. At
  subdiv 3 a 642-vertex mesh against 625 coefficients makes a band-limit residual
  ≈0 by construction, and it looks like a result.

## A note on the name

`shapegen` is historical — the project began generating bottles, which are a
convenient non-star-shaped genus-0 test case. The goal is generic shapes, and the
current target is faces. Renaming is pending.

## Requirements

NumPy and SciPy for the codec; PyTorch only for `shapegen/models/`. Tests:
`python -m pytest tests/`.
