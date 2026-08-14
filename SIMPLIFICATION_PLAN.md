# Simplification plan: reduce to a one-shot ρ ↔ shape codec

**Goal.** Strip the repo to the ρ encode/decode path, so the next step is training a
generative model over ρ̂ without navigating a second, competing representation.

**Nothing here has been executed.** Every file listed under *Delete* is committed,
so all of it is recoverable with `git checkout <sha> -- <path>`.

---

## 1. Evidence

Transitive import closure from `bottlegen.geom.encode` reaches **9 of 41**
`bottlegen` modules. The one non-obvious result: `spectral/swsh.py` (279 lines) is
imported by `conformal.py`, but *only* `mesh_to_sigma` uses it. Nothing on the ρ
path touches the spin-weighted machinery.

| conformal.py function | needs swsh / curvature? | on the ρ path? |
|---|---|---|
| `spherical_conformal_map` | no | **yes** |
| `mobius_normalize` | no | **yes** |
| `ball_automorphism` | no | yes (used by `mobius_normalize`) |
| `conformal_factor` | no | no |
| `hopf_on_sphere` | `hopf_q`, `tangent_frames`, `vertex_one_ring` | no |
| `mesh_to_sigma` | `swsh` | no |
| `gauge_fix` | no | no |

---

## 2. Keep

| module | lines | role |
|---|---|---|
| `spectral/quaternion.py` | 65 | quaternion algebra |
| `geom/mesh.py` | 159 | `Mesh`, `icosphere`, `triangle_areas` |
| `geom/operators.py` | 111 | cotan Laplacian, vertex areas/normals, edge weights |
| `geom/curvature.py` | → ~40 | **trim to `mean_curvature`** |
| `geom/conformal.py` | → ~110 | **trim to `spherical_conformal_map`, `mobius_normalize`, `ball_automorphism`** |
| `geom/dirac.py` | 445 | decode: operator, spin transform, edge rule, Poisson |
| `geom/dirac_adjoint.py` | 305 | adjoint of the decode |
| `geom/encode.py` | 353 | `encode_sh` / `decode_sh` / basis / preconditioner |
| `geom/spin.py` | 173 | λ-route initialiser — **optional**, see §5 |
| `data/dataset.py` | 143 | bottle generator |
| `eval/metrics.py` | 155 | `flip_fraction`, `genus`, `willmore_energy` — decode validation |
| `scripts/willmore_flow.py` | 215 | flow driver |
| `scripts/flow_trajectories.py` | 216 | reverse-process training data |
| `scripts/eval_sh_roundtrip.py` + `plot_` | 247 | the accuracy benchmark |

## 3. Delete

| path | lines | why |
|---|---|---|
| `bottlegen/models/` (8 files) | ~830 | VAE, rect-flow, cold-diffusion, DiT, DiffusionNet — all built on **σ̂** |
| `bottlegen/flow/` (4 files) | ~250 | schedule/state/willmore for the σ̂ stack |
| `train/` (3 files) | ~220 | trainers for those models |
| `bottlegen/decode/` (4 files) | ~300 | quadmesh, reconstruct, sample_points — output-side |
| `bottlegen/spectral/swsh.py`, `grids.py` | ~330 | spin-weighted SH; only `mesh_to_sigma` needs it |
| `bottlegen/geom/transport.py`, `diffusion_ops.py` | ~160 | cross-field / DiffusionNet support |
| `bottlegen/eval/viz.py` | 130 | plotting for the old eval |
| `bottlegen/data/classprior.py` | — | σ̂ class prior |
| `scripts/eval_all.py`, `eval_decode.py`, `quad_demo.py`, `sample_*.py` | ~600 | drive the deleted stack |

**Tests to delete** (each imports only deleted modules):
`test_swsh.py`, `test_state.py`, `test_flow.py`, `test_physics.py`,
`test_quadmesh.py`, `test_rect_flow.py`, `test_vae.py`, `test_simple.py`,
`test_decode.py`, `test_dataset.py` (imports `bottlegen.flow`).

**Tests to keep:** `test_quaternion.py`, `test_curvature.py`, `test_conformal.py`,
`test_dirac.py`, `test_dirac_adjoint.py`, `test_encode.py`, `test_spin.py`.

⚠️ `test_curvature.py` and `test_conformal.py` cover functions being trimmed away —
they need pruning in the same commit, not after.

Total: **~2,700 lines, 30 modules.**

---

## 4. The `bottlegen/codec.py` façade

One import for all future training code, so it never reaches into `dirac` or
`dirac_adjoint` directly. Thin — it composes what exists, adds no maths.

```python
"""The ρ codec: mesh ↔ band-limited SH coefficients of ρ."""

def base_sphere(mesh, *, method="cmcf", **kw) -> np.ndarray:
    """Sphere parameterisation on the mesh's own connectivity, Möbius-normalised.

    method="cmcf"     conformalized mean-curvature flow      (geom.conformal)
    method="willmore" gradual conformal Willmore flow        (scripts.willmore_flow)

    Measured on a bottle at subdiv 3: qc distortion 0.0194 (cmcf) vs 0.0170
    (willmore). Either works — because the encoder optimises *shape* error, the
    parameterisation's conformal quality is largely compensated (the round trip
    lands ~6× below the qc figure).
    """

def encode(mesh, L, *, sphere=None, maxiter=200, **kw) -> EncodeResult:
    """mesh → ρ̂ ∈ ℝ^((L+1)²). Wraps geom.encode.encode_sh.

    Returns coeffs, sphere, err, niter, gnorm. ALWAYS check niter/gnorm: a run at
    the cap is an upper bound, not a converged encode.
    """

def decode(coeffs, sphere, faces) -> np.ndarray:
    """ρ̂ → vertex positions. Wraps geom.encode.decode_sh.

    Determined only up to a similarity; compare with `shape_error`, never with a
    raw vertex distance.
    """

def shape_error(target, X) -> float:
    """Relative Procrustes error — the metric the encoder optimises."""

def encode_dataset(meshes, L, *, n_jobs=1) -> np.ndarray:
    """(N, (L+1)²) coefficient matrix. Encoding is ~2.5 min/mesh and
    embarrassingly parallel across meshes; this is the parallelisation point."""
```

`EncodeResult` should be a small dataclass rather than the current dict, so
`niter`/`gnorm` are visible in an IDE and can't be silently dropped.

---

## 5. Judgement calls

**`geom/spin.py` — keep, but know it's optional.** Measured: zero-init reaches the
*identical* optimum (0.01232 / 0.00550 to five digits), costing 48 vs 56 iterations
at L=8 and 155 vs 106 at L=16. So it's worth ~30% at moderate bandwidth and nothing
at low. Keep it as the default initialiser; it is not load-bearing, and if the
`lambda_from_correspondence` BFS ever causes trouble, delete it without ceremony.

**`eval/metrics.py` — keep.** `flip_fraction` and `genus` are how you check a decode
produced a valid surface, which matters more once a *network* is generating ρ̂ and
can emit values no encoder would.

**`data/dataset.py` — keep, expect to replace.** Fine for developing the codec;
a real generative model needs real bottles.

---

## 6. Execution order

Each step leaves the repo green, so a bisect stays meaningful:

1. Delete `train/` and the dead `scripts/` — nothing imports them.
2. Delete `models/`, `flow/`, `decode/` and their tests together (they are
   mutually referential; splitting breaks imports mid-way).
3. Delete `spectral/swsh.py`, `grids.py`, `test_swsh.py` **and** trim
   `conformal.mesh_to_sigma` / `hopf_on_sphere` in the same commit — these are
   circular, so they cannot be split.
4. Trim `curvature.py` to `mean_curvature`; prune `test_curvature.py` to match.
5. Delete `geom/transport.py`, `diffusion_ops.py`, `eval/viz.py`,
   `data/classprior.py`.
6. Add `bottlegen/codec.py` plus `tests/test_codec.py` (round trip through the
   façade, and that `decode(encode(m))` reproduces the reported error).
7. Update `CONVENTIONS.md`: drop the σ̂/Hopf and cross-field sections, which will
   describe code that no longer exists.

Verify after each: `python -m pytest -q`, and after step 6 re-run
`python -m scripts.eval_sh_roundtrip --subdiv 4 --levels 2,10,20` and confirm
`0.1119 / 0.0065 / 0.0049` are unchanged.

---

## 7. Blockers before training — not fixed by cleanup

**Rotation gauge is unresolved, and it is the real one.** Möbius normalisation kills
the ℓ=1 translation freedom; nothing pins rotation (`CONVENTIONS.md` says as much:
"rotation is handled separately / left to canonicalization"). Two bottles differing
only by a rotation encode to entirely different ρ̂. A network trained on this burns
capacity learning SO(3) rather than shape. Options: canonicalise orientation in the
dataset (PCA frame, or align the ℓ=2 moment), or use a rotation-equivariant
architecture over SH bands. **Decide this before generating a dataset**, because it
determines what the data has to look like.

**Encoding cost.** ~2.5 min/mesh at subdiv 3, L=12, maxiter=200. Parallelise across
meshes (`encode_dataset`), and note that the inverse-iteration eigensolve — still
unbuilt — is worth a further ~1.5×.

**High-L rows are budget-limited, not converged.** At L≥20 the optimiser stops at
the cap with a falling gradient; the tail follows `err(k) = 0.00139 + 0.0927·k^-1.06`,
so `maxiter≈800` reaches within 5% of the limit. Irrelevant at L=12; relevant if the
latent bandwidth grows.

**The trajectory data has a Möbius-snap outlier.** In `flow_trajectories.py` the
final `V_{T-1} → S` step absorbs the gauge fix and is ~3× a median flow step. Drop
that sample or treat it separately when training.
