# Plan — flow matching over ρ̂ SH coefficients, for smoothed faces

**Written 2026-09-07; implemented the same day (`f28f7f9`).** Sections carry
inline **RESULT** notes where a measurement replaced a guess. Two of the plan's
central assumptions turned out wrong and are marked as such — the cost wall does
not exist, and the sample-count rule it relies on is necessary but not
sufficient. Read those before the prose around them.

The generative model produces a **band-limited spherical-harmonic coefficient
vector of ρ**, directly. No PCA stage: the model's output *is* ρ̂, and decoding
is `codec.decode(ρ̂, template, faces)`.

---

## 1. Why this is the hard version, stated up front

On faces, ρ̂ is measurably a worse representation than PCA over positions:

| representation | coefficients | held-out shape error |
|---|---:|---:|
| PCA over smoothed positions, k=64 | 64 | **0.0058** |
| PCA over smoothed positions, k=32 | 32 | 0.0089 |
| ρ̂, **direct**, the head's own sphere, any L | 81–1089 | ~0.084 |
| ρ̂, **direct**, one shared template (what this plan uses) | 169 (L=12) | **0.103** |
| ρ̂, adjoint, L=32, the head's *own* cMCF sphere | 1089 | 0.0209 |

**RESULT — and a correction.** With `direct`, the shared-template L-sweep is
0.133 (L=4), 0.109 (L=8), **0.103 (L=12)**, 0.107 (L=16), degrading to 0.121 at
L=40 and **0.348** with no band limit at all. I read that as "more bandwidth does
not help" and generalised it to the representation. **That was wrong.**

The same heads, same shared template, with the **adjoint** encoder:

| L | direct | adjoint |
|---|---|---|
| 8 | 0.09201 | **0.05616** |
| 12 | 0.09044 | **0.04345** |
| 16 | 0.08753 | **0.03512** |

**Monotone in L, and 2.5x better than `direct` at L=16** — every one of those
iteration-capped, so they are upper bounds. Bandwidth *does* buy accuracy. The
non-monotonicity belongs to `direct`, whose bias lives in the high bands, and not
to ρ̂. The ceiling in this table is an artefact of the encoder this plan chose for
cost reasons, not a property of the representation.

The reason to do it anyway is real: **ρ̂ needs no correspondence.** PCA over
positions only exists because FaceScape ships registered meshes. ρ̂ works on any
genus-0 surface, which is what the rest of this repo is for. Faces are then a
test bed for a pipeline meant to outlive them, not the target the pipeline is
tuned to.

---

## 2. One blocker that turns out not to exist

`codec.py` and `CONVENTIONS.md` both warn that ρ̂ is not rotation-invariant, and
that has been treated all along as the thing standing between this codec and a
generative model. **For registered data on a shared template it is false.**

Measured: a random rotation of the input mesh changes ρ̂ by

- **3.7e-15** (direct encoder)
- **6.9e-10** (adjoint encoder)

i.e. machine precision. The reason is structural: `ρ = H·e^u − 1`, `H` is
rotation-invariant, and `e^u` is recovered from edge *lengths*, also invariant.
Rotating the shape does not touch the template→shape correspondence.

**RESULT.** Confirmed inside the built pipeline on real heads (9.7e-13) and
pinned by `tests/test_models.py::test_rho_is_rotation_invariant`. The alignment
stage is gone from the ρ path.

**Consequences.** No Procrustes alignment stage — ρ cannot see it, so it would be
pure cost. No Wigner-D augmentation. No equivariant architecture. The warning in
`codec.py` still stands for **unregistered** data, where each shape gets its own
cMCF sphere that is itself only defined up to rotation; it should be narrowed to
say so.

---

## 3. The blocker that does exist: degeneracy, not just dimension count

Measured on this pipeline (340 training heads, PCA codes, flow matching), the
sample dispersion — `max |std − 1|` on whitened codes, want 0 — against samples
per dimension:

| N/k | 85.0 | 42.5 | 21.2 | 10.6 | 5.3 |
|---|---|---|---|---|---|
| `max abs(std−1)` | **0.018** | 0.174 | 0.164 | 0.229 | **0.329** |

It is not a budget problem. Tripling training steps moved 0.329 → 0.256. Going
from 50 to 1000 Euler steps moved 0.329 → 0.298. Neither is the lever.

Naively applied to ρ̂ at 446 subjects, this is fatal:

| bandwidth | K = (L+1)² | N/K at 446 |
|---|---:|---:|
| L=8 | 81 | 5.5 |
| L=12 | 169 | 2.6 |
| L=24 | 625 | 0.7 |

But **N/K is the wrong diagnostic for ρ̂**, and getting that wrong would send the
work in the wrong direction. The PCA sweep varied `k` over *whitened,
decorrelated* coordinates, where every direction genuinely carries variance. Raw
ρ̂ vectors do not: 446 heads span at most 445 dimensions of a 169-dimensional
space, and their intrinsic dimensionality is the same face-identity manifold PCA
finds at k≈32. So the ρ̂ data distribution is **near-singular** — almost all of
its variance sits in a low-dimensional subset of the 169.

Flow matching to a near-singular target is ill-posed rather than merely
data-starved: the velocity field has to collapse probability onto a
measure-zero set, which forces it to blow up as `t → 1`. The symptom looks the
same (shrunken samples) but the fix is different, and §4 is about that difference.

**RESULT — this is where the plan was most wrong, in both directions.**

Check 0a measured the participation ratio of the ρ̂ code matrix:

| subjects | participation ratio | directions for 99% | N / d_eff |
|---|---|---|---|
| 30 | 2.3 | 23 of 169 | 13.2 |
| 360 | **9.1** | **74 of 169** | **39.7** |

So the target *is* near-singular as predicted — 74 of 169 directions carry
everything — but `d_eff` **grows with the sample count**, because 30 heads simply
do not reveal the manifold's structure. Any threshold computed from a small pilot
underestimates it.

And more importantly: **N/d_eff ≥ 20 is necessary but not sufficient.** At 39.7,
comfortably past the rule, the dispersion check still fails (max |std gap| 0.590).
The rule was derived on *PCA* codes, which are whitened and isotropic; ρ̂ is
whitened band-wise and keeps a long tail of low-variance directions inside each
band, so it is a harder target at the same nominal ratio. Do not reuse the
threshold across representations — measure the dispersion directly.

---

## 4. Five mitigations, all SH-native

Ordered by expected value per unit of work.

**(a) Use every expression, not just `1_neutral`.** — *implemented
(`expressions=`), not yet exercised at full scale.*
FaceScape ships **20 expressions per subject**. 446 × 20 ≈ **8 900 meshes**
against the 446 currently used — 20× the data, taking N/K to ~53 at L=12, above
the threshold in §3. Expressions within a subject are correlated so the effective
sample size is smaller than 8 900, but this is by far the largest single lever
and it costs only encode time. It changes what is modelled — identity *and*
expression rather than identity alone — which is a fair reading of "generate
faces". Caveat from `registered.py`: the mouth cap is invented geometry whose
boundary moves with expression, so it enters the code looking like measured
shape. Consider excluding the most extreme mouth expressions.

**(b) A variance floor on the target.** — *implemented, and the effective lever.*
Convolve the data with a small isotropic Gaussian before training — add fresh
noise to `z1` each time it is drawn (`train_flow(sigma=...)`).

**RESULT**, 360 training heads at L=12, dispersion `max |std gap|` (want < 0.25):

| σ | t_end | max std gap |
|---|---|---|
| 0.0 | 1.0 | 0.632 |
| 0.0 | 0.95 | 0.622 |
| 0.05 | 1.0 | 0.528 |
| **0.15** | **0.95** | **0.416** |

Monotone in σ and worth **34%** at σ=0.15 — but still short of the threshold, so it
mitigates rather than fixes.

**(c) Stop the ODE at `t = 1 − ε`.** — *implemented; on its own it does nothing.*
`sample_flow(t_end=...)`. **RESULT: 0.632 → 0.622 alone** — within noise. The
premise was that the field blows up at the endpoint; measured, it does not. The
flow is **underfitting**, not diverging (the loss was still descending at 20 000
steps), and truncation does not help underfitting. Keep it as a companion to (b),
not as a lever of its own.

**(d) Coarse-to-fine over bands.**
The SH-native answer, and the one worth building if (a)–(c) are not enough.
Generate bands `ℓ ≤ 4` with one field, then bands `5–8` conditioned on them, then
`9–12`. Each stage is small — band ℓ has only `2ℓ+1` coefficients — so each
conditional density is estimated where data supports it, and the cascade matches
how the spectrum actually behaves. Conditioning a velocity field is extra input
channels, nothing structural.

**(e) Band-wise whitening, not per-coefficient.**
Already implemented in `RhoBasis`. The `2ℓ+1` coefficients of a band share a
physical scale, so pooling them estimates that scale from `2ℓ+1`× as much data;
per-coefficient `std` is noisy when N is not ≫ K. Without *some* whitening the
flow's unweighted MSE fits the largest-magnitude bands and ignores the rest —
the same failure `band_precond` exists to fix in the encoder.

---

## 5. Data path

Differs from the position pipeline in two ways, both forced by ρ.

```
load (all roots, all expressions)
  → cap        REQUIRED: cMCF, the Dirac eigensolve and the cotan-Poisson
               integration are all undefined with boundary. FaceScape ships open
               (4 loops: neck 96, mouth 96, eyes 91/94).
  → normalise  centre + unit mean radius.
  → smooth     implicit Laplacian, (M + tL)V' = MV, t=1e-3 (~0.98%, about 2 mm).
               Matters MORE here than for positions: H is a second derivative, so
               it amplifies scan noise by ~l^2. Measured on a raw FaceScape head,
               two independent curvature estimators disagree by 92.7% (the same
               test on a procedural bottle: 2.36%) -- H on raw scans is noise.
  → NO align   rho cannot see rotation (section 2).
```

`bottlegen/data/faces.py` (built) does load / normalise / smooth and skips
capping. It needs a `cap=True` path — reverting a choice made for the position
model — plus an `expressions=` argument.

---

## 6. Encoder: `direct`, and what that costs

**This plan uses the `direct` encoder** — now the codec default. The adjoint takes
**321–376 s** on one 26 317-vertex head, which puts §4(a)'s 8 900 meshes at **two
to three months** single-threaded.

**RESULT — the cost wall does not exist.** Measured end to end on all 400
subjects via `scripts/encode_faces.py`:

```
loaded + smoothed 400 heads (26 321 verts)   74 s
template cMCF sphere                         29 s   (once, for the whole set)
rho fields (400, 26321)                      45 s   = 0.11 s/mesh
```

**148 s total.** Extrapolated, §4(a)'s 8 900 meshes are well under an hour. No
decimation, no process pool needed — `n_jobs` is implemented (13.3 s → 4.4 s on 8
bottles at 4 workers) but is not on the critical path for `direct`.

What made it cheap is not just the encoder: **ρ_direct does not depend on L**, only
the projection does, so `RhoBasis.rho_fields()` computes the per-mesh field once
and every bandwidth in a sweep reuses it. That turns an L-sweep from `N x |L|`
edge-least-squares solves into `N`.

The expensive direction turned out to be the **decode**, not the encode: one ρ
decode is an eigensolve on a `4n x 4n` matrix, **5.6 s** for a 26 321-vertex head.
The §8 checks are budgeted accordingly.

Two other reasons it is the right choice here, independent of cost:

- **Its codes are comparable across meshes** (per-band 7.7% at L=4, 12.4% at L=12
  between triangulations of one surface). The adjoint's are not — the same figures
  are 7.4% and **46.0%**, because its fit absorbs the discretisation it was fitted
  against. For a generative model over a dataset that is a defect, not a detail.
- It has no iteration cap to police, so no code in the set is a silently
  budget-limited upper bound.

**The price, stated plainly.** On a smoothed head `direct` sits at ~0.084 against
the adjoint's 0.021 — a **4× accuracy ceiling** the generative model inherits and
cannot beat. And on faces specifically ρ_direct comes out **10–30× too small**
(‖ρ_direct‖ 71 against the least-squares optimal 774), because `e^u`
exponentiates the error in `u`. **That is a bug-shaped result, not a fundamental
limit, and §8 check 0b should try to fix it before accepting the ceiling.**

**Choosing L is not "as high as affordable".** For `direct`, L is a regularisation
parameter: measured on a bottle, 0.115 (L=4), 0.054 (L=8), 0.064 (L=12), 0.0641
(L=24), and decoding with no band limit at all also gives 0.0641. The curve
converges to the error of reproducing ρ_direct faithfully; low L wins by
accidentally truncating the estimator's bias, which lives in the high bands
(per-band error 0.15–0.33× for ℓ ≤ 6, 1.3–11× for ℓ ≥ 8). γ rises monotonically
with bandwidth, confirming it. **The optimum moves with mesh resolution** (L=8 at
subdiv 3–4, L=12 at subdiv 5), so it must be chosen by validation on the actual
face meshes, not inherited from the bottle numbers.

Still worth doing:

1. **Cache.** Encode once to an `.npz` keyed by (template, L, method). Never
   re-encode inside a training loop.
2. **Parallelise if needed.** `encode_dataset` has an `n_jobs` hook left
   deliberately unimplemented; a process pool would help, but at 0.81 s/mesh it is
   no longer on the critical path.
3. **Keep the adjoint reachable** for a held-out accuracy reference — encoding 40
   heads with it costs ~4 hours and tells you exactly how much the ceiling in
   §1 is costing the finished model.

---

## 7. Components

| file | state | change |
|---|---|---|
| `bottlegen/models/flow.py` | built | none — it takes an `(N,K)` array and never sees a mesh |
| `bottlegen/models/basis.py` | built | `RhoBasis` exists (shared sphere, band-wise whitening). Add the §4(b) variance floor and §4(c) truncated sampling |
| `bottlegen/data/faces.py` | built | add `cap=True` and `expressions=` |
| `bottlegen/codec.py` | **done** | `direct` default; `sphere` decides sphere policy; `n_jobs` pool |
| `scripts/encode_faces.py` | **done** | one-time encode → cached `.npz` |
| `scripts/train_faces.py` | **done** | `--basis rho\|pca`, `--cache`, `--sigma`, `--t-end`, §8 checks |

---

## 8. Dry run — what "representing properly" means

A flow-matching loss floor is set by the irreducible variance of its conditional
target, not by fit quality. **The loss curve is not evidence.** Every check is
about the data path or the sample distribution.

**Check 0a — run this before building anything.** Encode 200 heads, band-whiten,
and take the singular-value spectrum of the ρ̂ matrix. Report the participation
ratio / how many directions hold 99% of the variance. That number, not `(L+1)²`,
is the effective dimension the flow has to learn, and it decides whether §4(a)
alone is enough or §4(d) is required. *Everything downstream is contingent on it.*

**Check 0b — try to fix ρ_direct on faces first.** ‖ρ_direct‖ is 10–30× below the
least-squares optimal ρ for the same correspondence, and that gap is what sets the
0.084 ceiling. The suspect is `e^u`: `u_edge_lsq` is exact to 3e-16 on conformal
input, but a FaceScape head's conformal factor spans ~2 000×, and ρ multiplies by
`exp(u)`. Check whether the `u` residual is large where ρ is most wrong. If a few
hours here move 0.084 materially, everything downstream gets better for free; if
not, the ceiling is real and §9's success criterion stands.

Then, from the existing script:

1. **Round trip at full rank** — `decode(encode(X)) ≈ X`. Catches basis, whitening
   and reshape errors. Currently passes at 6e-16.
2. **Held-out reconstruction vs L** — must reproduce §1's table. A different
   number means this is not the pipeline that produced the measurement.
3. **Smoothing helps** — guards the stage that carries the result.
4. **Codes are whitened** — mean ≈ 0, std ≈ 1 band-wise.
5. **Samples match the code moments** — `max abs(std−1) < 0.25`. Under-dispersion
   is *the* characteristic failure; read it against §3.
6. **Samples are not memorised** — reference is how far a **held-out** face sits
   from the training set. Train-to-train nearest neighbour is the wrong reference
   and was used here at first: it produces a ratio of 0.17–0.39 at every k,
   including k=4 where the model is essentially perfect. A criterion that fails a
   known-good model is not a criterion.
7. **Beats a diagonal Gaussian.** Read a narrow win with suspicion: under-dispersed
   samples cluster near the mean, and held-out faces are nearer the mean than the
   tails, so shrinkage *flatters* this metric.

ρ-specific additions:

8. **ρ̂ rotation-invariance** — assert the 1e-9 of §2 as a regression gate. If a
   future change breaks it, everything above silently degrades.
9. **γ on generated codes.** The Dirac residual falls out of every decode for
   free and is a genuine off-manifold score — γ ≈ 0 means the sample is exactly
   realisable as an immersion. Compare generated γ against training γ. **No other
   representation in this repo hands you a validity metric you did not invent**,
   and it is the strongest argument for ρ̂ over PCA.
10. **Decoded samples are valid meshes** — genus 0, finite, no NaN, Willmore
    energy in the training range.

---

## 8b. What the dry run actually found

Run: 400 subjects (360 train / 40 held out), L=12, σ=0.15, t_end=0.95.
**7/10 checks pass.** Passing: basis agrees with the codec, reconstruction sane,
codes whitened band-wise, **ρ̂ rotation-invariant (9.7e-13)**, flow beats a diagonal
Gaussian, **generated γ matches the data's** (1.28e-2 against 1.39e-2 — samples are
as realisable as real heads), decoded samples are valid genus-0 meshes.

Failing: sample dispersion (§4b), the memorisation check (which is coupled to it —
see below), and "smoothing improves the curve", which flipped sign between the
40-head and 400-head runs and is measured over only 12 decodes; treat it as noise
until re-run with a bigger budget.

**The decisive result is visual, not numeric.** Rendering real → ρ̂ round trip →
generated side by side: **the round trip already loses the ears and most identity
detail.** The generated heads are about as good as the round trips. So the flow is
*not* the bottleneck — the round-trip ceiling is.

**But that ceiling is the `direct` encoder's, not ρ̂'s** (see §1). The adjoint
reaches 0.035 at L=16 against direct's 0.088, monotonically in L. Everything below
about "accepting a blurrier model" was written against the wrong ceiling.

**A check-design flaw worth carrying forward.** Checks 5 and 6 failed and passed
together across every configuration, which is the signature of one measurement
wearing two hats: under-dispersed samples cluster toward the mean, where training
points are denser, so they score "close to train" for a reason unrelated to
copying. Check 6 is now a **tail** statistic — what fraction of samples sit nearer
the training set than 95% of real held-out heads do — because memorisation is a
tail event and a mean cannot see it.

---

## 9. Expected outcome, and when to stop

Expect ρ̂ + flow to land **worse on faces than PCA + Gaussian**, by roughly the
factor in §1. That is not a reason to abandon it — the point is a pipeline that
does not need a solved correspondence — but it does mean the success criterion is
"produces valid, diverse, face-like heads", not "beats the 3DMM".

Stop and rescope if:

- **Check 0** says the effective dimension exceeds ~1/20 of the sample count even
  with all expressions, *and* §4(d) does not bring it under. The density is then
  not estimable at this data scale and no amount of architecture fixes it.
- Check 0b finds no fix and 0.084 is confirmed as the ceiling *and* the samples
  are visibly not face-like — a 4× worse encoder may still generate recognisable
  heads, and that is the thing to look at before concluding.
- Generated γ sits far outside the training range — the model is producing
  coefficient vectors that are not curvature half-densities of any surface, and
  band-limiting alone will not repair that.

The fallback is not defeat: neutral-only at L=8 with §4(b)+(c), accepting a
blurrier model, still demonstrates the full path end to end and is the thing to
generalise to unregistered shapes.

**RESULT.** The full path runs, samples are valid genus-0 heads with matching γ,
and they look like heads. They also look like *the same* head — but the cause is
now identified and it is fixable.

**The next move is a better encoder, not a better generative model, and not more
data.** The adjoint reaches 0.035 at L=16 where `direct` sits at 0.088, and unlike
`direct` it keeps improving with bandwidth. The plan chose `direct` because the
adjoint was budgeted at 321-376 s/head; that is still true (≈450 s at L=16), which
puts 400 subjects at **50 hours** — long, but a one-off, embarrassingly parallel
(`n_jobs` is implemented), and cached forever afterwards. At 8 cores it is an
overnight job. That is the single highest-value thing left.

Only after that is it worth asking whether §4(a)'s extra data widens the code
distribution, or whether ρ̂-on-a-shared-template is simply the wrong ceiling for
faces.

**A methodological note worth keeping.** `direct`'s non-monotonicity in L is a
property of a *biased estimator*, and for several turns I treated it as a property
of the *representation*. The tell was available the whole time and I did not chase
it: the adjoint beat `direct` 2x at fixed L, which already implies a better ρ
exists at that bandwidth. Never infer a limit of the representation from the
behaviour of one encoder.
