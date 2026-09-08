# Plan — flow matching over ρ̂ SH coefficients, for smoothed faces

**Written 2026-09-07. Nothing here has been executed yet** — the sections marked
*built* exist as uncommitted working files; everything else is proposed. If you
are reading this after the work landed, check `git log` before trusting it.

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
| ρ̂, **direct** (the encoder this plan uses), any L | 81–1089 | **~0.084** |
| ρ̂, adjoint, L=32, the head's *own* cMCF sphere | 1089 | 0.0209 |

A *shared* template sphere — which generation requires — will make the ρ̂ row
worse still by an amount §8 measures.

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

**This must be measured before anything else is built** — see §8, check 0a.

---

## 4. Five mitigations, all SH-native

Ordered by expected value per unit of work.

**(a) Use every expression, not just `1_neutral`.**
FaceScape ships **20 expressions per subject**. 446 × 20 ≈ **8 900 meshes**
against the 446 currently used — 20× the data, taking N/K to ~53 at L=12, above
the threshold in §3. Expressions within a subject are correlated so the effective
sample size is smaller than 8 900, but this is by far the largest single lever
and it costs only encode time. It changes what is modelled — identity *and*
expression rather than identity alone — which is a fair reading of "generate
faces". Caveat from `registered.py`: the mouth cap is invented geometry whose
boundary moves with expression, so it enters the code looking like measured
shape. Consider excluding the most extreme mouth expressions.

**(b) A variance floor on the target.**
Convolve the data with a small isotropic Gaussian (σ ≈ 0.05 in whitened units)
before training — i.e. add fresh noise to `z1` each time it is drawn. This makes
the target non-singular, removes the endpoint stiffness, and costs a blur of
known size. It is what diffusion models do implicitly and is one line.

**(c) Stop the ODE at `t = 1 − ε`.**
The degeneracy only bites at the endpoint. Integrating to `1 − ε` (ε ≈ 0.02)
samples a slightly blurred distribution and avoids the blow-up. One line, and it
composes with (b).

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
to three months** single-threaded. `direct` is closed-form: **0.81 s** at 2 562
vertices, so the whole dataset encodes in hours, on one core, today. No decimation
required to make the plan exist.

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
| `bottlegen/codec.py` | exists | `direct` is now the default, and `sphere` decides sphere policy rather than `method` — a shared template with `direct` is what this plan needs and used to be rejected |
| `scripts/encode_faces.py` | **new** | one-time encode → cached `.npz` |
| `scripts/train_faces.py` | built | switch basis; swap the checks per §8 |

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
