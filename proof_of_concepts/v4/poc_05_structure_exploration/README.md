# POC-05 (v4): Does the engine build spatial structure?

**Status**: active
**Pillar**: SEC

## Thesis

Exploratory, not predictive. The question is whether any accessible regime of the canonical
pipeline produces spatial geometry in the mass field — and to record honestly which levers did
nothing.

## Result — contrast and connectivity are decoupled

The density can be driven harder than a real cosmic web's. The topology does not move at all.

| | range explored | cv (contrast) | percolation |
|---|---|---|---|
| baseline | — | 1.33 | 0.012 |
| loss rate `eta` | 0 → 3.2 | 0.80 → 1.60 | 0.019 → 0.013 |
| clamp `field_scale` | 20 → 5000 | 0.83 → 0.76 | — |
| grid / ticks | 128×32 → 128×128, 3k → 12k | 0.82 → 1.01 | 0.019 |
| gradient transport | 0 → 50× | 1.33 → **3.895** | 0.014 |
| curl transport | 0 → 50× | 1.33 → 2.28 | 0.010 |

*(synthetic web 1.000 in one component · white noise 0.003 across 1394)*

**Contrast spans 5×. Percolation never leaves 0.007–0.019.** cv 3.895 is nearly twice a real
cosmic web's ~2.0, at percolation 0.014. Roughly forty runs, four grid configurations, three
tick counts past the 8450 convergence point, both Helmholtz halves of the EM force, transport
spanning 100× — nothing moved the topology.

This rules out a **class** of intervention, not a list of settings: every fix tried was some
version of *push the density around differently*. Full write-up in
[`journals/2026-08-14_contrast-and-connectivity-are-decoupled.md`](journals/2026-08-14_contrast-and-connectivity-are-decoupled.md).

The mechanical reason: every force here is the gradient of a scalar potential. Gravity is a
Poisson solve; `ChargeDynamicsOperator` builds a vector force and adds only its **divergence**
to `dE_dt`, discarding a curl 89% the magnitude of what it keeps. Gradient flow has point
attractors — it makes blobs, and a filament is a line. Restoring the curl is null, because the
curl velocity is five times *less* coherent than white noise: the Poisson 1/k² weighting puts
the smooth power in the gradient by construction, so the curl is the high-k residue, and `Q` is
a derivative of an `E` field that is already noise.

---

## Supporting detail

**The engine builds large-scale power and no web geometry.** Two findings, and they point in
opposite directions.

Coherent power fraction, 3000 ticks, 5 seeds, against a white-noise sampling distribution
measured on the same grid (n = 40, 0.0722 ± 0.0059):

| configuration | ξ_u | coherent fraction | excess |
|---|---|---|---|
| RBF, qp = 0.02 (default) | 0.6453 ± 0.0091 | 0.0991 ± 0.0031 | +4.6σ |
| PAC balance, qp = 0.02 | 0.6877 ± 0.0097 | 0.1009 ± 0.0066 | +4.9σ |
| RBF, qp = 0.00 | 0.7327 ± 0.0098 | 0.1449 ± 0.0086 | +12.3σ |
| **PAC balance, qp = 0.00** | **0.7778 ± 0.0147** | **0.1457 ± 0.0105** | **+12.5σ** |

Twice the white-noise share of low-k power with quantum pressure off, still growing at
t = 3000, and above the buried-cosine reference of 0.116.

Against exp_09's web criterion (voids > 0.3, CV > 1.0, filaments > 0.05), the same fields:

| | void fraction | density CV | `is_web` |
|---|---|---|---|
| **exp_09 particle web** | **0.50** | **~2.0** | **yes** |
| white noise \|N(0,1)\| | 0.053 | 0.741 | no |
| PAC, qp = 0.02 | 0.241 | 0.661 | no |
| PAC, qp = 0.00 | 0.081 | 0.487 | no |

**Density contrast is below white noise's in every configuration.** Rendering the isolated
low-k component shows why: it is a field of random blobs, morphologically indistinguishable
from low-passed white noise. Excess large-scale power is not web geometry.

### qp = 0.30 scores `is_web = True`, and it is a checkerboard

Void 0.605 — better than exp_09's 0.50 — and CV 1.273, passing all three of exp_09's
conditions. Rendered (`results/qp030_iswebtrue.png`), it is alternating zero and bright cells
at single-cell scale: C(1) = **−0.358**, ξ_u = 0.465 (below the 0.632 noise floor), coherent
fraction 0.017 (below noise).

exp_09's criterion is purely statistical — no scale, no connectivity — and a lattice
checkerboard satisfies it trivially: half the cells sit at zero and read as voids, the bimodal
distribution gives CV > 1, the bright half are overdense. **That failure mode cannot arise in a
particle substrate**, which is why exp_09 never guarded against it.

`web_metrics` now carries a fourth condition — ξ above the white-noise floor — with the
checkerboard as a permanent selftest case. exp_09's thresholds are untouched.

### Two things fall out

- **Voids and large-scale power move apart.** qp = 0.02 maximises voids (0.241); qp = 0.00
  maximises low-k power (0.145). Nothing in the engine produces both.
- **A web needs contrast the engine suppresses by construction.** `NormalizationOperator`
  tanh-clamps E and I, hard-caps M, and spreads a uniform PAC correction over every cell.
  exp_09's web came from unbounded local concentration held by entropy pressure.

The engine clumps. It does not web. That is the founding fact of Milestone 16 — and it is
*not* what the 2026-08-13 journals recorded.

## Why this POC was reconstructed

It ran on 2026-08-13 as **journals only**. No script, no results, no calibrated estimator. The
numbers it produced — correlation length 1.00 cell, spectral tilt 9.09× vs 2.23×, a preferred
wavelength moving 23.2 → 10.5 → 2.1 with quantum pressure — founded Milestone 16 and could not
be reproduced by anyone, including their author.

Two of those numbers also contradicted each other. A field with a preferred wavelength near 10
cells cannot have a correlation length of 1 cell.

## What the reconstruction changed

**1. ξ is reported per direction.** The manifold is 128 (periodic circumference) × 32
(bounded strip width), with the twist coupling `u + π` to `v → 1 − v`. A circular FFT along the
bounded axis imposes false wraparound across the strip edges. Measured on a field that is
genuinely smooth along `v`: circular reads 3.82, the correct unbiased linear estimator reads
8.04 — a 2.1× underestimate. An isotropic radial average over a 4:1 grid compounds it.

**2. `coherent_fraction` is reported beside ξ.** This is the discriminator the original panel
lacked. A correlation length at the floor is consistent with two very different fields:

| field | ξ_u | coherent fraction |
|---|---|---|
| white noise | 0.635 | 0.072 |
| λ = 16 cosine buried under 3× noise | 0.660 | **0.116** |

Both read at the floor on ξ. Only the coherent fraction separates *absent* from *buried*.
Without it, "the field is noise" was not something the original panel could establish.

**3. Every number is reported against a white-noise control on the same grid**, so "at the
floor" is a comparison rather than an assertion.

**4. The estimators carry analytically derived selftests.** Not tuned ones — white noise gives
ξ = 1 − 1/e; a Gaussian of width σ gives ξ = 2σ; a cosine of wavelength λ gives
ξ = arccos(1/e)·λ/2π. Two smoothing widths are tested because the *ratio* is the check that
matters: a pinned estimator can match one value by luck, never the scaling. See
`proof_of_concepts/v4/structure.py`.

## The 1.00 that founded a milestone

The original table reported exactly 1.00 in every cell, at every timestep, under both balance
operators. That is the **integer-lag quantum**, not a measurement: an estimator that returns
integer lags cannot report anything between 0 and 1, so it returns 1.00 for any field whose
neighbours are uncorrelated *and* for any field with weak but real correlation.

The calibrated floor is **1 − 1/e = 0.632**, from linear interpolation between C(0) = 1 and
C(1) = 0.

**And the conclusion drawn from it was wrong.** "The field is noise" was what an instrument
that quantises to integer lags, averages isotropically over a 4:1 anisotropic grid, and
reports no low-k power fraction is bound to say about a field carrying a +12σ large-scale
component. The founding fact needed correcting, not just the number behind it — see
`dawn-field-theory/experiments/milestones/milestone16/journals/2026-08-14_refounding.md`.

## Growth peak ≠ power peak

The "quantum pressure sets a preferred scale" claim was about where the **growth ratio**
P(k,late)/P(k,early) peaks, not where the power itself sits. Those are different fields. A
growth ratio can peak at λ ≈ 10 while the power spectrum stays white, if the growing component
starts far below the noise floor. This panel reports both, separately, which is what dissolves
the apparent contradiction.

## Running it

```
python proof_of_concepts/v4/poc_05_structure_exploration/scripts/exp_01_structure_panel.py
python proof_of_concepts/v4/structure.py          # calibration must PASS first
```

`spectral_tilt`'s exact definition is a choice made in `structure.py`, not one recovered — the
journals never recorded how the 9.09× and 2.23× were computed, so **those two figures are not
reproducible under any convention** and should not be compared against this panel's tilts.

## Known gap

`poc_04_entropy_reduction` is also journals-only. Its entropy-reduction finding has no
committed script. Not addressed here.

## Feeds

Milestone 16 (`dawn-field-theory/experiments/milestones/milestone16/`) — this POC's re-measured
result is that milestone's founding fact.
