# 2026-08-17 (late): the corpus had the tool — shuffle nulls

exp_06 established that the **network** conducts along filaments. The obvious next question is
whether the **clock field** exploits that, or whether exp_06 only measured a property of the
graph. Two attempts produced large, confident, meaningless numbers. The third worked, and the
method came out of the corpus rather than from me.

---

## The question

`LocalTime`'s viscosity diffuses τ over a **distance ball** — the same operator shape that made
exp_03 unable to answer its own question. So the substrate still carried the defect exp_06 had
just exposed. Fix: add bounded-degree coupling (`time_coupling="knn"`), then ask whether τ's
non-local content follows the web.

## Attempt 1 — ball vs k-NN. Confounded by kernel size.

Prediction stated in advance: if the ball was hiding conduction, graph coupling should carry
*more* non-local information.

It carried **less** — partial correlation +0.646 against the ball's +0.746. Prediction refused.
But rather than write that up, the sweep:

| coupling | degree | partial |
|---|---|---|
| knn k=6 | 7.6 | +0.085 |
| knn k=12 | 14.9 | +0.112 |
| knn k=24 | 29.6 | +0.180 |
| knn k=48 | 58.8 | +0.275 |
| ball | 237.0 | +0.399 |

Monotone, roughly logarithmic, climbing straight toward the ball. **τ's non-local content is a
function of coupling degree** — of how far information diffuses per step. Ball-vs-knn compares
two smoothing radii, not two topologies. Both the prediction and the test were wrong.

That the sweep is monotone is itself worth keeping: it says exp_02's ν > 0 result scales with
kernel size, which nobody had measured.

## Attempt 2 — web vs uniform. Confounded by control-variable variance.

Matched coupling, matched k, positions frozen. The **control scored higher**: uniform +0.360
against the web's +0.088, −9.31σ.

Not physics. **Partial correlation is not comparable across systems whose control variable has
different spread.** In a uniform box δ barely varies, so regressing it out removes almost
nothing and leaves τ's diffused structure intact. In a web δ is large and dominates τ, so
regressing it out strips most of the signal. The raw column corroborates: corr(τ, Φ) is 0.80–0.83
in the web against 0.60–0.66 uniform, so the two arms were never on the same footing.

Two designs, two large sigma values, neither meaning anything.

## The corpus already solved this

Peter pointed me at earlier work on utility and Euclidean distance:
`experiments/studies/euclidean_distance_validation`. Its **experiment_25** faced the mirror
image of my problem — *"your metrics correlate because they are redundant, not because of
conservation"* — and answered it by **shuffling**:

> Metrics are independent — r drops from 0.79 to −0.29 when shuffled.
> Shuffling breaks correlation — ξ drops 0.87 → 0.17 when structure is removed.
> Random baseline over 100 trials, z-score against it.

`experiment_08_null_hypothesis_tests.py` runs five distinct nulls rather than one.

A **degree-preserving edge shuffle** holds everything fixed except the wiring: same particles,
positions, density field, δ and its variance, degree per node, edge count, kernel size. Only
*which* particles are connected changes. Both of my confounds become impossible by construction
— and **no partial correlation is needed**, which removes the statistic that had been generating
the false signals in the first place.

## The result

Degree preservation asserted on every run.

| configuration | real | shuffled | gap | z |
|---|---|---|---|---|
| web | +0.708 | +0.091 | **+0.617** | +23.5 |
| uniform | +0.783 | +0.095 | **+0.688** | +41.6 |

Rewiring collapses τ's agreement with the long-range potential from 0.71 to 0.09. **87% of it is
carried by the wiring.** That is a real and large effect.

But the shuffle destroys **two** things together — the web's filamentary topology *and* spatial
locality in general, since random rewiring creates long-range edges and τ homogenizes. So a
web-only null cannot attribute the effect to the web.

Running the same null inside a **uniform box** separates them. Each arm is compared against a
shuffle of *itself*, so the gaps are comparable even though the raw correlations are not:

**EXCESS = −0.071 (−4.27σ).** The web's gap is *smaller* than uniform's.

**τ follows spatial locality. The web's topology adds nothing on top.**

## This does not contradict exp_06

Different questions, and both hold:

- **exp_06** — the *network* conducts preferentially through dense regions. Effective
  resistance, +9.14σ, degree-matched, k-independent. A property of the graph.
- **exp_07** — the *clock field's* agreement with Φ does not exploit that. It needs the coupling
  to be local, and nothing more.

The graph has conductive structure; τ isn't using it. Which sharpens the open question rather
than closing it: either the mean-relaxation form washes the advantage out — testable by swapping
in the conduction Laplacian, which exp_06 showed behaves differently — or **Φ is the wrong
observable**, because a smooth long-range potential may not care about network structure at all.
That second possibility is the more interesting one and it has not been tested.

## What I take from it

Three attempts; the first two produced −9.31σ and a refused prediction, and both were artifacts
of the *statistic* rather than the physics. The one that worked was not more clever — it was
**better controlled**, and the control came from work already in the repository.

Two lessons worth keeping separately:

1. **Partial correlation was the problem.** It looks like rigour and it silently assumes the
   control variable is comparable across arms. When it isn't, it manufactures large values with
   confident signs. The shuffle needs no such assumption.
2. **Search the corpus before building the instrument.** I built two bad designs over the time
   it would have taken to read `euclidean_distance_validation`. The corpus is not only a record
   of results — it is a record of *methods that survived*, and that is the part I keep
   under-using.

And this is the seventh measurement failure in a day whose errors all ran one direction. The
difference is that this time the null it produced is one I believe, because for once the
apparatus could have said otherwise.
