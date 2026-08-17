# Shuffle nulls: the clock field follows spatial locality, not the web

## The question

`exp_06` showed the **network** conducts along filaments (+9.14σ, degree-matched). Whether the
**clock field** exploits that is separate — and `LocalTime`'s viscosity still diffused over a
**distance ball**, the same operator shape that made `exp_03` unable to answer its own question.

## Two confounded designs, then the corpus's method

**Ball vs k-NN — confounded by kernel size.** A prediction stated in advance (graph coupling
should carry *more* non-local information) was refused: +0.646 against the ball's +0.746. The
sweep shows why — τ's non-local content is monotone and roughly logarithmic in coupling degree:

| coupling | degree | partial |
|---|---|---|
| knn k=6 | 7.6 | +0.085 |
| knn k=24 | 29.6 | +0.180 |
| knn k=48 | 58.8 | +0.275 |
| ball | 237.0 | +0.399 |

So ball-vs-knn compares two smoothing radii, not two topologies. Worth keeping independently:
exp_02's ν > 0 result **scales with kernel size**, which had not been measured.

**Web vs uniform — confounded by control-variable variance.** The control scored *higher*
(+0.360 vs +0.088, −9.31σ). Partial correlation is not comparable across arms whose control
variable has different spread: in a uniform box δ barely varies so regressing it out removes
nothing; in a web it dominates τ so regressing it out strips the signal.

**The method that works came from the corpus** — `dawn-field-theory`
`experiments/studies/euclidean_distance_validation`, experiment_25, which faced the mirror-image
accusation ("your metrics correlate because they are redundant") and answered it by shuffling
(r 0.79 → −0.29; ξ 0.87 → 0.17), with a 100-trial baseline and a z-score.

A **degree-preserving edge shuffle** holds particles, positions, density, δ and its variance,
degree, edge count and kernel size all fixed. Only *which* particles are connected changes. Both
confounds become impossible by construction — and no partial correlation is needed, removing the
statistic that generated the false signals.

## Result

Degree preservation asserted every run.

| configuration | real | shuffled | gap | z |
|---|---|---|---|---|
| web | +0.708 | +0.091 | **+0.617** | +23.5 |
| uniform | +0.783 | +0.095 | **+0.688** | +41.6 |

Rewiring destroys **87%** of τ's agreement with the long-range potential. But the shuffle removes
the web's topology *and* spatial locality together, so the same null inside a uniform box
separates them — each arm compared against a shuffle of itself, so the gaps are comparable.

**EXCESS = −0.071 (−4.27σ).** The web's gap is *smaller*.

**τ follows spatial locality; the web's topology adds nothing on top.**

## Not a contradiction with exp_06

Different questions, both hold. The *network* conducts preferentially through dense regions; the
*clock field's* agreement with Φ does not exploit that, needing only that the coupling be local.
The graph has the conductive structure and τ isn't using it.

Either the mean-relaxation form washes the advantage out — testable by swapping in the conduction
Laplacian, which exp_06 showed behaves differently — or **Φ is the wrong observable**, since a
smooth long-range potential may not care about network structure at all. The second is untested
and is the more interesting one.

## Code

- `particles.py` — `LocalTime` gains `time_coupling` (`"ball"` | `"knn"`) and `time_k`. Bounded
  degree: 237 → 7.6. Default unchanged (`"ball"`), inert at `time_mode="global"`.
- `poc_10/scripts/exp_07_shuffle_null.py` — new; `--compare-uniform` runs both arms.
- `exp_02_dilation.py` — `--time-coupling` / `--time-k`.
- `results/exp_07_shuffle_null_web_vs_uniform.{json,png}`

`pytest` — 142 passed.

## Note

Seventh measurement failure in a day whose errors all ran one direction. The difference is that
this null is one I believe, because the apparatus could have said otherwise. Two lessons kept
separately: **partial correlation looks like rigour and silently assumes the control variable is
comparable across arms**; and **search the corpus before building the instrument** — two bad
designs cost more than reading `euclidean_distance_validation` would have. The corpus records
methods that survived, not only results.
