# 2026-08-13: Exploration panel — the engine does build structure

Exploratory, not predictive. Goal was a working machine, and to record honestly which
levers did nothing.

## First: the earlier metric was broken and its results are void

The box-counting estimator used through most of this work thresholded by quantile and
fitted saturated scales. Calibration against known geometries:

| input | true D | broken estimator |
|---|---|---|
| straight filament | 1.0 | **2.000** |
| sine filament | 1.0 | **2.000** |
| single blob | ~2 | 2.000 |
| scattered points | ~0 | **2.000** |
| unstructured noise | — | **1.459** |

It returned 2.000 for everything structured and 1.459 for noise. The "D ≈ 1.44, flat under
every intervention" result was the estimator reporting *no structure*, over and over.

`proof_of_concepts/v4/structure.py` replaces it: adaptive threshold to a target occupancy,
unsaturated scales only, nan rather than a fitted-through-saturation slope, and a
`selftest()` that must pass — filament 1.000, sine filament 0.862, half-plane 2.000,
points 0.083.

## Panel: 128x64, 5000 ticks, calibrated D(M)

| variant | init | D start -> end | dD |
|---|---|---|---|
| baseline RBF | big_bang | 0.887 -> 0.906 | +0.018 |
| baseline RBF | entropy_dominated | 0.559 -> 0.614 | +0.056 |
| baseline RBF | info_dominated | 0.547 -> 0.913 | **+0.366** |
| **PAC balance** | big_bang | 0.531 -> 0.874 | **+0.344** |
| **PAC balance** | entropy_dominated | 0.540 -> 0.897 | **+0.357** |
| **PAC balance** | info_dominated | 0.525 -> 0.877 | **+0.352** |
| RBF, no gravity | big_bang | 0.887 -> 0.899 | +0.012 |
| PAC, no gravity | big_bang | 0.529 -> 0.578 | **+0.049** |
| RBF, poisson M^0.8 | big_bang | DIVERGED | — |
| RBF, high viscosity | big_bang | 0.886 -> 0.910 | +0.024 |
| RBF, low damping | big_bang | 0.885 -> 0.908 | +0.022 |
| RBF, unenforced PAC | big_bang | 0.888 -> 0.910 | +0.022 |
| RBF, thermal noise | big_bang | 0.885 -> 0.899 | +0.014 |

### Three things fall out

**1. PACBalance grows structure; RBF (from big_bang) does not.** +0.344 vs +0.018, a 19x
difference on the same initial condition. D lands near **0.88** — filament-like (1.0), not
space-filling (2.0) and not point collapse (0.0).

**2. PACBalance makes growth initial-condition-independent.** All three inits converge to
D = 0.874, 0.897, 0.877 from starts of 0.53, 0.54, 0.53. RBF does not: big_bang +0.018
against info_dominated +0.366. Different paths, one endpoint is the corpus's own attractor
signature (`pac-necessity-proof`, spike 12, Ember III r15).

**3. Gravity is required, and my earlier claim that it was decorative was wrong.**
PACBalance with gravity +0.344; without it +0.049. Sevenfold. The earlier "removing
gravity changes nothing" was the broken metric reporting no-structure either way.

## Spectral picture — where the power goes

Growth ratio P(k, late)/P(k, early) of the mass field:

| | large/small-scale tilt | at box scale |
|---|---|---|
| RBF | **9.09x** | 47.2x |
| PACBalance | **2.23x** | 2.4x |

RBF pours power into the box-scale mode — global clumping. Mechanically that is its
laplacian: `∇²` damps as k², so small scales die and large scales survive, and power piles
up at k -> 0. PACBalance has no laplacian and is close to scale-neutral.

## Quantum pressure sets a preferred scale

PACBalance base, growth-ratio peak location vs `quantum_pressure_coeff`:

| qp | peak at | tilt |
|---|---|---|
| 0.0 | lambda = 23.2 | 1.53x |
| **0.02** (default) | **lambda = 10.5** | **0.95x** |
| 0.3 | lambda = 2.1 (edge) | 0.13x |

Pressure moves the preferred scale, and the default sits at an interior peak with a
near-flat tilt — the gravity/pressure competition behaves like a Jeans scale and is
tunable.

## Density perturbation growth

Seeded sinusoids on M, tracked 200 -> 3000 ticks: k=2 grew 1.82x, k=4 **decayed** 0.37x,
k=8 grew, k=16 grew 1.69x. Growth is scale-selective, not uniform diffusion.

## Levers that did nothing

Recorded so they are not retried: viscosity across 40x, low damping, PAC enforcement
on/off, thermal noise on/off, the pi-harmonic even-depth null, growing Xi in the tiling
filter (20x range). All within +-0.01 of baseline on dD.

`poisson M^0.8` diverges. `M^1.0` diverges faster. The engine has no mechanism to survive
genuine density contrast — the stability comes from suppressing the thing you want.

## Open

- D lands at 0.88, not 1.6 -> 1.06 as the darkmatter spike's observational comparison ran.
  Different measure and different system, but worth reconciling.
- PACBalance's lambda and alpha are Tier 3 in exp_30 — structure forced, parameters not.
  Unswept here.
- Whether the D growth saturates past 5000 ticks.
