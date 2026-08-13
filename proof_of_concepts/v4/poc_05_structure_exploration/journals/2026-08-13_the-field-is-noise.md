# 2026-08-13: The field is noise. Correlation length = 1 cell.

## What the picture shows

Rendered M for RBF and PACBalance, same seed, three timesteps. Both are **pixel-level
speckle**. The thresholded set that `box_dimension` measures is scattered isolated pixels —
no filaments, no connected structures, no web.

RBF develops visible large-scale patchiness by t=5000 (loose purple/yellow domains), which
matches the measured spectral tilt of 9.09x toward the box mode. PACBalance stays uniform
speckle, matching its 2.23x. So the spectral measurement was right and is visible.

## Quantified

Spatial autocorrelation length of M, in cells:

| | t=500 | t=1666 | t=5000 |
|---|---|---|---|
| RBF | 1.00 | 1.00 | 1.00 |
| PACBalance | 1.00 | 1.00 | 1.00 |

**1.00 cell is the floor** — it means neighbouring cells are uncorrelated. A cosmic web has
a correlation length of many cells (the filament width). The engine has none, under either
balance operator, at any time measured.

## What this retracts

Written the same day as the panel it corrects.

- **"D = 0.88 is filament-like"** — no. D = 0.88 is what scattered pixel noise scores at
  10% occupancy. The calibration passed on clean synthetic shapes and still does; it simply
  does not distinguish noise from filaments at this occupancy, which the eye does instantly.
- **"PACBalance builds structure (+0.344)"** — the D change is a change in the statistics
  of a noise field, not the emergence of structure. Both endpoints are noise.
- **"Initial-condition independence"** — three inits converge to the same *noise statistic*.
  Real convergence, but not a structure result.

## What survives

- **The spectral tilt.** RBF 9.09x vs PACBalance 2.23x, and RBF's box-scale patchiness is
  visible in the render. Mechanism understood: RBF's laplacian damps as k^2, so small
  scales die and power piles into the box mode.
- **The engine produces no spatial structure.** That is the working-machine problem, stated
  plainly, and it is the most useful result of the day.

## The diagnosis this points to

Correlation length 1.0 means the dynamics are effectively **pointwise**. Almost every
operator in the pipeline acts per-cell; the only spatially-coupling terms are RBF's
laplacian and gravity's Poisson solve, and both are evidently too weak relative to the
local terms to build correlation between neighbours.

Nothing downstream can make a web out of an uncorrelated field. Contrast growth, Jeans
scales, fractal dimension and entropy are all measurements of a spatial structure that does
not exist yet. **The prior question is how neighbouring cells come to agree at all.**

## Method note

Hours of statistics were computed on this field before anyone looked at it. The broken
box-counting estimator (D = 2.000 for a filament) survived that long for the same reason.
One rendered frame beside the number would have caught both immediately.

`render_fields.py` exists now. Render before trusting a statistic.
