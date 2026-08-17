# Emergent local time, a runnable world model, and a retraction

## The retraction first

**POC-09's headline claim was wrong, and the error was mine, not exp_11's.**

POC-09 reported that the corpus's published 3D cosmic web "does not percolate" — 0.0068 against
a 0.0025 noise control — and that claim became the empirical anchor for a milestone in
`dawn-field-theory` (M17), which opened on the premise that four independent routes had
triangulated on a percolation floor and the framework was "maximally sub-critical."

exp_11's web percolates. Same run, binning alone:

| res | particles/cell | percolation | `is_web` |
|---|---|---|---|
| 16 | 0.98 | 0.472 | True |
| **32 — exp_11's own** | 0.12 | **0.385** | **True** |
| **64 — the retracted claim** | **0.015** | 0.062 | False |

4000 particles on a 64³ grid is 0.015 per cell: the density field is empty by construction, the
overdense set shatters into singletons, and *any* web — real or deliberately synthetic — reads as
disconnected. A known-connected 3D control reads **1.000 across occupancy 0.082–0.268**, so the
instrument was sound and the sampling was not.

**POC-09's own committed script confirms it**: run unmodified at exp_11's resolution it reports
percolation **0.3443 ± 0.0628**. The claim was never reproducible by the code that shipped with it.

Three further defects found while checking:

- The script defaulted to `res=64` — the exact resolution POC-09's *own* `meta.yaml` identifies
  as the failed first attempt ("read CV 15.4 against a target of 2.94"). Documented, never fixed.
  Now defaults to 32.
- "REPLICATION IS EXACT" (void 0.888 / CV 2.948) is **not reproducible**: the script gives void
  0.939, CV 7.105 at 32³, failing its own CV gate. Now recorded as UNVERIFIED.
- Ξ ambiguity, exactly as `CLAUDE.md` warns: POC-09 uses `Ξ_analytic/φ = 0.65414`, POC-10 uses
  `Ξ_discrete/φ = 0.65334`. Two rounds at different operating points, neither saying so.

The 2026-08-16 journal is preserved unedited with a superseded header; corrections layer forward.

## POC-10 — emergent local time

New `LocalTime` operator on the v4 particle substrate: per-particle clock rate `τ`, per-particle
proper time, neighbour diffusion of `τ`. `τ` is normalised to mean 1, so local time is a
*redistribution* of a fixed global budget rather than a speed-up — the ledger stays auditable.
`Integrator` advances each particle by its own `dt = dt_base · τᵢ`. Inert at
`time_mode="global"`, so nothing changes when it is off.

Premise: a collapse event is a tick, mass is spent potential, so mass-dense regions have less
budget left and tick slowly. Dilation from PAC bookkeeping rather than inserted.

### Viscosity is the mechanism, not the stabiliser

Partial correlation of `τ` with the long-range Newtonian potential — which the substrate never
computes — controlling for `τ`'s own exact input:

| ν | partial given grid δ | **partial given τ's exact input** |
|---|---|---|
| 0.0 | +0.630 | **−0.009 ± 0.04** |
| 0.3 | +0.746 | **+0.240** |
| 0.6 | +0.817 | **+0.322** |

Zero at ν=0 — with diffusion off, emergent time is its own local input relabelled. That null is
what makes the rest mean anything. Neighbour diffusion, iterated, is what carries clock
information past `r0` and makes time a **field** rather than a per-particle label.

The middle column is the trap: controlling for *grid-scale* δ leaves +0.630 at ν=0, which looks
like a result and is a scale confound (τ is built at r0=10, the grid cell is 3.0).

### What does not hold

- **The form is wrong for GR.** τ(Φ_newton): quadratic R² 0.908, √(1+2Φ) 0.802, linear 0.749.
  If this were dilation the √ form should win.
- **Time does not conduct along the web.** Under the conduction (unnormalised Laplacian) form the
  web channels no better than a uniform control: +0.217 vs +0.179, **+0.30σ**. The clock field
  tracks how much matter is nearby, not how it is connected. Which Laplacian SEC viscosity is was
  never fixed by the corpus, and the two forms predict opposite signs.

### Structure character is scale-dependent

Hessian classification across a smoothing ladder: filaments peak at intermediate scale (58% at
0.45 r0), nodes rise outward, voids dominate the small end. Across epochs the filament peak
migrates **outward** (0.45 → 0.64 r0) while small-scale voids grow 6.3% → 25.3% → 32.7%.

### An "ended or moved?" detector

| system | vs measured control | verdict |
|---|---|---|
| frozen | 0.00× | TERMINATED ✓ |
| dissolving | 0.97× / 1.03× | TERMINATED ✓ |
| web at t=450 | **2.34× / 2.43×** | **RE-ENTERING** |

A snapshot at the last epoch reports a collapsed clumped field; the trajectory reports handover.
Discriminating quantity is cross-scale **exchange of collapse order**. Three other candidates
failed on overlapping ranges and are recorded. The threshold is not hard-coded — the dissolving
control runs in the same invocation and everything is a multiple of it.

## Tooling

- **`worldmodel.py`** — runnable world model. `run` evolves and writes an animated GIF of the
  density field beside a live metrics trace; `sweep` renders one panel per parameter value. 2D
  and 3D, JSON alongside.
- **`matched_res()`** — picks the metrics grid from `n^(1/d)` so the sampling artifact above
  cannot recur by default; the run banner prints particles/cell every time.
- **Rendering decoupled from measurement.** Opposite requirements: statistics want ~1
  particle/cell, an image wants many per pixel. Separate grids, slab projection in 3D, Gaussian
  deposit, and the display plots overdensity ρ/ρ̄ so it cannot be silently rescaled.
- **Zel'dovich initial conditions** (`ic="zeldovich"`) — correlated ICs off a P(k) field.
  Measured +2.1 to +3.0σ on percolation in 2D; **only +1.07σ in 3D**, so not the key ingredient.
  Default remains `lattice`, matching exp_09/exp_11.
- **`ParticleEngine.field_of()`** — bins an *intensive* per-particle quantity as a weighted mean
  (summing it would re-measure density under another name).

## Verification

- `pytest` — 142 passed. Nothing under `src/` was touched.
- `structure.py` and `law_detector.py` selftests — PASS, including the 3D geometries.
- All five POC-10 scripts run after relocation into `scripts/`.

## Notes

Every measurement error in this session ran one direction: **understating structure**. Percolation
read low, transport read null, τ's structure read as artifact, the web read as decay. A null
reads as rigour, so the error mode that survived review was the one that looked careful. The
habit that held up was measuring the reference in the same run rather than reasoning about
thresholds — every result that survived had a control that could have killed it.
