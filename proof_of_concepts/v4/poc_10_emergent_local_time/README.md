# POC-10 (v4): Emergent local time

**Status**: active · **Pillar**: SEC

## Why

If a collapse event *is* a tick, the local framerate is set by how much collapse the
neighbourhood can still do. Mass is **spent potential**, so where potential has been used up
there are fewer collapse events left and the clock runs slow. That makes gravitational time
dilation a consequence of PAC bookkeeping rather than something inserted — nothing in the
substrate knows about general relativity.

It also lands on a real defect. v3's `TimeEmergence` computes `dt = dt_base / (1 + κ·max|E−I|)`
— a `.max()` over the whole grid, one scalar for every cell — while its own docstring says
"regions". The operator already claimed to be local and never was.

## What was built

`LocalTime` in [`../particles.py`](../particles.py): a per-particle clock rate `τ`, per-particle
`proper_time`, and neighbour diffusion of `τ`. **`τ` is normalised to mean 1**, so local time is
a *redistribution* of a fixed global budget rather than a global speed-up — which is what keeps
the ledger auditable. `Integrator` advances each particle by its own `dt = dt_base · τᵢ`.

Two candidate framerate sources, because the corpus does not fix the choice and they diverge
exactly where it matters — a region can be potential-rich but quiescent:

| mode | τ | slow clocks sit on |
|---|---|---|
| `potential` | `1/(1 + κ·δ)` — remaining budget | filaments and nodes |
| `rate` | `1/(1 + κ·\|dδ/dt\|)` — instantaneous collapse | infall fronts |

Pipelines `CANONICAL_TIME` and `EXP11_TIME`. All of it is inert at `time_mode="global"`, so the
substrate behaves exactly as before when local time is off.

## Results

### Viscosity is the mechanism, and it is not a stabiliser

The headline. Partial correlation of `τ` with the **long-range Newtonian potential**, controlling
for `τ`'s own exact input (`state.prev_delta`, the neighbour-count overdensity it is computed
from). The substrate never computes `Φ_newton`:

| ν | corr(τ, Φ_newton) | partial, given grid δ | **partial, given τ's exact input** |
|---|---|---|---|
| 0.0 | +0.773 | +0.630 | **−0.009 ± 0.04** |
| 0.3 | +0.844 | +0.746 | **+0.240** |
| 0.6 | +0.883 | +0.817 | **+0.322** |

**At ν = 0 it is exactly zero** — with diffusion off, emergent time is its own local input
relabelled. That null is what makes the rest meaningful: the test can return nothing. Switch
diffusion on and the clock field acquires non-local structure that scales with ν.

So viscosity is not the anti-collapse stabiliser it was added as. **It is the mechanism by which
time becomes a field** rather than a per-particle label — diffusion over neighbours, iterated,
carries clock information past `r0`. The channel was named before the run, not after.

The middle column is the cautionary one: controlling for grid-scale δ leaves +0.630 at ν=0,
which looks like a strong result and is entirely a **scale confound** — `τ` is built from
neighbour counts within `r0 = 10` while the grid cell is 3.0, so `f(δ_r0)` survives a regression
on `δ_grid`. Regressing out the exact input is what collapses it to zero.

### The functional form is wrong for GR

Fitting `τ` against `Φ_newton`:

| form | R² |
|---|---|
| quadratic | **0.908** |
| GR weak field, `√(1 + 2Φ)` | 0.802 |
| linear, `1 + Φ` | 0.749 |

If this were gravitational dilation the √ form should win. A clock field that responds
non-locally to the potential — yes. **Dilation in the GR sense — not demonstrated.**

### Remaining budget beats collapse rate

`potential` puts **89%** of slow clocks on filaments and 11% on boundaries, `corr(τ,δ) = −0.673`.
`rate` gives 56/37 and −0.089. The open question closes in favour of remaining budget.

> Note `corr(τ, δ) = −0.673` is **near-definitional** — τ *is* a decreasing function of δ. It is
> reported because it separates the two modes, not as evidence of anything.

### Time does not conduct along the web

| viscosity form | web | uniform control | difference |
|---|---|---|---|
| mean (random-walk Laplacian) | −0.346 | +0.093 | −4.83σ |
| conduction (unnormalised) | +0.217 | +0.179 | **+0.30σ** |

Under the conduction form — the "like macro EM" reading — the web channels no better than an
unstructured control at the same `n`, `box` and `r0`. The clock field tracks **how much matter is
nearby, not how it is connected**. The control returning +0.179 rather than 0 is the giveaway:
local density fluctuations produce most of the effect and topology adds nothing measurable.

**Which Laplacian SEC viscosity is was never fixed by the corpus, and the two predict opposite
signs.** That is the live question, not a settled null. A distance-ball coupling cannot see
connectivity by construction; testing conduction properly needs a neighbour *graph* with bounded
degree, so filaments are paths rather than blobs. Different operator, not a parameter change.

### Structure character is scale-dependent, and the organising scale migrates

Hessian classification (void / sheet / filament / node) across a smoothing ladder, t = 180:

| R / r0 | void | sheet | filament | node |
|---|---|---|---|---|
| 0.11 | 25.3% | 44.0% | 28.1% | 2.5% |
| 0.45 | 2.3% | 33.8% | **58.3%** | 5.7% |
| 1.12 | 4.6% | 45.2% | 41.3% | **8.9%** |

One field, three different answers depending on scale. Across epochs the filament peak moves
**outward** (0.45 → 0.64 r0) while small-scale void fraction grows 6.3% → 25.3% → 32.7%: small
scales evacuate as structure re-forms at larger ones.

### Ended, or moved? — a calibrated detector

| system | exchange | vs control | verdict |
|---|---|---|---|
| frozen | 0.000 | 0.00× | TERMINATED ✓ |
| dissolving | 0.137 / 0.146 | 0.97× / 1.03× | TERMINATED ✓ |
| **web at t=450** | 0.330 / 0.343 | **2.34× / 2.43×** | **RE-ENTERING** |

A snapshot at the last epoch reports a collapsed, clumped field. The trajectory reports handover.
The discriminating quantity is **cross-scale exchange of collapse order**: a dissolving system
moves every scale the same direction; a re-entering one moves scales against each other.

Three other candidates were measured and failed — `1 − |Σd|/Σ|d|` (overlap), signed net drift
(overlap), late/early exchange ratio (overlap). The threshold is not hard-coded: the dissolving
control runs in the same invocation at the same `n`/`box`/`scales` and everything is a multiple
of it.

## Scripts

| script | question |
|---|---|
| [`exp_01_time_flow.py`](scripts/exp_01_time_flow.py) | Does the clock field trace the web? Four panels: matter, τ, flow, correlation. |
| [`exp_02_dilation.py`](scripts/exp_02_dilation.py) | Is it dilation, or density relabelled? The partial-correlation test. |
| [`exp_03_transport.py`](scripts/exp_03_transport.py) | Does time flow *along* the filaments? |
| [`exp_04_scale_character.py`](scripts/exp_04_scale_character.py) | How does character depend on scale and epoch? |
| [`exp_05_terminated_or_reentering.py`](scripts/exp_05_terminated_or_reentering.py) | Has this ended, or moved? Calibrated against systems built to die. |

## What went wrong on the way

Recorded because the failures were more instructive than the results, and all ran one direction.

**A τ claim was retracted mid-round.** τ is sheet-dominated where matter is filament-dominated,
at every scale and epoch, and this was briefly written up as "the clock field sits one axis of
collapse behind matter." Applying τ's *formula* to the same ρ with no viscosity and no evolution
reproduces the inversion (30.5% filament / 58.6% sheet against matter's 61.5 / 29.9). **The
inversion is the nonlinear transform, not the physics.** What survives is second-order.

**Three observables failed before one worked** in the transport experiment — amplitude
(degree-confounded), arrival time (also degree-confounded, after I claimed it wasn't), and a
first run on a **saturated graph** where every particle was reached in ~2 hops and arrival
fraction was 1.00 everywhere. There was no front to measure and both "results" were noise.

**The detector's calibration once passed while broken.** An intermediate version defaulted every
verdict to TERMINATED through a NaN path, so both controls "passed" for reasons unrelated to what
they guard. It now also requires that the reference *separated* the web from the controls.

## Known limits

- **One model of death.** The detector's controls contain progressive smoothing only. Freezing
  mid-structure, or fragmenting without smoothing, are unrepresented and each could fool it.
- **exp_05 needs enough scales and resolution.** At n=8000/res=32/6 scales it reports 2.34× and
  PASS; at n=3000/res=24/3 scales it reports 0.79× and correctly *refuses* to answer.
- **Ξ ambiguity.** This POC uses `Ξ_discrete/φ = 0.65334`; POC-09's script uses
  `Ξ_analytic/φ = 0.65414`. 0.1% apart, but they are not the same operating point.
- One configuration (exp_11's), one epoch range, 2–3 seeds per arm.
