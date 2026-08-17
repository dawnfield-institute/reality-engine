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

### Time DOES conduct along the web — exp_03's null was a measurement failure

> **CORRECTED 2026-08-17 by exp_06.** exp_03 reported that time does not conduct along the web
> (+0.30σ against a uniform control). That test could not have detected conduction: it coupled
> particles by **distance ball**, which entangles "how much matter is nearby" with "how it is
> connected" in the definition of the operator. In the web a ball held ~798 neighbours against
> 116 in the control at the same `n` and `box`. The null was structural, not physical.

On a **k-nearest-neighbour graph** every particle has bounded degree — 7.37 in the web against
7.14 in the control, CV 0.163 vs 0.170 — so a dense region earns no extra conductance for being
dense. Degree dilution is removed by construction rather than corrected for, and anything that
survives is topology. Measuring **effective resistance** from a source (a steady-state linear
solve, no threshold and no step count) against local density within fixed Euclidean shells:

| k | web | uniform control | difference |
|---|---|---|---|
| 4 | +0.410 | −0.003 | +5.21σ |
| 6 | +0.422 | −0.022 | +6.22σ |
| 10 | +0.442 | −0.028 | **+9.14σ** |

Topology-only edges (every edge conducting equally), 5 seeds. **The effect size is
k-independent** — 0.41 to 0.44 — with significance rising only because more edges means less
noise. Controls sit at zero throughout. It is *stronger* with length removed than with
conductance ∝ 1/length (+0.297, +2.57σ), so it is not the edge-length artifact either.

**Dense particles are better connected than their Euclidean separation implies. Filaments are
conduction paths.**

### And it is a parallel-path effect, not a shortest-path one

The geodesic arm — shortest **physical path length** along the graph — shows *nothing*:
+0.68σ, +0.02σ, −1.64σ across k, scattering around zero with no consistent sign.

Effective resistance and shortest path diverge in only one way: **many redundant routes.** A
filament does not conduct because it is a short wire; it conducts because it is a thick bundle
of parallel ones. Resistance sees that, shortest-path cannot. That also sits better with the
re-entry picture than a single-route model would — redundancy is a property of the structure at
a scale, not of any one path through it.

**Still open**: which Laplacian SEC viscosity actually is. The mean (random-walk) and conduction
(unnormalised) forms predict opposite signs and the corpus never fixed the choice.

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
| [`exp_06_conduction_on_the_graph.py`](scripts/exp_06_conduction_on_the_graph.py) | Does it conduct along the web? Asked on a bounded-degree graph, not a ball. **Corrects exp_03.** |

## What went wrong on the way

Recorded because the failures were more instructive than the results, and all ran one direction.

**A τ claim was retracted mid-round.** τ is sheet-dominated where matter is filament-dominated,
at every scale and epoch, and this was briefly written up as "the clock field sits one axis of
collapse behind matter." Applying τ's *formula* to the same ρ with no viscosity and no evolution
reproduces the inversion (30.5% filament / 58.6% sheet against matter's 61.5 / 29.9). **The
inversion is the nonlinear transform, not the physics.** What survives is second-order.

**exp_03's operator could not answer its own question**, and I reported its null as a result
anyway. A distance ball entangles density with connectivity by construction. exp_06 redoes it on
a bounded-degree graph and finds the opposite, at +9.14σ.

**Three more observables failed before one worked** in exp_06 alone — hop count (edge length
scales with local spacing, so dense material costs more hops for free), explicit time-stepping
(conductance weights make the network stiff; it returned all-NaN, which I nearly read as
"nothing arrives"), and then the length-weighting question itself, settled by an unweighted run.

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
