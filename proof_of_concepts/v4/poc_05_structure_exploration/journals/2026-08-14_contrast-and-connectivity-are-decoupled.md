# 2026-08-14: Contrast and connectivity are decoupled

The engine's density can be driven harder than a real cosmic web's. Its connectivity does not
move at all, under any intervention tried.

---

## The result

Percolation — largest connected component of the overdense set, as a fraction of that set.
A synthetic web scores **1.000** in one component. White noise scores **0.003** across 1394.

| intervention | range explored | cv (contrast) | percolation |
|---|---|---|---|
| baseline | — | 1.33 | 0.012 |
| de-actualization rate `eta` | 0.0 → 3.2 | 0.80 → **1.60** | 0.019 → 0.013 |
| `field_scale` (clamp) | 20 → 5000 | 0.83 → 0.76 | — |
| quantum pressure | 0.0 → 0.30 | 0.49 → 1.27 | — |
| grid | 128×32 → 128×128 | 0.83 → 1.01 | 0.019 |
| ticks | 3000 → 12000 | 0.82 → 1.01 | 0.019 |
| gradient transport | 0 → 50× | 1.33 → **3.90** | 0.014 |
| curl transport | 0 → 50× | 1.33 → 2.28 | 0.010 |
| both | 0 → 50× | 1.33 → 2.59 | 0.010 |

**Contrast spans a 5× range. Percolation spans 0.007 – 0.019 and never leaves the noise
floor.** cv reaches 3.895 — nearly twice a real cosmic web's ~2.0 — with percolation at 0.014.

Across roughly forty runs, four grid configurations, three tick counts, both Helmholtz halves
of the EM force, and transport strengths spanning 100×, **nothing moved the topology.**

## Why this is the interesting number

Every other measure in this work is a one-point statistic — void fraction, density CV,
overdense fraction, even correlation length — computed cell by cell with no reference to what
any neighbour *is*. Percolation is the only one that asks whether the overdense cells **touch**.

That is the one that came back flat.

exp_09 produced a cosmic web from 5000 particles and measured connectivity directly, as a
clustering coefficient of 0.54. It did not need a percolation gate, because in a particle
substrate the neighbour graph *is* the model. Here the neighbour graph does not exist, and no
amount of forcing conjures one.

## What this rules out

A whole class of intervention, not a list of settings. Every fix attempted this round was some
version of *push the density around differently*:

- more loss (`eta` up 128×)
- less clamping (`field_scale` up 250×)
- an accumulating local refractory (QPL, ported from `brain.py`)
- patch-local conservation (designed, superseded before running)
- transporting mass along the EM force, gradient half and curl half separately
- more room, more time, more noise, less damping, unenforced PAC

The result is invariant across all of it. Connectivity is not a density phenomenon that needs
more forcing.

## The mechanical reason

**Every force in this engine is the gradient of a scalar potential.** Gravity is a Poisson
solve. `ChargeDynamicsOperator` builds a genuine vector force from the charge potential and
then adds only its **divergence** to `dE_dt` — the curl is discarded, and it is 89% the
magnitude of what is kept (|curl| rms 0.714 vs |div| 0.799 at 128×128, t=3000).
`SpinStatisticsOperator` computes `S = curl(E−I)` and uses it only as a scalar weight.

Gradient flow is downhill flow. It has point attractors. It makes blobs. Filaments are lines,
and a line cannot be built from an irrotational flow. So "contrast yes, connectivity no" and
"divergence kept, curl discarded" are the same fact stated twice.

## Why restoring the curl did not fix it

It was the obvious next move and it is null: curl transport leaves percolation at 0.010.

The reason is visible in the curl field itself:

| field | ξ_u | coherent fraction |
|---|---|---|
| white noise reference | 0.626 | 0.0723 |
| E field | 0.628 | 0.0668 |
| gradient velocity | 1.013 | 0.1793 |
| **curl velocity** | **0.477** | **0.0154** |

The curl velocity is **five times less coherent than white noise**, with ξ below the floor —
grid-scale churn. And structurally so: the Poisson solve weights by 1/k², so the gradient part
*is* the smooth large-scale part by construction and the curl is the high-k remainder. `Q` is
itself a derivative of `E`, and `E` is white noise (ξ 0.628, identical to the reference).

So the curl here is not carrying missing structure — it is carrying the noise. **EM cannot make
filaments out of a field that has none.** Coherent structure needs coherent charge, coherent
charge needs coherent fields. Nothing in the engine breaks that circle, and charge is recomputed
from scratch every tick rather than accumulating any.

## Two bad tests before the real one, recorded

The curl result took three attempts and the first two were mine, not the physics':

1. **Underpowered.** `dt = 0.0007`, so 3000 ticks is 2.1 time units and a normalised velocity
   moved mass about **one cell** across the whole run. It read as a clean null.
2. **Numerically invalid.** Raising the transport 50× diverged in every arm — because centred
   differences with forward Euler are **unconditionally unstable for advection** at any
   timestep. It is fine for diffusion, which is why it looked reasonable.

Fixed with donor-cell upwind, verified before use: mass conserved 1.000000 → 1.000000, and a
blob advected 200 steps at v=1, dt=0.05 lands at u=42 from u=32, exactly as it should.

## The frame this belongs to

From the repository's own opening paragraph:

> When a hammer shatters glass, thermodynamics tells us where the energy goes — heat, sound,
> kinetic motion. But **new information was created**: each shard now has unique geometry,
> distinct edges, specific boundaries. Standard physics has no framework for where that
> structural information comes from.

Everything measured in this POC before percolation was where the energy goes. Density, contrast,
voids, CV, correlation length — the thermodynamic half. The result is that the thermodynamic
half can be driven arbitrarily hard and **the structural information does not appear**. The
shards never get edges.

A filament network is a graph. Its information is in its connectivity, which no density field
carries — the density is the shadow. That is why measuring the shadow harder was never going to
find it, and why every intervention in the list above failed in the same way.

## What this does not say

- **Not** that the engine is broken. It does what it was built to do, and does it well.
- **Not** that EM is irrelevant to filaments. It says the claim cannot be tested in a substrate
  where every derived quantity is a derivative of noise.
- **Not** that `eta` is unimportant — it is the strongest contrast lever found, moves every
  density metric monotonically across seeds with no saturation, and its default of 0.025 was
  set by spike 04 as optimal for **coupling accuracy**, a different objective that pulls the
  other way. That is a real finding and it stands on its own.

## Reproduce

```
python proof_of_concepts/v4/structure.py                                  # calibration must PASS
python proof_of_concepts/v4/poc_05_structure_exploration/scripts/exp_01_structure_panel.py
python proof_of_concepts/v4/poc_05_structure_exploration/scripts/explore_friction.py
python proof_of_concepts/v4/poc_05_structure_exploration/scripts/explore_loss.py
python proof_of_concepts/v4/poc_05_structure_exploration/scripts/explore_curl.py
```
