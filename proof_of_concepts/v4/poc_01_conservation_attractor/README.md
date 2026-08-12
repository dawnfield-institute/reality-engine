# POC-01 (v4) — Is PAC conservation an attractor or an artifact?

**Status:** pre-registered · not yet run
**Registered:** 2026-08-11, before any sweep was executed
**Generation:** v4 proof-of-concept 1

---

## Why

The engine measures **running couplings** — RG flow with init-independent beta functions,
gravity running 1.3× at high-z. That is time-varying effective law. It *also* enforces
exact PAC conservation on every tick. Those two things are in tension: by Noether, a
conserved quantity exists because of a continuous symmetry, and energy conservation
follows from time-translation invariance. If the laws change as time moves forward, that
invariance is broken by construction and exact conservation does not follow. This is the
ordinary situation in an expanding FRW spacetime, which has no global timelike Killing
vector and therefore no globally conserved energy.

DFT's position (Peter, 2026-08-11) is that **conservation is an attractor, not a
constant** — and that if our global is some parent's local, a persistent bounded imbalance
is what should be observed from inside.

The engine cannot currently see any of this. Measured on `main` at 64×32 over 400 ticks
with the default pipeline:

```
correction FIRED on 399/400 ticks (99.8%)
pre-correction |residual|: median 8.50e-03, max 1.29e-02
trigger threshold:         1.0e-08
trajectory: ~1e-6 at t=0, rising, plateau at ~1.29e-02
```

The correction is unconditional, so the residual is removed before it can be observed.
Spike 9's recorded result — *"PAC conservation | PASS | max deviation 2.06e-14"* — is
therefore measuring the residual left **after** enforcement, not conservation by the
dynamics. That number describes the corrector, not the physics.

The drift is not machine epsilon (so conservation is not exact) and does not random-walk
(so it is not noise). **It saturates.** This POC asks whether that saturation is real.

## Hypothesis

**H1.** With enforcement disabled, the PAC residual saturates at a finite, non-zero value
that **persists under grid and timestep refinement**.

## The competing explanation this must rule out

At 1.29e-02 in float64 the drift is far too large to be floating-point roundoff, but it is
entirely consistent with **truncation error** from the Laplacian stencil and the timestep.
That is the boring explanation and it is the one this POC exists to exclude.

## Registered thresholds

> ### Amendment 1 — 2026-08-11, before any run
>
> The original registration measured the plateau of the **per-tick residual**. That is the
> wrong quantity and the criterion it carried was vacuous.
>
> Per-tick residual ≈ (dQ/dt)·dt. Halving `dt` halves it **whether the drift is physical or
> numerical**, so the registered "falls by >40% per halving of dt" would have been
> satisfied in both cases and could not discriminate. Corrected below to measure the
> **rate**, which is the quantity that actually distinguishes them. Amended before the
> sweep was written to disk and before any results exist.

> ### Amendment 2 — 2026-08-11, before any run
>
> The default pipeline includes `ThermalNoise`, and the registration did not account for
> it. Its amplitude is `√(2·T·dt)` — correct Langevin scaling — but that means a
> noise-dominated residual behaves as `|residual| ~ √dt`, so
> `rel_rate = |residual|/dt ~ dt^(−1/2)`, which **increases** under refinement. That is a
> third signature, distinct from both registered outcomes, and it would have been
> misread as "not converging, therefore interesting".
>
> Also: nothing in the engine is seeded, so runs are not reproducible and single runs
> cannot be compared. A wiring check at `sim_time=0.1` gave plateaus of 1.95e-01,
> 1.53e+00, 3.52e-01 across the three timesteps — scatter, not trend.
>
> **Primary sweep therefore runs deterministically (`noise_scale = 0`)**, which is the
> configuration the registered physical/numerical criteria actually discriminate.
> The stochastic case is characterised separately with repeats and reported as a
> distinct result, not folded into the primary verdict.
>
> Added a third registered outcome for it: `rel_rate` rising as `dt^(−1/2)` under
> refinement means the residual is noise-dominated and says nothing about conservation.

The measured quantity is the **relative drift rate**, dimensionless and comparable across
both grid and timestep:

```
rate      = |residual| / dt          absolute drift per unit simulated time
rel_rate  = rate / |E + I + M|       fractional drift per unit simulated time
```

`P` is the median `rel_rate` over the final 20% of ticks of a run with
`enforce_pac = False`.

**Runs are compared at equal simulated time, not equal tick count** — tick budget is
`T / dt`, so halving `dt` doubles the ticks. Comparing at equal ticks would compare
different amounts of evolution.

Refinement sweep: grid ∈ {32×16, 64×32, 128×64} × dt ∈ {1e-3, 5e-4, 2.5e-4}.

| Outcome | Criterion | Reading |
|---|---|---|
| **Physical** | `P` changes by **< 10%** across successive refinements in both `dt` and grid | converged to a finite non-zero limit |
| **Numerical** | `P` falls **monotonically** with refinement, consistent with `O(dt^p)` / `O(h^q)`, `p,q > 0`, trending toward 0 | truncation error |
| **Noise-dominated** | `P` **rises** under refinement, consistent with `dt^(−1/2)` | residual is thermal noise; says nothing about conservation |
| **Ambiguous** | anything between, or the two axes disagreeing | not decisive; report as such, do not round toward the preferred answer |

## Kill sentence

> **If the relative drift rate `P` scales systematically toward zero as `dt → 0` and the
> grid refines, the imbalance is discretization error, conservation is exact in the
> continuum, and the attractor claim gets no support from this engine.**

This outcome is a real result and will be recorded as such. It would also mean the
engine's conservation enforcement is masking a solver accuracy problem rather than a
physical effect, which is worth knowing on its own.

## Registered invariant, not coordinates

Per the corpus rule — *registered relations survive, registered coordinates die* — the
registered quantity is **not** the number 1.29e-02, which will move with every config.

**Registered:** the *ratio* of plateau value to the coupling-drift saturation timescale,
which should be resolution-independent even if neither quantity is individually.

Secondary and exploratory (not registered, because the estimator is not yet trusted):
whether `P` shifts when the coupling approach is shifted.

## What this does not claim

- Nothing about intelligence, GAIA, or observers.
- Nothing about parent/sibling universes. A persistent bounded imbalance would be
  *consistent* with that reading; it would not be evidence for it. Other explanations
  (an unmodelled sink, an operator that is not self-adjoint on this manifold) are not
  excluded by this design.
- No claim that the engine's dynamics are correct — only a measurement of what they do
  when not corrected.

## Method

1. Add `enforce_pac: bool = True` to `SimulationConfig`. Default preserves current
   behaviour exactly; the 138-test suite must stay green.
2. `NormalizationOperator` skips the correction when it is `False`, and always records the
   pre-correction residual (already implemented).
3. Sweep grid × dt with enforcement off; record the full residual trajectory per run.
4. Compute `P` per run; compare across refinements against the table above.
5. Record the outcome against the kill sentence **as measured**, including "ambiguous".

## Files

```
poc_01_conservation_attractor/
├── README.md          this pre-registration
├── meta.yaml
├── scripts/           exp_01_refinement_sweep.py
├── results/           exp_01_refinement_sweep_YYYYMMDD_HHMMSS.json
└── journals/          dated findings
```
