# 2026-08-11: The global ledger is neutrally conserved — no basin

## Result

**Registered outcome NEUTRAL. The kill sentence fired.**

Displace the global PAC ledger and the displacement **persists undiminished, forever.**
Recovery ratio `R = 1.000` in every trial.

| | R (median) |
|---|---|
| 64×32 @ dt=1e-3 | 0.9999 |
| 32×16 @ dt=1e-3 | 0.9998 |
| 64×32 @ dt=5e-4 | 1.0000 |
| impulse +0.05 | 0.9999 |
| impulse −0.05 | 0.9999 |
| impulse +0.20 | 1.0000 |

18 trials: 3 configs × 3 impulses (±5%, +20%) × 2 seeds. Twin sanity check exact —
pre-impulse gap `0.000e+00`, so the two copies were bit-identical before the impulse and
the difference is purely the perturbation.

Registered band for NEUTRAL is `0.9 ≤ R ≤ 1.1`. Observed: 1.000 to four significant
figures, across every axis varied.

## What it means

**The engine has no mechanism that would make conservation an attractor.** The ledger is
conserved *neutrally* — like momentum in free space, not like a damped oscillator
returning to equilibrium. Add ΔQ and the system simply carries on at Q + ΔQ. Nothing in
the dynamics references an absolute ledger value and pulls toward it.

The only thing that makes conservation hold in this engine is the **explicit correction**
in `NormalizationOperator`, which fires on 99.8% of ticks. Switch it off and Q is free.

## What it does and does not settle

- **Settles:** this engine does not implement conservation-as-attractor. If DFT holds that
  conservation is an attractor rather than a constant, **the mechanism is absent from the
  code**. That is a concrete, buildable gap.
- **Does not settle:** whether conservation is an attractor *in DFT*. A missing mechanism
  in an instrument is not evidence against the physics. This is a statement about what the
  engine implements.
- **Does not settle** anything about local behaviour. Local balance was recorded and
  graded against nothing, per the rule that SEC is local and local leaks are not defects.

## The registered invariant held

`R` was registered as the invariant precisely because it should be independent of impulse
magnitude, ledger value, and config — and it was: 0.9998–1.0000 across a 4× range of
impulse magnitude, both signs, two grids and two timesteps.

Had the number `D₀` or the ledger value been registered instead, nothing would have been
comparable across those runs. *Registered relations survive; registered coordinates die.*

## Why this design succeeded where v1 failed

v1 asked whether the residual survives refinement, which presupposes a continuum ground
truth that DFT denies, and measured the scalar sum rather than the balance RBF and QBE
govern.

This design differences against an identically-seeded twin, so the engine's own baseline
drift — the entire unresolvable question of v1 — cancels exactly and never enters the
measurement. Whatever the unperturbed dynamics do, both copies do it.

That is the transferable lesson: **when a background is contested, difference it away
rather than trying to characterise it.**

## Next

The open question is now a design one, and it belongs to the physics rather than the code:

- What would make the ledger restoring? A restoring term must reference a balance point.
  Ξ is that point, and `RBF` is the Recursive **Balance** Field — but the measurement says
  it currently supplies no restoring force at the global level.
- Is the attractor claim about the ledger at all, or about the **local E↔I exchange**
  approaching Ξ? The scorecard's Tier 1 already measures coupling attractors
  (f→γ_EM, γ→1/φ, α→ln 2, G→1/φ², λ→1−ln 2), so the machinery for testing that exists
  and was not used here.

That second possibility is the more likely reading of the hypothesis, and it is the
natural POC-02.
