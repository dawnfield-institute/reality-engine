# The speed clamp is the equation of motion; sec_balance is inert (diagnostic)

**Date:** 2026-08-28 (entry written 2026-09-05, retroactively — the commit carried a journal and no
changelog) · **Commit:** `2bab371` (merged to main at `21b9dcc`) · **Type:** research / diagnostic

Found while trying to measure whether the v4 particle substrate has a critical point
(dawn-field-theory M17 Block B). The measurement never got off the ground; the reason was
worth more than the measurement.

## Findings (exp_11's 3D config: n=4000, box=60, r0=10, g=1.5)

- **Every particle pinned, every tick.** From tick ~100, `at_cap = 1.0000` with mean, median and
  p99 speed all exactly `max_speed = 2.0`. The integrator renormalises velocity to the cap, so
  force *magnitude* is discarded and only direction survives — the substrate integrates a
  direction field, not the force law. Inevitable: drag-terminal speed `a·dt/(1−damping)` ≈ 46
  against a cap of 2.0; any force above ~0.4 saturates it, and forces only get large once
  structure forms.
- **`sec_balance` live in the code, inert in the dynamics.** SEC pressure exceeds gravity by
  500×–17,000× and scales linearly with `sec_balance` (4431 → 8503 → 14010 for 0.35 → 0.654 →
  1.058 at tick 100), but it multiplies a force the clamp then renormalises away. A sweep over
  0.35–1.25 left connectivity length flat at 3.84–3.93.
- **SEC memory never releases.** `SECUpdate` decayed entropy only on the non-dense branch; once
  `dense_fraction` reached ~0.97 nearly every particle was on the growth branch. Entropy ran
  100 → 1728 → 4419 monotonically. Confirmed in a toy
  (`dawn-field-theory/experiments/spikes/postsymbolic_selection`).

## Consequences named
Raising `max_speed` relocates the blowup (poc_08 measured 242× KE growth at cap 20); the step is
the problem. Bound or release the entropy. **Instrument the bounds** — `at_cap_frac` in
`state.metrics` would have caught this on the first run rather than the fifteenth. Re-examine
any `sec_balance` sweep taken in the saturated regime, the "Ξ is optimal" reading first.

## The pattern
Fifth quantity in two weeks pinned against a numerical bound and read as physics (M16's ξ
estimator floor, M17's particles-per-cell floor, M17 exp_02's ξ ≈ 2-cell resolution floor, the
documented 1 − 1/e floor, the velocity ceiling). A saturation check was the guard still missing.

## Files
`proof_of_concepts/v4/poc_07_particle_substrate/journals/2026-08-28_the-clamp-is-the-equation-of-motion.md`,
`proof_of_concepts/v4/poc_07_particle_substrate/scripts/diag_clamp_saturation.py`.
Repaired 2026-09-05: see `.changelog/20260905_*_v4_integrator_owns_dt.md`.
