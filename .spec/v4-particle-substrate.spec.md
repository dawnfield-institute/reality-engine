# v4 Particle Substrate — integrator hygiene

**Status:** implemented 2026-09-05 (commits `996296d` → `0f2cd20`, branch `fix/v4-integrator-owns-dt`;
R1–R9 in code and under test; exp_04 acceptance: controller lines 12/12, pressure lines partial — see Acceptance) · **Scope of this revision:** integrator hygiene only. This
is not a full substrate specification; it specifies the one property the substrate lacked — that
its dynamics are the force law it declares — and the instrumentation that makes that checkable.

## Overview

`proof_of_concepts/v4/particles.py` is a particle substrate (positions, velocities, masses, a
per-particle SEC entropy) with two forces, finite-range gravity and entropy-gradient pressure,
and a memory channel that writes and decays entropy. It had no spec and no tests. On
2026-08-28 (`poc_07_particle_substrate/journals/2026-08-28_the-clamp-is-the-equation-of-motion.md`)
it was found that from tick ~100 every particle sits at `max_speed`: the integrator renormalises
velocity to the cap, so force magnitude is discarded and only direction survives. `sec_balance`
was live in the code and inert in the dynamics; entropy grew without bound because release only
ever applied to particles not flagged dense.

The requirements below make the substrate integrate its force law, and make any bound that
binds visible on the first run rather than the fifteenth.

## Requirements

- **R1 — report the bound.** Every tick, the Integrator records `at_cap_frac` (fraction of
  particles at the speed guard, measured *before* renormalisation, tolerance
  `speed >= max_speed * 0.999` to match the diagnostic), `speed_p99` and `speed_max`. The engine
  keeps a running `bounds` record and warns once per engine above 2%. The engine never asserts.
- **R2 — SEC release is unconditional.** `memory_decay` applies to every particle every tick;
  growth applies only where dense. The operator's docstring (`SECUpdate`, "dense regions
  accumulate; sparse ones forget… without the decay the pressure never releases") is the
  intent; the code matches it. Entropy is then bounded by `0.1·(n − expected)/(1 − memory_decay)`
  with no new knob. No cap is added: with release present a cap is inert (postsymbolic_selection).
- **R3 — the Integrator owns the timestep.** Force operators accumulate acceleration into
  `state.acc`; the Integrator chooses one global `dt` per tick from *this* tick's accelerations
  and velocities by a Courant rule on a robust statistic (p99, never max), floored at `dt_min`,
  and reports `dt_eff`, `dt_at_floor`, `accel_p99`, `cfl_number`, `sim_time`.
- **R4 — per-tick constants become rates.** `damping`, `memory_decay` and the SEC growth
  coefficient are stated at `dt_ref`; they are applied as `x ** (dt_eff / dt_ref)` (or scaled by
  it), which is bit-identical at `dt_eff = dt_ref`. An adaptive step that left them per-tick would
  add dissipation whenever `dt` shrank — a repair that works by turning the physics off.
- **R5 — the speed guard is derived, not chosen.** When `max_speed` is `None` the guard is the
  Courant displacement limit `cfl · r0 / dt_eff`, kept consistent with `dt` by the same rule, so
  it binds on at most the 1 − Q tail by construction. An explicit `max_speed` is honoured (prior
  experiments swept it) and reported like any other bound.
- **R6 — tests fail first.** `tests/v4/` is collected by `pytest.ini`; CPU; under 15 s. Tests
  that fail on the unfixed code carry `xfail(strict=True)` until the fixing commit removes the
  marker — an unexpectedly passing test is then an error, so "fails first" is enforced by the
  runner, not by commit order.
- **R7 — no physics constant is added.** `cfl`, the quantile and `dt_min` are numerical.
  `g`, `r0`, `sec_balance`, `memory_decay`, `damping` and the growth coefficient `0.1` are not
  changed.
- **R8 — local time is untouched.** `LocalTime`, `tau`, `proper_time`, the mean-1
  normalisation and the `*_TIME` pipelines are not modified. With `time_mode="global"` the
  substrate's behaviour at base `dt` is bit-identical to before.
- **R9 — the record.** Results are append-only and timestamped; the 2026-08-28 journal is not
  edited; a new journal and a README banner layer the correction forward; changelog entries
  exist for the diagnostic and for this revision.

## Design

**Interfaces.** `ParticleState.acc: Optional[Tensor]` (N, d), accumulated by force operators
within a tick, consumed and cleared by the Integrator. `ParticleState.dt_last: Optional[float]`,
the step actually taken, read by operators that run before the Integrator on the next tick.
`ParticleConfig.max_speed: Optional[float] = None`, `cfl: float = 0.2`, `dt_min: Optional[float]
= None` (→ `dt / 20`), `dt_ref: float = 0.05`.

**Data flow per tick (CANONICAL).** `SECUpdate` reads `dt_last` for its rate scaling → `LocalGravity`
and `SECPressure` accumulate `acc` → `Integrator` chooses `dt`, kicks, damps as a rate, guards,
drifts, writes `dt_last` and the metrics → `PACLedger` records.

**The Courant rule.** With `a99 = |acc|.quantile(0.99)` and `v99 = (|vel|·tau_i).quantile(0.99)`:
`dt = min(c.dt, cfl·sqrt(r0/a99), cfl·r0/(v99 + a99·dt))`, then `dt = max(dt, dt_min)`, and
`cap = cfl·r0/dt`. On exp_11's quiescent phase (a99 ≈ 8, v99 ≈ 5) `dt` stays at `c.dt`; the
rule only bites when the forces do.

## Acceptance

Measured on exp_11's config (`n=4000, box=60, r0=10, g=1.5, dims=3`) to `sim_time = 15`, three
seeds, and on the `tests/v4` proxy:

- `at_cap_frac ≤ 0.02` at every mark with `dt` above the floor.
- `entropy_mean` is not monotone and never exceeds `0.1·(n − expected)/(1 − memory_decay)`.
- `sec_pressure_mean / gravity_force_mean ≤ 100` at every mark; time-average over
  `sim_time ∈ [5, 15]` ≤ 10.
- No NaN. `ticks_at_dt_floor / ticks ≤ 0.15` (reported, not failed).
- `tests/v3` unchanged (142); `tests/v4` all pass on the branch; the tabled subset fails on `main`.

**Falsification of this spec:** `at_cap_frac > 0.02` at any mark with `dt` above the floor means
the controller is wrong, not the physics.

**Measured (exp_04, 2026-09-05, thresholds as written):** the four controller lines hold on 12/12
runs and the falsification did not fire. Entropy released 11/12 (one run ends mid-collapse).
`press/grav ≤ 100` at every mark holds on 4/12 — the peak is the first detonation and is linear in
`sec_balance` — and the [5,15] average ≤ 10 holds on 11/12. The pressure lines are the force law's
(`.spec/challenges.md` C4.1), not the controller's; they are recorded as failed, not relaxed.

## What this spec does not claim

No physical result. Whether the substrate, once it integrates its force law, produces a web,
a critical point, or a golden coupling is a question for the experiments that consume it
(dawn-field-theory M17 Block B; the M18 dynamics question). The design pass that produced this
spec found that with the clamp no longer binding the substrate is a relaxation oscillator
(collapse → entropy spike → detonation → drag → re-collapse), because the clamp had been its
only energy sink. That is recorded as `.spec/challenges.md` C4.1, not fixed here.
