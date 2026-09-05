# v4 particle substrate: the Integrator owns dt — the substrate integrates its force law

**Date:** 2026-09-05 · **Branch:** `fix/v4-integrator-owns-dt` (commits `996296d` → `0f2cd20` +
this record) · **Type:** fix / instrumentation · **Spec:** `.spec/v4-particle-substrate.spec.md`
· **Repairs:** `.changelog/20260828_215838_clamp_saturation_diagnostic.md`

Numerical hygiene, instrumented. **No physics claim.**

## Changed (`proof_of_concepts/v4/particles.py`)
- **Instrument first (C1).** `Integrator` reports `at_cap_frac` (pre-clamp speed, the
  diagnostic's tolerance), `speed_p99`, `speed_max`; `ParticleEngine.bounds` keeps the running
  record and warns **once** above 2% — annotate always, never assert. `worldmodel.py` prints and
  traces `at_cap`. The unchanged diagnostic reproduced the 08-28 numbers to every digit.
- **Ledger (C2).** `momentum_z` recorded in 3D; it had been silently absent.
- **SEC release unconditional (C4).** `memory_decay` applies to every particle every tick;
  growth only where dense; both as rates against `ParticleConfig.dt_ref` read through
  `ParticleState.dt_last`. Entropy is bounded by `0.1·(n − expected)/(1 − memory_decay)` with no
  cap and no new knob. Non-dense branch bit-identical at base dt.
- **The Integrator owns dt (C5).** Forces accumulate `state.acc`; the Integrator chooses one
  global step per tick — `min(c.dt, cfl·√(r0/a99), cfl·r0/(v99 + a99·dt))`, floored at dt/20, on
  p99 statistics so the tail never sets the clock — kicks, damps as a rate (`damping**(dt/dt_ref)`),
  guards with a **derived** cap (`cfl·r0/dt_eff`; `max_speed=None` default, an explicit value
  honoured and reported), drifts. Reports `dt_eff`, `dt_at_floor`, `cap_eff`, `accel_p99`,
  `cfl_number`, `sim_time`. Cosmology advances on the step taken. `LocalTime`/`tau` untouched.
- `diag_clamp_saturation.py` reads the derived cap from metrics.

## Tests and gate
`tests/v4/` (9, CPU, ~2 s) on an exp_11-density proxy; `pytest.ini` now collects `tests/v3 tests/v4`.
Fail-first enforced by `xfail(strict=True)`; on unmodified main the file gives 3 failed / 4 xfailed
/ 2 passed. On the branch: **151 passed**. CLAUDE.md counts updated.

## Measured (before → after; exp_11's config n=4000, 300 ticks, `diag_clamp_saturation.py`)
| | at_cap @100/@300 | entropy @100/@300 | press/grav @100/@300 |
|---|---|---|---|
| sec Ξ/φ | 1.0000/1.0000 → **0.0010/0.0018** | 100/4419 → **34.4/28.5** | 910/10124 → **92.5/86.6** |

`sec_balance` reaches the dynamics now (press/grav 91 / 143 / 199 across 0.35 / Ξ/φ / Ξ at tick
200; it was inert). Proxy bounds: at_cap_frac_max 0.010, ticks at floor 0, dt_eff_min 0.0051.

**Acceptance at matched simulated time (exp_04, 12 runs to sim_time 15, `results/exp_04_integrator_owns_dt_20260905_161653.json`):**
controller lines hold 12/12 — at_cap max 0.009, 0 ticks at the dt floor (min step 0.0025), entropy max
mean 43 against a bound of 7845, no NaN; the spec's falsification did not fire. Entropy released
11/12 (0.35 seed 1 ends mid-collapse at 0.63). `press/grav ≤ 100` at every mark **4/12** — every
peak is the first detonation at t = 3.0 and scales with the knob (52–72 @ 0.35 … 255–321 @ 1.25);
the [5,15] average ≤ 10 holds 11/12 (Ξ/φ seed 2 at 12.5). Thresholds not moved; the pressure
lines are C4.1. Response: xi_u does not move (range 0.016 vs std 0.011); the percolation flag
trips on one outlier run (0.35 seed 1 at 0.190) and is not read as a response — the other three
arms are 0.024–0.029.

## What it is now
A relaxation oscillator — collapse, entropy spike, detonation (p99 speeds 370–680), dispersal,
drag, re-collapse. The clamp had been the only energy sink. Filed as `.spec/challenges.md` C4.1
for the M18 dynamics conversation; not tuned here. Also filed: C4.2 (design B, `tau` in the
kick), C4.3 (two clocks: `proper_time` vs `sim_time`).

## Files
`proof_of_concepts/v4/particles.py`, `worldmodel.py`, `poc_07_particle_substrate/{README.md
(banner), meta.yaml (notes), journals/2026-09-05_the-integrator-owns-dt.md, scripts/exp_04_integrator_owns_dt.py,
scripts/diag_clamp_saturation.py, results/exp_04_*}`, `tests/v4/`, `pytest.ini`,
`.spec/v4-particle-substrate.spec.md`, `.spec/challenges.md`, `CLAUDE.md`,
`.changelog/20260828_215838_clamp_saturation_diagnostic.md` (retro).
