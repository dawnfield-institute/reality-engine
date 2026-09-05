# The Integrator owns dt — the substrate integrates its force law

**2026-09-05.** Repair of the defect recorded in
[`2026-08-28_the-clamp-is-the-equation-of-motion.md`](2026-08-28_the-clamp-is-the-equation-of-motion.md).
That journal stands as written; this one layers forward. Spec: `.spec/v4-particle-substrate.spec.md`.
Scope: **integrator hygiene, instrumented.** No physics claim is made here.

## What was wrong, in three parts, and one the diagnostic did not say

1. **The cap was the equation of motion.** `max_speed = 2.0` sat *below* the physical speed
   scale of the forces (free fall √(2·a·r0) ≈ 5; drag-terminal a·dt/(1−damping) ≈ 6–40), so from
   tick ~100 every particle was renormalised to it and force magnitude was discarded.
2. **SEC memory never released.** Decay applied only to particles *not* flagged dense; at
   `dense_fraction` ≈ 0.97 the memory could not release. Entropy 100 → 1728 → 4419.
3. **Nothing reported either.** The Integrator was the only pipeline operator writing no
   metrics. Fifteen runs read a pinned quantity as physics.
4. **Gravity alone pins the cap first** (found by the design pass, not in the diagnostic). On an
   exp_11-density proxy `at_cap = 0.64` at tick 50 with entropy and pressure still exactly zero.
   The entropy runaway is the second act; a release-only fix could not have worked.

## What changed (commits `996296d` → `0f2cd20`, branch `fix/v4-integrator-owns-dt`)

| commit | change |
|---|---|
| C1 `996296d` | **Instrument first.** `at_cap_frac`, `speed_p99`, `speed_max` from the pre-clamp speed; `engine.bounds`; warn once above 2%; `at_cap` column in `worldmodel.py`. No dynamics touched — the unchanged diagnostic reproduced the 08-28 numbers to every digit. |
| C2 `644f401` | `PACLedger` records `momentum_z` in 3D (it recorded x and y only). |
| C3 `2ca7945` | `tests/v4` (9 tests, fail-first via `xfail(strict=True)`), `pytest.ini` gate extended, the spec, `.spec/challenges.md` C4.1–C4.3. |
| C4 `3ec9414` | **SEC release unconditional**; growth and decay as rates against a new `dt_ref`. Non-dense branch bit-identical at base dt (261/261 particles); every dense particle lower. `entropy_bounded_and_not_monotone` passed at this commit already — release alone bounds the entropy even under the clamp. |
| C5 `e73d72c` | **The Integrator owns dt.** Forces accumulate `state.acc`; one global step per tick from *this* tick's forces by a Courant rule on p99 (never max), floored at dt/20; `damping` as a rate; the guard **derived** (`cfl·r0/dt_eff`, `max_speed=None` by default; an explicit value is honoured and reported). `LocalTime`/`tau`/`proper_time` untouched; forceless path bit-identical at base dt. |
| C6 `0f2cd20` | `exp_04_integrator_owns_dt.py` (evidence, matched simulated time); retro changelog for 08-28; CLAUDE.md. |

Two hygiene facts drove the design. **Per-tick constants had to become rates**: an adaptive step
that left `damping`, `memory_decay` and the growth `0.1` per tick would add dissipation whenever
it shrank — a repair that works by turning the physics off. And **no fixed cap near 2.0 could
ever be a never-bind guard**, so the guard is now the Courant displacement limit, kept consistent
with the step by the same rule and binding on at most the p99 tail by construction.

## The numbers

**Proxy** (exp_11's density at n = 500, 3D, sec = Ξ/φ; the test fixture), repaired substrate:

```
 tick  sim_t  at_cap  dt_eff     cap     a99   v_p99  ent_mean  ent_max press/grav
   50   2.50   0.000  0.0500    40.0     7.0     4.6      0.00      0.0        0.0
  100   3.19   0.002  0.0052   383.0   459.8   369.4     58.92     93.4      112.0
  300   4.38   0.000  0.0067   298.9   152.3   291.0     17.53     27.7       26.6
  600   7.03   0.006  0.0118   169.3     7.8   168.6      1.17      2.1        1.8
  800  10.24   0.000  0.0226    88.5     2.9    87.9      0.04      0.1        0.1
bounds: at_cap_frac_max 0.010 · ticks_at_cap_gt_1pct 0 · ticks_at_dt_floor 0 · dt_eff_min 0.0051
```

**Full size** (`diag_clamp_saturation.py`, exp_11's config n = 4000, 300 ticks, three arms), before → after:

| | at_cap @100 / @300 | entropy @100 / @300 | press/grav @100 / @300 |
|---|---|---|---|
| sec 0.35 | 1.0000 / 1.0000 → **0.0005 / 0.0000** | 98 / 4353 → **47.0 / 30.3** | 479 / 5336 → **62.8 / 55.7** |
| sec Ξ/φ | 1.0000 / 1.0000 → **0.0010 / 0.0018** | 100 / 4419 → **34.4 / 28.5** | 910 / 10124 → **92.5 / 86.6** |
| sec Ξ | 1.0000 / 1.0000 → **0.0003 / 0.0010** | 102 / 4457 → **26.6 / 26.7** | 1489 / 16705 → **101.4 / 139.7** |

`sec_balance` now reaches the dynamics: at tick 200, press/grav 90.9 / 143.3 / 198.7 and mean
speed 214 / 258 / 298 across the three arms. It was inert before. Note that 300 ticks after the
repair is only `sim_time` ≈ 4 (the step shrinks to ~0.005 in the detonation), so this table is
the apples-to-apples printout on the diagnostic's own horizon, not the acceptance measurement.

**Tests.** `tests/v3` 142 + `tests/v4` 9 = **151 passed**, no expected failures remaining. Against
unmodified main the same v4 file gives 3 failed (`at_cap_frac` missing, DID NOT WARN, `momentum_z`
missing), 4 xfailed, 2 passed (the anchors).

## What the substrate is now, honestly

With the clamp no longer binding, **the substrate is a relaxation oscillator.** Collapse → the
first dense event dumps entropy → the entropy-gradient force is impulsive (a99 ≈ 460 on the
proxy against free fall ≈ 5) → the clump detonates to p99 speeds of 370 (proxy) and 500–680
(full size) → disperses → drag bleeds the energy over ~10 time units → re-collapse. The clamp
had been the substrate's only energy sink; poc_08's 242× kinetic-energy number said so and was
read as a cap effect. Whether the pressure parametrisation is physical or an artifact of
exp_09's per-tick calibration is `.spec/challenges.md` **C4.1**, owned by the M18 dynamics
conversation. It is not tuned here.

## Acceptance at matched simulated time — exp_04

`results/exp_04_integrator_owns_dt_20260905_161653.json` (+ `_log.txt`). exp_11's config
(n = 4000, box = 60, r0 = 10, g = 1.5, 3D) × sec_balance {0.35, Ξ/φ, 1.0, 1.25} × seeds {1, 2, 3},
each to `sim_time = 15` (1046–1757 ticks, 140–240 s CPU). The JSON records commit `e73d72c`: the
run was launched from the working tree before the script itself was committed at `0f2cd20`.

| arm | seed | at_cap max | dt min | floor | entropy mean max → end | release | press/grav peak (at t) | avg [5,15] | xi_u | perc |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.35 | 1 | 0.009 | 0.0037 | 0 | 41.2 → 26.0 | **0.63** | 51.6 (3.0) | 6.85 | 0.643 | **0.190** |
| 0.35 | 2 | 0.007 | 0.0045 | 0 | 28.4 → 0.00 | 0.00 | 51.4 (3.0) | 2.27 | 0.588 | 0.032 |
| 0.35 | 3 | 0.007 | 0.0041 | 0 | 30.7 → 0.03 | 0.00 | 72.1 (3.0) | 2.41 | 0.595 | 0.037 |
| Ξ/φ | 1 | 0.009 | 0.0030 | 0 | 43.1 → 0.00 | 0.00 | **139.1** (3.0) | 3.20 | 0.599 | 0.028 |
| Ξ/φ | 2 | 0.009 | 0.0036 | 0 | 36.0 → 8.95 | 0.25 | **132.8** (3.0) | **12.50** | 0.592 | 0.032 |
| Ξ/φ | 3 | 0.007 | 0.0033 | 0 | 30.3 → 0.00 | 0.00 | 74.0 (3.0) | 3.40 | 0.587 | 0.028 |
| 1.0 | 1 | 0.009 | 0.0027 | 0 | 37.7 → 0.00 | 0.00 | **218.6** (3.0) | 4.06 | 0.588 | 0.028 |
| 1.0 | 2 | 0.009 | 0.0031 | 0 | 34.2 → 0.00 | 0.00 | **229.3** (3.0) | 3.96 | 0.603 | 0.029 |
| 1.0 | 3 | 0.008 | 0.0029 | 0 | 34.3 → 0.00 | 0.00 | **182.9** (3.0) | 3.96 | 0.592 | 0.014 |
| 1.25 | 1 | 0.009 | 0.0025 | 0 | 33.9 → 0.00 | 0.00 | **271.7** (3.0) | 4.32 | 0.605 | 0.025 |
| 1.25 | 2 | 0.009 | 0.0029 | 0 | 31.2 → 0.00 | 0.00 | **320.5** (3.0) | 4.29 | 0.587 | 0.018 |
| 1.25 | 3 | 0.007 | 0.0027 | 0 | 32.3 → 0.00 | 0.00 | **255.4** (3.0) | 4.74 | 0.602 | 0.028 |

**Scored against the spec's acceptance lines, thresholds as written, none moved:**

| line | result |
|---|---|
| `at_cap_frac ≤ 0.02` at every mark with dt above the floor | **holds, 12/12** (max 0.009) |
| entropy never above the provable bound (7845) | **holds, 12/12** (max mean 43) |
| `ticks_at_dt_floor / ticks ≤ 0.15` | **holds, 12/12** (0 ticks at floor; min step 0.0025) |
| no NaN | **holds, 12/12** |
| entropy released: `mean(t_end) ≤ 0.5 · max` | **11/12** — fails on 0.35 seed 1 (0.63), which at t = 15 is still inside a collapse (its percolation 0.190 is the outlier below) |
| `press/grav ≤ 100` at every mark | **4/12** — fails on 8; every peak is the first detonation at t = 3.0 and scales with the knob: 52–72 @ 0.35, 74–139 @ Ξ/φ, 183–229 @ 1.0, 255–321 @ 1.25 |
| `press/grav` averaged over `sim_time ∈ [5, 15]` ≤ 10 | **11/12** — fails on Ξ/φ seed 2 (12.5); the other eleven sit at 2.3–6.9 |

The spec's own **falsification** (`at_cap_frac > 0.02` with dt above the floor ⇒ the controller is
wrong) did not fire on any run. The two pressure lines are properties of the force law the spec
said it would not tune: the peak is the impulsive first detonation and it is linear in
`sec_balance`, which is the same fact as "the knob now reaches the dynamics" seen from the
acceptance side. Filed under C4.1; **not** repaired here and the thresholds are **not** relaxed.

**Response to `sec_balance` (end state, 3 seeds; the pre-declared reading is range > 2× within-seed std):**

| | 0.35 | Ξ/φ | 1.0 | 1.25 | within-seed std | range | flag |
|---|---|---|---|---|---|---|---|
| xi_u | 0.608 | 0.592 | 0.594 | 0.598 | 0.011 | 0.016 | does not move |
| percolation | 0.086 | 0.029 | 0.024 | 0.024 | 0.022 | 0.062 | *moves* |

The percolation flag is **not read as a response.** The 0.35 mean is carried by one run (seed 1
at 0.190; its siblings 0.032 and 0.037) that ended mid-collapse; the other three arms are
0.024–0.029, indistinguishable. A mean-of-three with one outlier trips a 2σ range test — that is a
defect of the test's design, declared here, not a finding about the force law. Honest reading:
**no measurable end-state response to `sec_balance` above 0.35 at this size and horizon.** The
numbers go to the M18 dynamics conversation as measurements.

M17's `connectivity_length` was not importable from this worktree (sibling-repo path resolves
from the wrong parent) and is recorded as absent; M17 applies its own estimator to its own runs.

## What is deliberately untouched

`LocalTime`, `tau`, `proper_time`, the mean-1 normalisation and the `*_TIME` pipelines (poc_10's
physics; design B is C4.2). `SECUpdateRelative`'s rule (dt-scaled only). The values of `g`, `r0`,
`sec_balance`, `memory_decay`, `damping`, the growth `0.1`. `structure.py`, `law_detector.py`,
`src/v3`. The 08-28 journal and diagnostic script's reading text — the script reads the derived
cap now, its READING block describes the numbers it was written for.

## Consumers

M17 Block B (after M17 exp_02 v2 is registered with its ξ ≳ 2-cell domain of validity), and the
M18 dynamics question — does anything drive the branch coupling to 1/φ — which can now be asked
of a dynamics rather than of a clamp. exp_04's press/grav and connectivity numbers go to that
conversation as measurements.
