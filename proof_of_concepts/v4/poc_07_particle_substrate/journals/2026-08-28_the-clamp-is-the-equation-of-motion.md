# The speed clamp is the equation of motion, and sec_balance cannot reach the dynamics

**2026-08-28.** Found while trying to measure whether the v4 substrate has a critical point
(dawn-field-theory M17 Block B). The measurement never got off the ground, and the reason is
worth more than the measurement would have been.

Reproduce with `scripts/diag_clamp_saturation.py` on exp_11's own config
(n=4000, box=60, r0=10, g=1.5, 3D).

## 1. Every particle is pinned, every tick

```
sec_balance = 0.6541 (= Xi/phi, exp_11's derived value)
 tick   at_cap  mean_sp    p99   gravity   sec_press  press/grav   entropy  dense_f
   50   0.4330   1.5120  2.000      2.34         1.2         0.5       0.0   0.0012
  100   1.0000   2.0000  2.000      9.34      8503.2       910.1     100.1   0.5008
  200   1.0000   2.0000  2.000     26.84    152491.8      5681.4    1727.6   0.9417
  300   1.0000   2.0000  2.000     30.01    303851.4     10123.5    4419.3   0.9715
```

From tick ~100 onward **100.0% of particles sit exactly at `max_speed`** — mean, median and p99
all 2.0000. `Integrator` renormalises velocity to the cap, so **force magnitude is discarded
entirely** and only the direction of the net force survives. The substrate is integrating a
direction field, not the force law.

**This was inevitable, not marginal.** Terminal speed without a clamp is `a·dt/(1−damping)` =
9.25 × 0.05 / 0.01 ≈ **46**, against a cap of 2.0. Any force above ~0.4 saturates it — and
forces only get large once structure forms. **The clamp saturates precisely when the engine
starts doing something interesting.**

## 2. sec_balance is dynamically inert here

SEC pressure exceeds gravity by **500× to 17,000×**, and it scales linearly with `sec_balance`
exactly as designed (4431 → 8503 → 14010 for 0.35 → 0.654 → 1.058 at tick 100 — the operator is
*not* unwired). But `sec_balance` multiplies a force that is then renormalised away by the
clamp, and it cannot change the *direction* of a net force already dominated by pressure a
thousandfold. So:

> **The parameter is live in the code and inert in the dynamics.**

A sweep of `sec_balance` across 0.35–1.25 moves the resulting structure not at all — measured
separately: connectivity length flat at 3.84–3.93 for one seed across a 3.6× parameter change,
with seed-to-seed scatter far exceeding it.

**Consequence worth checking:** any prior result obtained by sweeping `sec_balance` in this
regime is measuring the sweep, not the physics. That includes the "Ξ is optimal" reading — this
run does not test it (2D, different config) but it is the obvious thing to re-examine.

## 3. The cause is that SEC memory never releases

`SECUpdate` decays entropy **only on the non-dense branch**:

```python
ent = torch.where(dense, s.entropy + 0.1 * (local - expected), s.entropy * c.memory_decay)
```

Once `dense_fraction` reaches ~0.97, nearly every particle is on the *growing* branch and
`memory_decay` never applies. Entropy runs 98 → 1705 → 4353 monotonically, pressure follows, and
the clamp locks. The negative feedback that should regulate this — pressure disperses clumps,
density falls, `dense` clears, memory releases — **is severed by the clamp**, because dispersal
cannot exceed `max_speed`.

Independently confirmed in a toy (`dawn-field-theory/experiments/spikes/postsymbolic_selection/`):
reinforcement without release saturates into *uniform* suppression, and uniform suppression
selects nothing. Adding a release channel is what produces sustained structure.

## Areas that need work

1. **Raising `max_speed` is the wrong fix** — it relocates the blowup. The real issue is `dt`
   being too large for the forces. The engine **already has per-particle local timesteps**
   (`state.tau`, `time_mode`), built for emergent-time physics and never used for stability.
   Adaptive dt is the fix, and the machinery exists.
2. **Bound the entropy, or make release unconditional.** A refractory that only accumulates has
   one steady state: everything suppressed.
3. **Instrument the bounds.** Nothing reports when a quantity is pinned. `at_cap_frac` belongs
   in `state.metrics` beside `gravity_force_mean` — one line, and it would have caught this on
   the first run rather than the fifteenth.
4. **Re-examine any `sec_balance` sweep** taken in the saturated regime.

## The pattern this belongs to

This is the fifth time in two weeks that a quantity **pinned against a numerical bound** was
read as physics:

| | the bound |
|---|---|
| M16's founding fact | the ξ estimator's white-noise floor |
| M17's founding fact | the particles-per-cell sampling floor |
| M17 exp_02 | the connectivity estimator's ξ ≈ 2-cell resolution floor |
| M17 exp_02 (estimator A) | 1 − 1/e, documented in the selftest and read past anyway |
| **here** | **the velocity ceiling** |

Two of those already earned guards after being paid for — `worldmodel.matched_res()` and the
documented white-noise floor. **A saturation check is the one still missing**, and item 3 above
is it. The general form: *before reading a quantity as physics, check whether it is against a
bound.*
