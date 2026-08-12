# 2026-08-11: The kill sentence fired — PAC drift is discretization error

## Summary

**The registered kill sentence fired.** With enforcement disabled, the relative PAC drift
rate falls monotonically under timestep refinement on all three grids, converging toward
zero at order ≈ 0.53. **The imbalance is discretization error. The attractor claim gets no
support from this engine.**

This is recorded as the result. It was not retried, retuned, or re-thresholded.

## Result

Deterministic (`noise_scale = 0`), `enforce_pac = False`, equal simulated time T = 0.4,
plateau = median |dQ/dt|/|Q| over the final 20% of ticks.

| grid | dt=1e-3 | dt=5e-4 | dt=2.5e-4 | ratios | order p |
|---|---|---|---|---|---|
| 32×16 | 1.703e-02 | 1.266e-02 | 8.207e-03 | 0.743, 0.648 | +0.527 |
| 64×32 | 4.220e-02 | 2.941e-02 | 2.043e-02 | 0.697, 0.695 | +0.523 |
| 128×64 | 4.971e-02 | 3.454e-02 | 2.291e-02 | 0.695, 0.663 | +0.559 |

Control, `enforce_pac = True`: **3.12e-13** — enforcement holds Q at machine precision, as
expected, confirming the flag is what changes the answer.

Registered criterion for NUMERICAL: *plateau falls monotonically with refinement, trending
toward 0.* Met on every grid, with a consistent exponent.

## Two findings that matter more than the verdict

**1. Convergence order is ≈ 0.5, which is poor.** A well-constructed scheme would be O(dt)
or O(dt²). Order ½ suggests the operator splitting, or the non-smooth clamps (`M` floored
at 0, `tanh` soft-clamp on E/I), are degrading the integrator. **The conservation
enforcement was masking a solver accuracy problem.** The pre-registration anticipated this
outcome and named it as valuable on its own; it is the actionable result of this POC.

**2. It is not converging in space.** At fixed dt, refining the grid *increases* the rate:

```
32×16   (512 cells)  1.703e-02
64×32  (2048 cells)  4.220e-02
128×64 (8192 cells)  4.971e-02
```

Growth is +148% then +18%, so it may be saturating toward a resolution-independent value,
but three points cannot establish that. The registered criterion required convergence in
**both** axes. It converges in time and does not clearly converge in space. The dominant
error is temporal, but the spatial behaviour is unresolved and should not be reported as
convergent.

## What this does and does not settle

- **Settles:** the ~1.29e-02 residual visible on `main` is discretization error, not
  physics. Spike 9's *"PAC conservation PASS, max deviation 2.06e-14"* measures the
  corrector's residual, not conservation by the dynamics — both statements can be true at
  once, and both are now understood.
- **Does not settle:** whether conservation is an attractor *in DFT*. This engine cannot
  test that claim, because its own temporal discretization error is larger than any effect
  it would be looking for. That is a statement about the instrument, not the hypothesis.
  A conclusive test needs an integrator whose error is demonstrably below the effect size.

## Three registration errors, all caught before data

Recorded because the process is the point, and each would have produced a confident wrong
answer:

1. **Amendment 1** — the original criterion measured the plateau of the *per-tick*
   residual. Per-tick residual ≈ (dQ/dt)·dt, so halving dt halves it whether the drift is
   physical or numerical. Vacuous. Corrected to a rate, at equal simulated time.
2. **Amendment 2** — the registration ignored `ThermalNoise`. Its `√(2·T·dt)` amplitude is
   correct Langevin scaling, but a noise-dominated residual gives rate ~ dt^(−1/2), which
   *rises* under refinement and would have read as "not converging, therefore
   interesting". Primary sweep made deterministic.
3. **Amendment 3** — caught only by inspecting the totals against the computed rate. With
   enforcement OFF, `metrics["pac_correction"]` holds the *cumulative* deviation from
   Q(0); with enforcement ON the correction resets it each tick so it holds a *per-tick
   increment*. The same variable means different things in the two modes. Dividing the
   cumulative form by dt inflated the rate as dt shrank — producing an apparent RISE,
   the exact opposite of the true result. Fixed by differencing Q(t) directly.

The first sweep, before Amendment 3, reported *"NOISE-DOMINATED — plateau rises under
refinement"* with noise switched off. That contradiction is what exposed the bug.

## Next

- Investigate the order-½ convergence. Likely candidates: operator splitting, the `M`
  floor, the `tanh` soft-clamp.
- Extend the spatial sweep before claiming anything about grid convergence.
- Do not re-run this POC hoping for a different verdict.
