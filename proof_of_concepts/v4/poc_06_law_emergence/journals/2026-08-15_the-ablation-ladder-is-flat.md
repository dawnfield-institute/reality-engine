# 2026-08-15: The ablation ladder is flat, and the one exciting reading was quiescence

exp_01 asked the 16-operator pipeline what laws it has and got one, enforced. The follow-up was
deliberately **subtractive** — every additive proposal in this round had failed the same way —
so: strip operators, score with the law detector at each rung, and see whether the emergent
count goes up, down, or stays flat.

## Result

| rung | ops | EMERG | ENF | PAC CV | act/cell (late) | emergent |
|---|---|---|---|---|---|---|
| 16 canonical | 16 | 0 | 1 | 1.84e-15 | 0.05073 | — |
| 12 no deep physics | 12 | 0 | 1 | 2.09e-15 | 0.07773 | — |
| 8 no forces/thermal | 8 | 0 | 1 | 1.34e-14 | 0.07685 | — |
| 6 core+memory+confluence | 7 | 0 | 1 | 1.25e-14 | 0.14957 | — |
| 5 core+memory | 6 | 0 | 1 | 4.66e-16 | 0.11151 | — |
| 4 core only | 5 | 2 | 1 | 2.26e-16 | **0.00273** | E*I, E²+I² |
| 4 core, enforce_pac off | 5 | 3 | 0 | 2.26e-16 | **0.00273** | E*I, E+I+M, E²+I² |
| 3 NO normalization | 4 | — | — | — | — | **DIVERGED at t=69** |

**Flat at zero across every rung that is still evolving.** The operators are not suppressing
emergence, and the axioms as implemented do not generate laws at any depth of the stack.

## The exciting reading, and why it is wrong

The first pass showed emergence *rising* on ablation — 0 at 16 operators, 2 at the core, 3 with
PAC enforcement off. That would have inverted the engine's whole development history.

It is quiescence. Direct measurement of the core-only pipeline:

```
tick   mean|E-E0|    std(E)      sum(E*I)   n_actual
 500   1.6012e+00  2.0629e+00  1.725895e+04     66
1500   1.6557e+00  2.0614e+00  1.740393e+04     26
3000   1.6635e+00  2.0619e+00  1.741362e+04      7
```

Actualization decays 66 → 7 events per tick on 4096 cells. `mean|E−E0|` plateaus, `std(E)` is
flat, and `sum(E*I)` still drifts 1.8e-3 over the last 2500 ticks — right at the 1e-3 tolerance,
so it is not conserved, it is settling.

**A dead system conserves everything.** Fewer operators means less driving, the field coasts to
a fixed point, and every quantity looks invariant.

The `act/cell` column makes it unambiguous without any threshold: **0.00273 against 0.05–0.15**,
20–55× less active than every other rung. Emergence appears exactly and only where the system
has stopped.

## Two detector faults this exposed

**Enforcement was a name lookup.** `ENFORCED_BY_CONSTRUCTION` is a static table, so `E+I+M` read
as ENFORCED at every rung — including `enforce_pac off`, where the correction never fires and
PAC holds by construction from QBE's `dI = −dE` with no mass being made. A lookup table cannot
draw the enforced/emergent distinction this module exists to draw. Enforcement is now decided
per run from what is actually in the pipeline.

**The divergence check only watched M.** The `NO normalization` rung returned `PAC CV = nan` — a
non-finite ledger that slipped through because E blew up while M stayed finite. Now checks E, I
and M.

Both would have made every rung look identical, which is the "flat at zero" answer this
experiment ended up giving — and it would have been the right answer for the wrong reason.

## The liveness gate

Added to `conservation_scan`: pass a per-tick activity series and it refuses to certify anything
when activity collapses — verdict **QUIESCENT** rather than EMERGENT. Selftested both ways: a
dead system with a decaying activity series returns QUIESCENT, the same series with flat
activity returns EMERGENT.

The threshold (5% of early activity) did **not** fire on the core rungs, which sit at 14%. It
was left alone rather than raised until the verdict flipped — **tuning to fail is the same fault
as tuning to pass.** The absolute activity figure is reported instead, so the reading does not
depend on a chosen number.

## Also recorded

`NO normalization` diverges at t=69. The clamps are load-bearing for stability even though the
PAC correction is not what does the conserving in a minimal pipeline. Two separable jobs inside
one operator.

## Standing

Six instrument faults this round, all caught by rendering, by a system with a known answer, or
by a direct measurement of whether anything was happening — none by any statistic taken at face
value.

This one was the subtlest: a conservation law that is **real, reproducible across seeds, and
completely meaningless.**
