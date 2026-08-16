# 2026-08-11: Local durability does not predict global conservation — and that is consistent

## Result

**Kill sentence fired.** n = 55 (11 live parameter settings × 5 seeds).

| | value |
|---|---|
| **within-seed ρ** (registered) | **median −0.346** |
| per-seed ρ | **+0.282, −0.473, −0.409, +0.391, −0.345** |
| pooled ρ (artifact, not registered) | −0.611 |
| confound: vigor vs global drift | **0.003** |
| confound: vigor vs local R | −0.124 |

The registered criterion was ρ > 0.6 for support. Median −0.346 is well below it.

**But the more accurate statement is not "negative correlation" — it is *no relationship*.**
The per-seed values do not agree on a **sign**: two positive, three negative, spanning
−0.47 to +0.39. A correlation whose direction flips between seeds is not a correlation.

## The pooled value would have been a false result, twice

Pooled ρ = −0.611 sits **outside the range of every within-seed value**, and is opposite in
sign to two of the five. That is Simpson's paradox: global drift clusters strongly by
initial condition, so pooling measures which seed a point came from.

The first run of this experiment used 2 seeds and reported pooled ρ = −0.43 as the result.
Had it been written up, it would have read as "local durability is *inversely* related to
global conservation" — a striking, publishable-sounding, entirely spurious finding.

## Correction: the confound I flagged was itself an artifact

After the 2-seed run I reported that the common-cause confound "looks real", citing
vigor→global-drift ρ = 0.781, and said it mattered for interpreting the result.

**With 5 seeds it is ρ = 0.003.** The 0.781 was a two-cluster artifact of the same kind as
everything else in that run. The confound is *absent*, not present. Correcting rather than
leaving it to stand.

## Why the null result is coherent rather than disappointing

POC-01 measured the global ledger as an **exact invariant**: R = 1.000 to four decimals
across every impulse. And POC-01 v1, whose verdict was withdrawn on ontological grounds,
nonetheless established a fact that survives: the observed drift in Q tracks `dt`, at
convergence order ≈ 0.5.

Put those together and the null is what should be expected:

- **The global ledger's drift is integration error.** It is numerical.
- **Local balance durability is dynamical.** It is physics.
- A numerical quantity and a dynamical one have no reason to correlate.

So local enforcement has nothing to do *to* the global ledger — the dynamics already
conserve it exactly, and what drift remains is the integrator, not the physics. **"Local
enforcement produces global conservation" gets no support here, because in this engine
global conservation needs no producing.**

That is not a refutation of the framework's claim. It is a statement that **this engine
cannot exhibit the mechanism**, because it has no level at which the global ledger is
anything other than trivially conserved. Testing the claim needs a system where the global
quantity is *not* structurally invariant — which is the nested-patch capability, where a
sub-grid's ledger is one term in a parent's sum and therefore genuinely free to move.

## Method findings

**`mass_gen_coeff` is a dead config field** — declared in `config.py`, read by zero
operators. `memory.py` computes `gamma_local = diseq²/total_field²` directly. Sweeping it
produced bit-identical runs. It was swept in POC-02 and in the first POC-03 run before this
was noticed; POC-02's breadth claim is corrected in its journal, and
`harness.assert_params_live()` now checks before sweeping.

**Seed dominates global drift by ~36×** against ~40% from parameters. Any correlational
question about global drift in this engine needs within-seed analysis or many seeds. This
is a property of the engine worth knowing independently of this POC: initial conditions
matter far more than any parameter tested.

## Next

- The nested-patch capability is now the only way to test the claim as stated. It is a v4
  engine feature, not an experiment.
- **Variance control**: a 36× seed effect makes correlational work expensive and fragile.
  Worth understanding before more of it is attempted.
