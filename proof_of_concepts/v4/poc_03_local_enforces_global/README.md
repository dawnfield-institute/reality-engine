# POC-03 (v4) — Does local enforcement produce global conservation?

**Status:** COMPLETE — **no relationship.** Within-seed ρ median −0.346 with inconsistent
sign across seeds (+0.28 to −0.47). Kill sentence fired. See
[`journals/2026-08-11_no-relationship.md`](journals/2026-08-11_no-relationship.md).
**Registered:** 2026-08-11
**Generation:** v4 proof-of-concept 3

---

## The claim

POC-01 and POC-02 together showed:

| | R | |
|---|---|---|
| global ledger | 1.000 | exact invariant — displace it, it stays displaced |
| local E/I balance | 0.028 | durable, wide basin |

The reading under test (Peter, 2026-08-11):

> **The global ledger is where redistribution is tallied; local balance is where it is
> actually enforced.** Conservation is therefore *emergent from* local enforcement rather
> than separately imposed.

That reading explains both numbers without strain: nothing acts on a sum, so nothing
restores it; the pushing happens per-cell. R = 1.000 is then not a missing mechanism but
the signature of a genuine invariant.

**But as stated it is a story, not a mechanism** — it explains the data after the fact and
forbids nothing. This POC turns it into something that can be wrong.

## Hypothesis

If local enforcement is what produces global conservation, then **settings with weaker
local balance durability should show worse global conservation.** The two should move
together across parameter space.

**H1.** Across parameter settings, local balance recovery ratio `R` and global ledger drift
rate are **positively correlated** — worse local durability (higher R) accompanies larger
global drift.

## Measured

Both quantities come from the *same runs*, so they cannot differ by run conditions.

| quantity | how |
|---|---|
| **local R** | `info_fraction` recovery ratio under a ledger-preserving impulse — as POC-02 |
| **global drift** | on the **unperturbed reference twin**: \|Q(end) − Q(start)\| / \|Q(start)\| / elapsed simulated time, with `enforce_pac = False` |

Global drift is taken from the reference, never the perturbed copy, so the impulse cannot
contaminate it.

Sweep: the POC-02 grid — `quantum_pressure_coeff`, `deactualization_rate`,
`mass_gen_coeff`, `confluence_weight` at ×0.5 and ×2, plus base — × 2 seeds = 18 points.

## Registered outcomes

Spearman rank correlation `ρ` between local R and global drift across all 18 runs.
Spearman rather than Pearson: the relationship need only be monotonic, and rank
correlation is robust to the outliers a parameter sweep reliably produces.

| Outcome | Criterion |
|---|---|
| **Supported** | `ρ > 0.6` |
| **Weak** | `0.3 < ρ ≤ 0.6` |
| **No support** | `ρ ≤ 0.3`, including any negative value |

## Kill sentence

> **If `ρ ≤ 0.3`, local durability does not predict global conservation. The two are
> independent properties of the engine, and "local enforcement produces global
> conservation" gets no support here — it remains a consistent story with no mechanism
> behind it.**

Recorded as the result either way. Not retried, not retuned.

## The confound this cannot fully exclude

Both quantities may be driven by a third thing: **how violent the dynamics are at a given
setting.** Cranking `mass_gen_coeff` might degrade local durability *and* increase global
drift without either causing the other — a common cause, not a mechanism.

A vigor proxy (`max_disequilibrium`, and `balance_magnitude`) is recorded per run so the
correlation can be inspected against it. **This is a limitation, not a control**: with 18
points a partial correlation would not be trustworthy, and it is not registered as one.
A positive `ρ` here is consistent with the mechanism and does not establish it.

Stated up front because POC-01 v1 failed by leaving its assumption unstated.

## Registered invariant, not coordinates

`ρ` — a rank correlation, dimensionless and invariant under any monotonic rescaling of
either axis. Not the drift values, not the R values, neither of which is comparable across
settings.

## Assumptions this design does not make

- **No continuum limit**, no privileged discretization.
- **No claim that either quantity has a correct value.** Only whether they co-vary.
- **No causal claim** from correlation alone — see the confound above.
