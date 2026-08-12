# POC-02 (v4) — How durable is balance?

**Status:** COMPLETE — **balance is durable, and it is LOCAL.** info_fraction R = 0.028,
durable in 18/18 parameter settings, against the global ledger's R = 1.000 in POC-01. See
[`journals/2026-08-11_balance-is-durable.md`](journals/2026-08-11_balance-is-durable.md).
**Registered:** 2026-08-11
**Generation:** v4 proof-of-concept 2


> ## ⚠ SUPERSEDED BY PRIOR WORK — 2026-08-11
>
> **This POC tested the wrong model.** It was designed without reading the corpus.
> See [`../RESEARCH.md`](../RESEARCH.md).
>
> - The ledger is **P + A + Δ = C**, not E + I + M. Δ is a bounded buffer of unreconciled
>   actualization, cleared at reconciliation boundaries. Validated 5/5 in
>   `dawn-field-theory/experiments/studies/asymmetric_conservation` (17 experiments).
> - Conservation is **structural** — enforced at reconciliation boundaries — **not
>   procedural**, per timestep. This POC tested the procedural form.
> - Measurements ran 600–1000 ticks. `big_bang` is symmetric init, which converges at
>   **tick 8450**. Everything here is transient.
> - "Attractor" was operationalised as perturbation-recovery. The corpus standard is
>   **path-independence** — different starts, one endpoint.
>
> Results are retained as lineage. The findings that survive are negative and structural:
> the engine implements no Δ, no reconciliation, and no parent/child nodes, so the
> validated model has never actually been run in it.

---

## Why this, after POC-01

POC-01 asked whether the **global ledger** is an attractor. Answer: **no** — R = 1.000,
displace it and it stays displaced. The engine conserves Q neutrally and has no restoring
mechanism at the global level.

But a restoring term must reference a **balance point**, and the ledger has none — nothing
in the dynamics knows what Q "should" be. Ξ *is* a balance point, RBF is the Recursive
**Balance** Field, and QBE drives dI/dt = −dE/dt. So the attractor claim is much more
likely about the **E↔I exchange** than about the scalar total.

This POC tests that, and asks the durability question directly: **displace the balance and
see whether it comes back — and across how much of parameter space it still does.**

## The perturbation isolates balance from the ledger

Add δ to E and subtract δ from I:

```
E → E + δ/N      I → I − δ/N      so  Q = Σ(E+I+M) is UNCHANGED
```

The ledger is untouched by construction, so anything observed here is about the balance,
not about the quantity POC-01 already resolved. That separation is the point of the design.

## Observables

Four, each recorded independently. All are engine metrics already emitted per tick — none
is invented for this experiment.

| metric | from | what it is |
|---|---|---|
| `info_fraction` | `sec_tracking` | \|I\|/(\|E\|+\|I\|) — the E/I balance; documented best SEC duty-cycle proxy (r = +0.954) |
| `balance_magnitude` | `rbf` | RBF's own balance field magnitude |
| `alpha_local_mean` | `rbf` | Tier-1 coupling; DFT target ln 2 ≈ 0.6931 |
| `lambda_local_mean` | `rbf` | Tier-1 coupling; DFT target 1 − ln 2 ≈ 0.3069 |

`alpha` and `lambda` are included because the scorecard already treats them as **coupling
attractors** — if anything in this engine has a basin, they are the strongest candidates.

## Method — the twin difference, reused

POC-01's design worked because it differenced against an identically-seeded twin, which
cancels the engine's baseline behaviour exactly. Same method here:

1. Run to a settled state (`enforce_pac = False`; enforcement is not the subject).
2. Fork. One copy untouched (**reference**), one receives the balance impulse.
3. Evolve both on identical seeds.
4. For each observable `X`, track `D(t) = X_perturbed(t) − X_reference(t)`.

`R = |median D over final 20%| / |D₀|`, per observable.

## Registered outcomes

Per observable, using POC-01's bands so the two are directly comparable:

| Outcome | Criterion |
|---|---|
| **Durable** (restoring) | `R < 0.5` — displacement decays |
| **Neutral** | `0.9 ≤ R ≤ 1.1` — displacement persists |
| **Unstable** | `R > 1.5` — displacement grows |
| **Ambiguous** | anything else — reported as such |

## Kill sentence

> **If every balance observable returns `R ≥ 0.9`, balance is not durable in this engine
> either, and "conservation is an attractor" has no support at the local level any more
> than it had at the global level.**

Recorded as the result either way. Not retried, not retuned.

## Durability across parameter space

"How durable" is two questions, and both are recorded:

1. **Does it return?** — `R` per observable, above.
2. **Over what range does it still return?** — the same measurement swept across
   parameters. An attractor with a narrow basin is a different claim from one with a wide
   one, and the width is the durability.

Swept: `quantum_pressure_coeff`, `deactualization_rate` (η), `mass_gen_coeff`,
`confluence_weight`, each at ×0.5, ×1, ×2 of default, one at a time from the default
configuration.

**The sweep is not a convergence study.** No parameter setting is privileged as correct;
this measures how wide the basin is, not which value is true. (POC-01 v1 was withdrawn for
exactly that confusion.)

## Registered invariant, not coordinates

The registered quantity is again the dimensionless **recovery ratio `R`**, plus the
**fraction of swept parameter settings that remain in the durable band** — a proportion,
not a parameter value. Not the balance values themselves, which move with every config.

## Assumptions this design does not make

- **No continuum limit.** Nothing is refined; no discretization is privileged.
- **Local leaks are not defects.** These observables are local by nature. They are graded
  only against their own unperturbed twin, never against a global expectation.
- **No claim that a DFT target value is correct.** `alpha` and `lambda` are measured
  against their own twin. Their distance from ln 2 / 1−ln 2 is recorded for context and
  is not a criterion.
- **No claim about intelligence, GAIA, or observers.**
