# POC-01 (v4) — Is the global PAC ledger an attractor?

**Status:** COMPLETE — **NEUTRAL**. The ledger is conserved but non-restoring: R = 1.000
across all 18 trials. The engine has no conservation-as-attractor mechanism. See
[`journals/2026-08-11_ledger-is-neutral.md`](journals/2026-08-11_ledger-is-neutral.md).
**Registered:** 2026-08-11
**Generation:** v4 proof-of-concept 1

> **Registration v1 is withdrawn.** It asked whether the residual survives grid/timestep
> refinement. That test presupposes the continuum is ground truth and the discrete system
> approximates it — which DFT denies: Ψ(k) = Ψ(k+1) + Ψ(k+2) is a recursion, φ comes from
> the discreteness, cascade depth is a count. It also measured the scalar sum rather than
> the balance RBF and QBE govern, conflated local with global, and never operationalised
> "attractor". Preserved in `journals/2026-08-11_kill-sentence-fired.md` with the reasons.

---

## The claim being tested

> **Conservation is an attractor, not a constant.** (Peter, 2026-08-11)

And the structure that goes with it:

> **SEC is local; only the global PAC ledger balances.** Local leaks are not defects.

## What "attractor" means operationally

An attractor is defined by its **basin**: displace the system and it returns. Not "stays
constant" — that is the property the hypothesis explicitly denies. So the test is a
**perturbation-recovery** test, which requires no assumption about continuum limits,
convergence, or which discretization is "true".

**Perturb the global ledger. Measure whether it comes back.**

## Hypothesis

**H1.** With enforcement disabled, an impulse applied to the global PAC ledger **decays** —
the ledger returns toward the trajectory it would have followed unperturbed.

## Kill sentence

> **If the displacement persists undiminished, or grows, the global ledger is not an
> attractor — it is either neutrally conserved or unstable, and "conservation is an
> attractor" gets no support from this engine.**

Recorded as the result either way. Not retried, not retuned.

## Design

1. Run to a settled state with `enforce_pac = False`. No enforcement anywhere in this
   experiment — the correction *is* the thing whose necessity is in question.
2. Fork the state. One copy continues untouched (**reference**). The other receives a
   single impulse `ΔQ` to the global ledger (**perturbed**).
3. Both run on identically seeded dynamics.
4. Measure `D(t) = Q_perturbed(t) − Q_reference(t)`, the displacement.

Differencing against a twin removes the engine's own drift entirely — whatever the
unperturbed dynamics do, both copies do it. **This is why the design needs no position on
whether that drift is physical or numerical.** It was the unresolvable question in v1 and
it is not on the critical path here.

### Registered outcomes

`D₀` is the displacement immediately after the impulse; `D_end` its median over the final
20% of ticks. Recovery ratio `R = |D_end| / |D₀|`.

| Outcome | Criterion | Reading |
|---|---|---|
| **Attractor** | `R < 0.5` | displacement decays; ledger restores |
| **Neutral** | `0.9 ≤ R ≤ 1.1` | conserved but non-restoring — no basin |
| **Unstable** | `R > 1.5` | displacement grows |
| **Ambiguous** | anything else | report as such; do not round toward the hypothesis |

Also recorded, not registered as criteria: the recovery timescale (fitted `D ~ exp(−t/τ)`
where recovery occurs), and whether `R` depends on impulse sign or magnitude.

## Local and global are measured separately, and not graded against each other

Per the corpus rule, **local leaks are not defects.** Two distinct observables:

- **Global** — the ledger `Q = Σ(E+I+M)`. This is where conservation-as-attractor is
  claimed and where the criteria above apply.
- **Local** — per-cell balance, and the `Ξ`-relevant E/I exchange RBF and QBE drive.
  Recorded for context, **graded against nothing.** A local leak is expected behaviour.

Reporting local deviation as error is the mistake v1 made. Whether the two views agree is
itself a result — that is the perspective-dependence, and it is observed, not assumed.

## Assumptions this design does **not** make

Stated explicitly, because v1 failed by making them silently:

- **No continuum limit.** Nothing is refined toward `dt → 0` or `h → 0`, and no
  discretization is privileged as "true". Config is varied only to check the phenomenon
  is not unique to one setting — reported as robustness, never as convergence.
- **No claim that drift is error.** The twin-difference makes the engine's baseline drift
  irrelevant to the measurement.
- **No claim that scale-dependence is a defect.** Running couplings are DFT physics; if
  `R` varies with resolution that is recorded as a finding, not a failure.
- **No claim about intelligence, GAIA, observers, or parent universes.** A restoring
  ledger would be *consistent* with the parent/sibling reading. It would not be evidence
  for it.

## Registered invariant, not coordinates

*Registered relations survive; registered coordinates die.* The registered quantity is the
**recovery ratio `R`** — dimensionless, and independent of the impulse magnitude, the
ledger's absolute value, and the config. Not `D₀`, not `Q`, not any drift rate.

## Files

```
poc_01_conservation_attractor/
├── README.md          this registration (v2)
├── meta.yaml
├── scripts/
│   ├── exp_01_refinement_sweep.py   v1, superseded — kept as lineage
│   └── exp_02_ledger_perturbation.py
├── results/
└── journals/
```
