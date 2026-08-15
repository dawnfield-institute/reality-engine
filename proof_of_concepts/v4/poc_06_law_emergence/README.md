# POC-06 (v4): What laws does v3 actually have?

**Status**: completed · **Pillar**: PAC

## Why

`docs/theory/law_emergence.md` states the project's purpose:

> Reality Engine is not just a simulator. It's a **physics discovery engine.**
> We don't program the laws. We discover what emerges.

The detectors that would answer that were designed, built for v1, and left in `archive/v1/` —
`law_quantifier.py`, `spikes/law_discovery/`, `analyzers/laws`, `scripts/quantify_laws.py`.
Last run November 2025. Never against v3.

The scorecard doesn't answer it. It asks *do the constants come out right* — 7/13 — and cannot
distinguish physics emerging from physics implemented and tuned.

## The design choice that matters

**A law detector run on a system whose laws are implemented will rediscover the
implementation.** The v1 detector's single "discovery" was `dI/dt = −dE/dt` at correlation
exactly −1.0 — which is `QBEOperator`, hardcoded. Exactly, not approximately, because it's an
identity.

So every result here is labelled:

| | |
|---|---|
| **ENFORCED** | an operator maintains it — finding it proves nothing |
| **EMERGENT** | nothing implements it and it holds anyway |
| **ABSENT** | expected, not found |

The enforced list is written out explicitly in `law_detector.py` rather than inferred, so it
can be argued with.

## Result

**128×128, 12000 ticks, past the 8450 convergence point.**

```
0 EMERGENT · 1 ENFORCED · 17 ABSENT
```

The one law is `E+I+M` at **CV 2.5e-15** — machine precision, because `NormalizationOperator`
corrects it every tick. The answer key, not an answer.

Seventeen other candidates drift: the full ledger with the Δ term (`E+I+M+P`), every field
individually, `E²+I²`, `E³+I³+M³`, `E·I`, total disequilibrium, both momentum-like objects,
mass entropy.

**Zero persistent peaks across 600 samples.** No objects, so no force law — not a weak one, but
nothing stable enough for a force to hold *between*. That's the same fact as POC-05's
percolation result and the absence of any velocity field in v3, arriving by a third route.

**Second law: entropy decreases on 36% of ticks**, with 16,384 cells, net slope +1.8e-5. The
honest reading isn't that the second law fails — it's that there's no arrow of time at the tick
scale; the entropy signal is fluctuation with a faint drift on top.

Nine months apart, on two engine generations, the discovery engine has returned the same
verdict: **one law, and it's the one that was hardcoded.**

## What this is and isn't

It's a **bearing, not a verdict on the framework**, and it coheres with what the corpus already
says. Spikes 09–13: PAC is global, SEC is local. One global conservation law and no others is
what the theory predicts — finding exactly that is the theory being right, not the engine being
wrong.

And law emergence demonstrably works elsewhere in the corpus. PR #171: **Dynkin topology alone
fixes the turbulence cascade exponent to 0.26% of Kolmogorov** with nothing tuned, and the
long-unexplained 3.3% miss turns out to be an A-family artifact. M15 exp_06 *derives* the
momentum operator rather than declaring it. The mechanism works; it's the engine that hasn't
shown it.

## Retracted in part

**The force arm of this verdict is an instrument limit, not a finding.** POC-07 established it:
`fit_force_law` recovers r^−2, r^−3 and r^−1 exactly (R² = 1.0000) from a clean two-body orbit,
and returns **R² = 0.0005** on a 4000-particle system whose force law is known by construction.
In a dense system each particle's acceleration sums over many neighbours, so a nearest-neighbour
projection measures mostly the others.

The conservation arm is unaffected, and POC-07 is what calibrated it: it correctly finds mass
conserved in the particle substrate and nothing conserved here.

## Remaining caution

The candidate list is mine — a conserved quantity in a form I didn't write down reads as ABSENT.
The peak tracker needs local maxima above 2× mean persisting 40 samples, so an object that
breathes or drifts quickly would be missed.

## Running it

```
python proof_of_concepts/v4/law_detector.py                     # calibration must PASS first
python proof_of_concepts/v4/poc_06_law_emergence/scripts/exp_01_what_laws_does_v3_have.py
```

Calibration covers: force-law recovery at three known exponents, energy and momentum
conservation in N-body, a non-conserved control, the second law in a system that has it, and
that the ENFORCED tag actually fires.
