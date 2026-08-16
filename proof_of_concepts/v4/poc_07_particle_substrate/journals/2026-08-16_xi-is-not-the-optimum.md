# 2026-08-16: Ξ is not the optimal operating point, in either convention

exp_10 reports two things. The first holds. The second is not supported by its own data.

> **NO discrete phase transition — SEC is CONTINUOUS control.** ✔ supported
>
> **Ξ ≈ 1.057 is not a phase transition point, but the OPTIMAL OPERATING POINT for maximum
> structural complexity.** ✘ not supported

---

## What exp_10's own data shows

21 arms, `sec_balance` from 0.30 to 1.30, **one seed**. Checking every metric it recorded:

| metric | argmax at | is Ξ a local max? | value at Ξ |
|---|---|---|---|
| density_cv | **1.300** (sweep endpoint) | no | 2.0903 |
| void_fraction | 1.250 | no | 0.7675 |
| filament_fraction | **0.300** (sweep endpoint) | no | 0.0466 |
| clustering | 0.700 | no | 0.6417 |
| max_entropy | 1.150 | no | 48.2742 |

**No metric peaks at Ξ, and Ξ is not a local maximum of any of them.** Two of the five maxima
sit at the boundaries of the swept range, so the optimum — if one exists — was never bracketed.
`density_cv`, the metric closest to "structural complexity", rises monotonically to the upper
endpoint: 1.793 → 2.157.

The first claim is solid and the data supports it clearly: the trends are smooth, with nothing
resembling a transition anywhere in the range. It is only the "optimal operating point" gloss
that the numbers do not carry.

## The rebuild, in the other convention

Sweeping `sec_balance` in the v4 particle substrate — 9 arms, **5 seeds**, 4000 particles,
1000 steps:

| sec_balance | peak CV | peak percolation |
|---|---|---|
| 0.300 | **6.254 ± 0.198** | 0.1057 ± 0.0039 |
| 0.700 | 5.914 ± 0.205 | 0.1122 ± 0.0236 |
| 1.000 | 5.661 ± 0.206 | 0.1322 ± 0.0181 |
| **1.057 (Ξ)** | 5.656 ± 0.230 | 0.1402 ± 0.0340 |
| 1.150 | 5.637 ± 0.141 | **0.1615 ± 0.0406** |
| 1.600 | 5.499 ± 0.101 | 0.1546 ± 0.0329 |

CV falls **monotonically**, maximum at the lower endpoint, beating the Ξ arm by **4.40σ**.
Percolation rises and then flattens above ~0.9, and its best arm beats Ξ by only **0.90σ** —
not resolved.

**Important caveat: this does not test exp_10's system.** This substrate uses attractive
gravity, and exp_10 (like exp_09 and exp_11) uses the repulsive convention — see POC-09, where
transcribing exp_11 literally reproduces its published numbers to three significant figures.
So the two sweeps are different dynamical systems.

What they have in common is the shape of the answer: **CV is monotonic in `sec_balance` in both
conventions, with the maximum at an endpoint, and Ξ is not distinguished in either.** It rises
with SEC in the repulsive convention and falls in the attractive one, which is what flipping the
sign of the dominant force should do.

## Reading

This is a bearing, not a demolition. exp_10's headline — SEC is continuous control with no
phase transition — is exactly what its data shows and it stands. The Ξ optimum is an
interpretation laid over a monotonic single-seed sweep whose maximum sits on the boundary.

It also fits the pattern from the rest of this round: the corpus's particle results measure
contrast (`density_cv`, `void_fraction`) very carefully, and those are one-point statistics.
Where a claim needs an interior optimum, or connectivity, the measurement to support it was not
among the ones taken.

Testing exp_10 properly means sweeping in its own convention, past 1.3 so the optimum is
bracketed, with seeds. That is cheap and is the obvious follow-up.
