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

## The fair test: exp_10's own convention, bracketed, five seeds

exp_03 transcribes exp_10's dynamics literally with its own config (n=2000, box=100, r0=6,
g=0.8, 600 steps), sweeps to **2.5** so a maximum near 1.3 would be bracketed rather than
sitting on the boundary, and runs five seeds:

| sec_balance | peak CV | peak percolation |
|---|---|---|
| 0.300 | 2.278 ± 0.011 | **0.0166 ± 0.0029** |
| 0.700 | 2.468 ± 0.009 | 0.0152 ± 0.0021 |
| **1.057 (Ξ)** | 2.565 ± 0.025 | 0.0141 ± 0.0016 |
| 1.300 | 2.620 ± 0.023 | 0.0121 ± 0.0009 |
| 2.000 | 2.682 ± 0.012 | 0.0112 ± 0.0007 |
| **2.500** | **2.729 ± 0.040** | 0.0116 ± 0.0012 |

**CV rises monotonically to 2.5 and is STILL at the endpoint.** Doubling the swept range did not
bracket an optimum — it moved the maximum further out. Ξ is beaten by **7.70σ**. There is no
interior optimum anywhere in 0.3–2.5, so "optimal operating point" has nothing to attach to.

exp_10's underlying trend replicates cleanly: CV rises with `sec_balance` in its convention,
as its own data showed. What does not replicate is the interpretation laid on top.

**And percolation runs the other way.** It is highest at the LOWEST SEC balance (0.0166 at 0.3)
and falls as contrast rises — best arm beats Ξ by 1.70σ, marginal, but the trend is clean and
monotone. Contrast and connectivity are not merely decoupled in this system; they **trade
off**.

**No setting produces connectivity at all.** Percolation stays within 0.011–0.017 across the
entire sweep, against a 2D white-noise floor of ~0.003 and a synthetic web's 1.000. There is no
value of `sec_balance` for which exp_10's system builds a connected structure.

## Reading

This is a bearing, not a demolition. exp_10's headline — SEC is continuous control with no
phase transition — is exactly what its data shows and it stands. The Ξ optimum is an
interpretation laid over a monotonic single-seed sweep whose maximum sits on the boundary.

It also fits the pattern from the rest of this round: the corpus's particle results measure
contrast (`density_cv`, `void_fraction`) very carefully, and those are one-point statistics.
Where a claim needs an interior optimum, or connectivity, the measurement to support it was not
among the ones taken.

Testing exp_10 properly meant sweeping in its own convention, past 1.3, with seeds — done above
as exp_03. The claim does not survive it: CV is monotonic to 2.5 and still climbing, Ξ loses by
7.70σ, and percolation trends the opposite way while never leaving the noise floor.
