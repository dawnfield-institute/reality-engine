# 2026-09-06 — the ledger holds the web; whose web it is depends on the size

Milestone R exp_29 (dawn-field-theory, sealed `43e4ebc9`) asked whether a PAC ledger lets this
substrate hold structure. Score 3/4 at both sizes, and by the letter of the seal the kill fires:
the ledgered arm at κ = 0.5 holds a web (percolation 0.51–0.76 on the proxy, 0.44–0.78 at n = 4000)
where the unbounded engine holds 0.03–0.10 (4.8–5.7 σ), but against gravity alone it wins 2/3 on
the proxy (0.2 σ) and 3/3 at n = 4000 by 1.1 σ against a 2 σ bar.

**What the ledger did.** Exactly what the spec says, at both sizes. The total
`KE + U_grav + E_SEC + ΣP` is conserved to truncation (closure ≤ 10⁻²), the transfer identity holds
to 10⁻⁹, the budget binds on every particle that would grow, 91–96% of a κ = 0.5 budget is spent
and not a unit more, and the substrate settles bound: KE/|U_grav| 0.49–0.54 at κ = 0.5, 0.29–0.36
at κ = 1, against 10.7–22.3 with no ledger. The step never shrinks below `dt_ref` at κ ≤ 2 — there
is no detonation. exp_28's relaxation oscillator is gone, and the virial arithmetic stated before
the run (bound through κ = 1, ordered above it, U-shaped with the minimum at κ = 1) is what all six
seeds show.

**What the pressure did depends on the size.** On the proxy (box 37.8, pressure range 2r0 = 20 —
more than half the box) the bounded engine adds nothing to gravity's web at any κ and removes it
monotonically: 0.51–0.76 → 0.40–0.53 → 0.15–0.28 at κ = 0.5 / 1 / 2 against gravity's 0.43–0.82. At
n = 4000 (box 60, the range a third of it) it **adds**: κ = 0.5 above gravity in every seed
(0.62 / 0.44 / 0.78 vs 0.40 / 0.32 / 0.61), **κ = 1 at 0.76 / 0.70 / 0.80, above gravity in every seed
by 0.315 against a pooled σ of 0.113 — 2.8 σ, post hoc, unregistered** — and κ = 2 collapses to
0.09–0.14. A bounded, density-sourced repulsion is not structurally inert: on a box large compared
with its range it adds web up to about one binding energy of budget and destroys it beyond.

**What this closes and opens.** The registered mapping is retired as *the object that holds
structure beyond gravity at κ = 0.5 with the proxy deciding*, as the seal says. The ledger
discipline survives as a gated instrument (`pac_kappa`, `sec_pair_energy`, the transfer, the
conserved total, bit-identical to the old path when off). The next registration is R1b: κ = 1 at
n = 4000 on fresh seeds, with the box-to-range ratio declared, the proxy retired as a decider for
pressure-range questions. The pressure's *form* (the SEC functional's gradient-penalising term)
stays a candidate, behind R1b. R2 (severance on a bound substrate) can run on κ = 1 at n = 4000.

**Caught on the way.** A test committed failing behind a pipe that masked pytest's exit code
(amended within the hour; `set -o pipefail` in every chain since). The gradient check needs
float64. The anchor against exp_28's recorded baseline needs `XI_ANALYTIC/PHI` exactly — with it
the inert path reproduces exp_28's B0 seed-1 marks to 0.00. Three registration clauses moved on
calibration numbers before the seal, disclosed in its §0. The proxy-only draft of this journal
said the pressure "can only take structure away"; the full grid contradicted it within the hour
and the journal was rewritten before anything was pushed. The registration's omission — no
declared box-to-range ratio — is the lesson carried forward.

Outcomes and scoring: dawn-field-theory `milestone-r/journals/2026-09-06_exp29_outcomes.md`.
