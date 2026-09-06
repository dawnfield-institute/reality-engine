# 2026-09-06 — the ledger holds the web, and the web is gravity's

Milestone R exp_29 (dawn-field-theory, sealed `43e4ebc9`) asked whether a PAC ledger lets this
substrate hold structure. Score 3/4 on the proxy, and by the letter of the seal the kill fires:
the ledgered arm at κ = 0.5 holds a web with percolation 0.51–0.76 where the unbounded engine
holds 0.06–0.10 (5.7 σ), but gravity alone holds 0.43–0.82 on the same seeds and the ledgered arm
beats it in two of three (margin 0.03 against a spread of 0.17). The web the bounded substrate
holds is gravity's own.

**What the ledger did.** Exactly what the spec says. The total `KE + U_grav + E_SEC + ΣP` is
conserved to truncation (closure ≤ 10⁻² per tick), the transfer identity holds to 10⁻⁹, the
budget binds on every particle that would grow, 91–93% of a κ = 0.5 budget is spent and not a
unit more, and the substrate settles bound: KE/|U_grav| 0.49–0.53 at κ = 0.5, 0.29–0.35 at κ = 1,
against 10.7–18.3 with no ledger. The step never shrinks below `dt_ref` at κ ≤ 2 — there is no
detonation. exp_28's relaxation oscillator is gone, and the virial arithmetic stated before the
run (bound through κ = 1, ordered above it, U-shaped with the minimum at κ = 1) is what all three
seeds show.

**What the pressure did.** Nothing good, at any budget. Percolation falls monotonically with κ:
0.51–0.76 → 0.40–0.53 → 0.15–0.28 → 0.06–0.10 at κ = 0.5 / 1 / 2 / ∞. Below κ = 1 the pressure does
net negative work (−1.2 P₀ at 0.5: a restoring force that neither builds nor breaks); above it,
it injects and disperses. An entropy-driven repulsion sourced by density fires where matter has
gathered and pushes it apart — bounded, it is close to inert; unbounded, it is exp_28's
detonation. The sign of what it does to structure is fixed by its form, not its budget.

**What this closes and opens.** The registered mapping — a per-particle budget priced by exp_09's
pair energy — is retired as *the object that holds structure*, as the seal says. The ledger
discipline survives as a gated instrument: `pac_kappa`, `sec_pair_energy`, the transfer, the
conserved total, all bit-identical to the old path when off. The next object is the pressure's
**form**: the fracton SEC functional's smoothing term `β∇²A` penalises entropy *gradients* rather
than entropy *magnitude*, which on particles is a force that holds filaments rather than
dispersing cores. `.spec/challenges.md` C4.1 option 2 proper. R2 (severance on a bound substrate)
can now run on a substrate that is actually bound.

**Caught on the way.** A test committed failing behind a pipe that masked pytest's exit code
(amended within the hour; `set -o pipefail` in every chain since). The gradient check needs
float64. The anchor against exp_28's recorded baseline needs `XI_ANALYTIC/PHI` exactly, not the
rounded 0.6541 — with it the inert path reproduces exp_28's B0 seed-1 marks to 0.00. Three
registration clauses moved on calibration numbers before the seal, disclosed in its §0.

n = 4000: see the outcomes journal in dawn-field-theory (`milestone-r/journals/2026-09-06_exp29_outcomes.md`).
