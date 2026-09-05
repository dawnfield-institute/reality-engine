# v4 SEC pressure: the pair force was self-propulsion; replaced by a third-law pair law

**Date:** 2026-09-05 (evening) · **Type:** fix (physics-correctness; a choice, recorded) ·
**Branch:** `feat/v4-derived-sink` · **Journal:** `poc_07_particle_substrate/journals/2026-09-05_the-pressure-was-self-propulsion.md`

- **Defect.** `SECPressure`'s inherited rule `sec * (S_i − S_j) * exp(−r/r0)` along the unit vector
  from j to i gives the force on j from i as the *same* vector as the force on i from j: the
  antisymmetric part of the pair interaction is identically zero. Every pair injected net momentum
  2F. Checked numerically (S = 5, 1; six apart: F_i = F_j = −2.195 x̂). This is exp_09's literal rule.
- **No sign fix exists**; a third-law pair force needs a magnitude symmetric in (i, j). Peter chose
  **mean-entropy repulsion**: `sec * (S_i + S_j)/2 * exp(−r/r0)`, always apart — a pressure whose
  strength is the local entropy density. Alternative recorded: contrast repulsion `|S_i − S_j|`.
- **Tests.** `tests/v4/test_pressure_momentum.py` (2): the pair force is antisymmetric; a
  200-particle cloud under pressure alone conserves total momentum. Both fail on the old rule.
  Suite: 153.
- **Constants.** `PHI`, `LN2`, `XI_ANALYTIC` now imported named from fracton (editable install in
  the repo venv); literal mirrors only as an import fallback.
- **Reclassified as lineage:** exp_04 and the C4.1 "relaxation oscillator" characterisation
  (measured on the self-propelling rule); POC-07/08/09/10 numbers carry the same caveat. The
  substrate under the corrected force is characterised fresh in POC-11 before any seal.
  `gravity_from_maxwell_pac/exp_09` is flagged upstream as a correction candidate; not changed.
