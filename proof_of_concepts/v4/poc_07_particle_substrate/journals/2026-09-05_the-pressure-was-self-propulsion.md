# The pressure was self-propulsion (2026-09-05, evening)

Found by the derived-sink design pass while adding momentum to the energy ledger. Layers forward
on [`2026-09-05_the-integrator-owns-dt.md`](2026-09-05_the-integrator-owns-dt.md); nothing earlier
is edited.

## The defect

`SECPressure`, inherited from `gravity_from_maxwell_pac/exp_09`, computed for every ordered pair

    F_i <- sec * (S_i - S_j) * exp(-r/r0) * u_ji        (u_ji: unit vector from j to i)

Under the swap i <-> j the entropy difference flips sign **and** the unit vector flips sign, so the
force on j from i is the *same vector* as the force on i from j. Checked on two particles, S = 5
and 1, six apart on x: F_i = F_j = (-2.195, 0, 0). The antisymmetric part of the pair interaction,
½(F_i - F_j), is identically zero. The term was never a pressure between particles; it was a
self-propulsion field in which a hot particle flees its cold neighbour and the cold neighbour is
dragged after it, and every pair injected net momentum 2F into the box.

This is exp_09's literal rule, and exp_09's published web numbers were produced with it. It is
therefore not a transcription error in this substrate; it is the reference mechanism.

## Why there is no sign fix

Any pair force along the separation with a magnitude that is *antisymmetric* in (i, j) — such as
S_i - S_j — is symmetric as a pair (both particles pushed the same way). A third-law pair force
needs a magnitude *symmetric* in (i, j). So the repair is a choice of pair law, not a sign, and
it is a physics choice. Two candidates were put to Peter:

| law | magnitude | what it is |
|---|---|---|
| **mean-entropy repulsion** (chosen) | `sec * (S_i + S_j)/2 * exp(-r/r0)`, apart | a pressure: strength set by local entropy density; a dense hot region pushes outward from its interior — "the counter-force that opens voids" |
| contrast repulsion | `sec * |S_i - S_j| * exp(-r/r0)`, apart | an interface force: a uniform-entropy clump has no interior pressure |

Momentum is conserved by construction and tested (`tests/v4/test_pressure_momentum.py`: the pair
force is antisymmetric; a 200-particle cloud under pressure alone keeps its total momentum).

## What this reclassifies

- **exp_04** (`results/exp_04_integrator_owns_dt_20260903_*.json`) and the C4.1 "relaxation
  oscillator" characterisation were measured on the self-propelling rule. They stand as the
  record of that rule. The substrate under the corrected force is characterised fresh in POC-11
  before anything is sealed — nothing from exp_04 is carried into a pre-registration as a prior.
- POC-07/08/09/10's numbers carry the same caveat, on top of the clamp caveat already recorded.
- `gravity_from_maxwell_pac/exp_09`'s web was produced with a momentum-injecting force. Flagged
  upstream as a correction candidate; not changed here.

Also in this commit: `PHI`, `LN2`, `XI_ANALYTIC` are imported named from fracton (the repo venv
has fracton as an editable install), with the literal mirrors as an import fallback only.
