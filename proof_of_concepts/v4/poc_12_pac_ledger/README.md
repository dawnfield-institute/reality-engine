# POC-12 (v4): The ledger on particles

**Status**: completed 2026-09-06 · **Pillar**: PAC / SEC · **Registered in**: dawn-field-theory Milestone R exp_29 (sealed `43e4ebc9`) · **Spec**: `.spec/v4-pac-ledger.spec.md` · **Result**: 3/4 — the ledger holds the web; whose web it is depends on the size

## Result

The ledger works at both sizes. With a budget of half the binding energy the substrate settles
**bound** (KE/|U_grav| 0.49–0.54, against 10.7–22.3 unbounded), the total `KE + U + E_SEC + ΣP` is
conserved, the transfer is exact, the budget binds on every particle that would grow, and the web
survives: whole-set percolation 0.51–0.76 (proxy) and 0.44–0.78 (n = 4000) over t ∈ [10, 15] where
the unbounded engine holds 0.03–0.10 (4.8–5.7 σ). Against gravity alone the registered arm wins 2/3
on the proxy (0.2 σ) and 3/3 at n = 4000 by 1.1 σ against a 2 σ bar: T1 fails at both sizes and the
registered kill fires — the mapping is retired as the object that holds structure beyond gravity
at κ = 0.5 with the proxy deciding. What the seal does not score: at n = 4000 the ledgered engine at
κ = 1 holds 0.70–0.80, above gravity alone in every seed by 2.8 σ (post hoc), and collapses at κ = 2;
on the proxy — whose pressure range exceeds half its box — it never adds. The virial arithmetic
stated before the run held in all six seeds. What survives is the ledger discipline as a gated
instrument; what is next is R1b, κ = 1 at n = 4000 on fresh seeds with the box-to-range ratio
declared; the pressure's form stays a candidate behind it. Journal:
`journals/2026-09-06_the-ledger-holds-the-web.md`; scoring and outcomes in dawn-field-theory
`milestone-r/journals/2026-09-06_exp29_outcomes.md`.

## Why

POC-11 (Milestone R exp_28, 0/4) showed the substrate cannot hold structure and its ledger showed
why: over a baseline run the SEC pressure does 670,000 to 1,140,000 of work against a binding
energy of about 63,000, while gravity's net work is about 1,000. The pressure is a source, not a
transfer. In the house's own terms — SEC local, PAC global — the local step is free to inject; what
the substrate lacks is the **ledger**: a potential to draw from, a bounded Δ, a reconciliation. The
PAC necessity proof says a system without one cannot hold structure, and this one did not.

The design pass located the source exactly. At fixed entropy the pressure is *already* the gradient
of a pair energy, `V_ij = sec·(S_i+S_j)/2·r0·e^{−r/r0}`; only the cutoff breaks it. The whole
non-conservation is the entropy ratchet: entropy grows where pairs are close and the pair energy is
large, and decays after dispersal where it is small. Booked tick by tick, the pair energy created by
entropy change is 689,700 against 667,300 of pressure work. The engine's fuel is energy that
entropy growth creates for free.

## What is built here

| piece | what it is | derived / declared |
|---|---|---|
| exact pair energy | the shifted kernel `K = sec·r0·(e^{−r/r0} − e^{−2})`, same force, zero at the cutoff | the force's own potential |
| potential budget | `P_i(0) = κ·\|U_grav(0)\|·m_i/Σm`; entropy growth debits `(∂E_SEC/∂S_i)ΔS_i`, decay credits it | κ **declared and swept**, a ratio never a coordinate |
| the conserved total | `KE + U_grav + E_SEC + ΣP`, closure reported every tick | the ledger |
| κ = 0 | entropy cannot grow: gravity only | the engine removed (control) |
| κ = ∞ | today's substrate | the unbounded engine (control) |

## Scripts

- `exp_01_ledger_gates.py` — the known-answer gates (KA-0..iii and the anchors). Must PASS first.
- `exp_02_budget_calibration_proxy.py` — where and when the budget binds across κ, on the proxy.
  Computes **no structure metric** by construction: calibration, not tuning.
- `exp_03_ledger_arms.py` — one run per invocation (`--kappa 0|0.5|1|2|inf --seed --size`), one
  JSON per run, whole-set structure at matched resolution every unit of simulated time.
- `exp_04_aggregate.py` — the grid JSON with per-run hashes and the commit; the random-field floor at
  the same occupancy beside every percolation. Aggregates only; **scoring lives in dawn-field-theory**
  (`milestone-r/scripts/exp_29_pac_ledger.py`).
- `run_grid.sh proxy | full` — the grid, phase by phase.

## The exp_31 instrument (2026-09-14): connectivity at fixed occupancy, on a count deposit

exp_30's post-mortem found two things wrong with reading `percolation` alone across arms. (1) It
compares the overdense sets at whatever occupancy each arm happens to have, and the arms differ:
gravity alone and κ = 0.5 occupy ~0.09 of cells above 2× the mean, κ = 1 ~0.15 — a fatter
overdense set percolates more easily for reasons that are not structure. (2) `density_field`
deposits MASS nearest-grid-point, masses are drawn 1 ± 0.1, and at ~1 particle per cell the 2×-mean
threshold sits at a count of two, so a two-particle cell lands on either side of it at random:
re-drawing the masses on the SAME positions moves percolation by 0.01 (κ ≤ 0.5) to 0.05 (κ = 1.5).

exp_03 therefore also records, per mark, `conn_q05`, `conn_q10`, `conn_q20` =
`structure.connectivity_at_occupancy(structure.cic_deposit(pos, box, res), q)`: the largest
connected component of the densest q of cells (face connectivity, the same labeller as
`percolation`), on a cloud-in-cell COUNT field of the alive set at `matched_res(n)`. Occupancy is
matched across arms by construction; there is no mass draw and no threshold boundary. exp_04
floors each on uniform positions and carries their window means into `_summary`. The mass draw
is now saved in the position sidecar (`mass`) so the legacy marks can be reproduced. `percolation`
and `occupancy` stay recorded and reported; `is_web` still uses exp_09's thresholds (a rank
threshold is wrong for that verdict — `structure.py` says why — and right for comparing
connectivity across arms). Tests: `tests/v4/test_structure_connectivity.py`.

## What this POC does not claim

Whether a ledgered substrate holds structure, or at which κ. That is registered, sealed and scored in
Milestone R exp_29, with the unbounded engine and the engine removed as controls, and a prediction
from virial arithmetic stated before the run: bound below κ ≈ ½, marginal at 1, unbound at 2.
