# POC-12 (v4): The ledger on particles

**Status**: active · **Pillar**: PAC / SEC · **Registered in**: dawn-field-theory Milestone R exp_29 · **Spec**: `.spec/v4-pac-ledger.spec.md`

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

## What this POC does not claim

Whether a ledgered substrate holds structure, or at which κ. That is registered, sealed and scored in
Milestone R exp_29, with the unbounded engine and the engine removed as controls, and a prediction
from virial arithmetic stated before the run: bound below κ ≈ ½, marginal at 1, unbound at 2.
