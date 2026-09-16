# v4: the edge instrumented — per-particle work by force, pair-form virial terms, the gross ledger legs; masses, entropies and works in the sidecar; the exploratory κ sweep

**Date**: 2026-09-14 · **Branch**: `feat/v4-ledger-virial` (stacked on `feat/v4-pac-ledger-r1c`, PR #13)

## What

No physics change (spec R4, R8): every quantity is a re-partition of numbers the engine already
computes, and every one is an identity with a test (`tests/v4/test_ledger_virial.py`).

- **Per-particle cumulative work by force** (`state.work_g_i`, `state.work_p_i`): the integrator's
  exact discrete kick identity applied row by row; the sums equal `work_gravity_cum` /
  `work_pressure_cum`. Summaries per tick: `work_pressure_pos_frac`, `work_pressure_pos_sum`,
  `work_pressure_neg_sum` — the LOCAL form of the edge (which particles carry positive cumulative
  pressure work).
- **Pair-form virial terms**: `virial_gravity = −Σ_{i<j} r_ij|f^g_ij|` (attractive, negative) and
  `virial_pressure = Σ_{i<j} r_ij|f^p_ij|` (repulsive, positive), from the same `r` and `mag` the
  forces use; origin-invariant on the torus because ΣF = 0.
- **Gross ledger legs**: `transfer_growth` (P → A, growth paid) and `transfer_credit` (A → P, decay
  credited), per tick and cumulative; growth − credit = the net `sec_transfer` exactly.
- **Sidecar**: `mass`, and per mark `S{i}` (entropy), `U{i}` (per-particle potential), `wp{i}`, `wg{i}`
  (cumulative works). The legacy marks and the pressure virial can now be reproduced after the fact.
- `exp_04` carries window means of the virial terms and the local-sign fraction, and the end
  values of the gross legs and the guard loss, into `_summary`.
- `results/full_explore/`: the exploratory κ sweep {0.25, 0.75, 1.25} × seeds 1–3 run today on
  `feat/v4-pac-ledger-r1b` with an uncommitted one-line change (masses in the sidecar); the runs
  record `87a1e51`. Exploring; feeds dawn-field-theory exp_31's §0 and the edge scoping note.

## Why

dawn-field-theory `internal/dft/2026-09-14_edge_scoping.md` §3–§4 and §7: the ledger identities
hold on every recorded run (ΔE_SEC = T − W_p; ΔKE = W_g + W_p except a 5–7 % Courant-guard loss on
the plateau arms), so the edge — the sign change of the pressure's net work between κ = 1 and 1.25 —
is the crossing T(κ) = ΔE_SEC(κ). To derive it the virial terms are needed (they are not
proportional to U_grav or E_SEC for these kernels; KE/|U| ≈ ½ is a boundedness proxy, not a virial
ratio), and to test its local form the per-particle work sign is needed. Neither was recorded.

## Correction carried here

The connectivity changelog and the POC-12 README said masses were saved in the sidecar from
`5f5d690`; they were not (the change lived uncommitted in another worktree). Corrected in both;
saved from this branch onward.
