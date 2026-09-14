# v4: connectivity at fixed occupancy on a count deposit — the exp_31 instrument; masses in the sidecar

**Date**: 2026-09-14 · **Branch**: `feat/v4-pac-ledger-r1c` (stacked on `feat/v4-pac-ledger-r1b`, PR #12)

## What

`structure.cic_deposit(pos, box, res)` — cloud-in-cell COUNT deposit on a periodic grid, sum equal to
the particle count — and `structure.connectivity_at_occupancy(F, q)` — the largest face-connected
component of the densest fraction `q` of cells, as a fraction of them, using `percolation()`'s
labeller on a rank-selected mask. POC-12 `exp_03` records `conn_q05 / conn_q10 / conn_q20` per mark
on the CIC count field of the alive set at `matched_res(n)`, beside the legacy `percolation` and
`occupancy` (unchanged, still reported); `exp_04` floors each on uniform positions and carries the
window means into `_summary` (nan on runs that predate the key). Seven tests in
`tests/v4/test_structure_connectivity.py`. **Correction (same day):** this entry originally said the
mass draw was now saved in the sidecar; that change was uncommitted in another worktree and the
exp_31 runs do not carry it — see `.changelog/20260914_2100*_v4_ledger_virial_instrumentation.md`.

## Why

dawn-field-theory exp_30's post-mortem: reading `percolation` alone across arms compares overdense
sets at unmatched occupancy (κ = 1's is 60 % larger than gravity's — a fatter set percolates more
easily for reasons that are not structure), and the recorded field deposits masses drawn 1 ± 0.1
nearest-grid-point at ~1 particle per cell, so the 2×-mean threshold sits on the count-two boundary
and re-drawing the masses on the SAME positions moves percolation by 0.01–0.05. A rank threshold
matches occupancy by construction; a count deposit has no mass draw and no boundary. The rank
threshold stays wrong for the `is_web` verdict (`structure.py` says why) and is right for comparing
connectivity across arms. No physics change; the ledger and the arms are untouched (spec R4, R8).
