# v4 Particle Substrate — the PAC ledger on particles

**Status:** specified 2026-09-06, branch `feat/v4-pac-ledger` (stacked on `feat/v4-derived-sink`,
PR #10) · **Registered in:** dawn-field-theory Milestone R exp_29 · **Companion:**
`v4-derived-sink.spec.md` (the energy ledger and severance this builds on) ·
**Scope:** the potential budget, the exact pair energy, and the conserved total. Nothing here is a
physics claim; the claim is exp_29's.

## Overview

exp_28 (sealed `bf833113`, 0/4) showed the substrate cannot hold structure and the ledger showed
why: over a baseline run the SEC pressure's work is 670,000–1,140,000 against a binding energy of
~63,000, while gravity's net work is ~1,000. The design pass of 2026-09-06 (proxy, seed 1,
read-only) located the source exactly:

- At fixed entropy the pressure **is already a gradient**. The force magnitude
  `sec_balance · (S_i+S_j)/2 · e^{−r/r0}` is `−dV_ij/dr` for the pair energy
  `V_ij = sec_balance · (S_i+S_j)/2 · r0 · e^{−r/r0}`. Only the hard cutoff at `2 r0` breaks
  conservation (`V(2r0) ≠ 0`); the **shifted** pair energy `K_ij = sec_balance · r0 ·
  (e^{−r/r0} − e^{−2})` on `r < 2 r0` has the same force and vanishes at the cutoff.
- The whole non-conservation is the **entropy ratchet**: entropy grows where pairs are close
  (`K` large) and decays after dispersal (`K ≈ 0`). Booked tick by tick,
  `Σ_i (∂E_SEC/∂S_i)·ΔS_i = 689,700` against pressure work `667,300` (ratio 1.03); `E_SEC`
  peaks at 1.46 × 10⁶ at t ≈ 4.5 and converts to kinetic energy as the clump expands.

In the house's terms (SEC local, PAC global): the local step is free to inject; what the substrate
lacked is the **ledger** — a potential to draw from, a bounded Δ, a reconciliation. This spec adds
it. A per-particle potential budget pays for entropy growth at the price the pair energy sets, is
repaid on decay, and the total `KE + U_grav + E_SEC + ΣP` is conserved.

## Requirements

- **R1 — the pair energy is exact.** `sec_pair_energy(s, c)` returns `E_SEC = ½ Σ_{i≠j} s_pair_ij K_ij`
  over retained pairs (`pair_alive`) and `∂E_SEC/∂S_i = ½ Σ_j K_ij`, with `K_ij` the shifted
  kernel above. The force `SECPressure` applies is `−∇_i E_SEC` at fixed `S` to finite-difference
  precision (the force code is unchanged: the shift changes the energy, not its gradient).
- **R2 — the budget pays for entropy growth; decay repays it.** With `pac_kappa` set, `SECUpdate`
  clips this tick's growth of particle i to what its budget can pay,
  `ΔS_i ← min(ΔS_i, P_i / (∂E_SEC/∂S_i))`, debits `P_i −= (∂E_SEC/∂S_i) ΔS_i` on growth and credits
  `P_i += (∂E_SEC/∂S_i) |ΔS_i|` on decay. The transfer is exact: per tick,
  `Σ_i (∂E_SEC/∂S_i) ΔS_i + ΔΣP = 0` to 1e-6 of `ΣP(0)`. `∂E_SEC/∂S_i` is evaluated at this tick's
  positions, before the kick (Milestone R's order: update, then forces).
- **R3 — the total is conserved.** `PACLedger` reports `sec_energy_int`, `budget_int`,
  `budget_frac = ΣP/ΣP(0)`, `total_pac = kinetic_int + potential_int + sec_energy_int + budget_int`,
  `closure_pac = |Δtotal_pac| / |potential_int|` per tick, and `transfer_residual` (R2's identity).
  `total_pac` is conserved to the integrator's truncation: bounded per tick by the Courant
  bound on the gross work, and halving the step over a smooth window halves the drift.
- **R4 — the budget is a ratio, never a coordinate.** `P_i(0) = pac_kappa · |U_grav(0)| · m_i / Σm`
  — a declared fraction κ of the initial binding energy, set once at `ParticleEngine._init` from
  the same gravity kernel and potential table the force uses. `budget0` is recorded on the engine.
  Consequences that follow from arithmetic and are *predicted*, not fitted: the engine can inject
  at most `κ|U₀|`; net creation `Σ (∂E/∂S) ΔS` over a run is ≤ `ΣP(0)` exactly; `κ = 0` is
  gravity-only (entropy cannot grow) and `κ = ∞` (None) is today's substrate.
- **R5 — inert by default.** With `pac_kappa = None` every operator is bit-identical to the
  derived-sink round's code: `budget` stays `None`, no ledger metrics are written, and
  `CANONICAL_SINK` with sinks off remains bit-identical to `CANONICAL`.
- **R6 — a severed particle's budget is frozen with its entropy.** It is excluded from
  `budget_int`, from the pair energy (through `pair_alive`), and from the transfer; nothing
  writes to it after severance.
- **R7 — reported, every tick.** `budget_bound_frac` (fraction of would-grow particles clipped
  this tick), `sec_transfer` (this tick's signed `Σ (∂E/∂S) ΔS`), and its cumulative sum. The engine's
  `bounds` gains `budget_bound_frac_max` and `budget_exhausted_tick` (first tick with
  `budget_frac < 0.01`). A bound that binds is visible on the first run.
- **R8 — no new physics constant, no tuned rate.** κ is swept and declared; `sec_balance`,
  `memory_decay`, `growth 0.1`, `g`, `r0` are untouched. The fracton functional's `β∇²A` and
  `γT·A` terms are declared follow-ons, not included.
- **R9 — tests fail first** (`xfail(strict=True)` until the implementing commit), CPU, fast.

## Acceptance (known answers; POC-12 exp_01 and `tests/v4/test_pac_ledger.py`)

KA-0 force = −∇ of the shifted pair energy, finite-difference on a small state, ≤ 1e-3 relative ·
KA-i transfer identity `Σ (∂E/∂S) ΔS + ΔΣP = 0` to 1e-6 of `ΣP(0)` on every tick of a proxy run ·
KA-ii net creation ≤ `ΣP(0)` exactly over a full proxy run at κ = 1, with the budget binding
(`budget_bound_frac_max > 0`), and pressure work ≤ `ΣP(0)` within the integrator's truncation
allowance (10%, reported) · KA-iii `total_pac` conserved to truncation: per-tick drift bounded by
the Courant bound on gross work; halving the step over the declared smooth window reduces the
drift ≥ 1.5× · anchors: `pac_kappa = None` structurally inert (no budget, no ledger keys) and
`CANONICAL_SINK ≡ CANONICAL` unchanged; κ = 0 bit-identical to `sec_balance = 0`; the None path
reproduces exp_28's recorded B0 seed-1 marks to 1e-6 on the same platform (exp_01 only) ·
severance: a severed particle's budget unchanged over 50 further ticks.

## Falsification of the instrument

`transfer_residual > 1e-6` on any tick, or `closure_pac` exceeding the Courant bound with the guard
unbound, means the ledger is wrong, not the physics, and no exp_29 number stands until it is fixed.

## What this spec does not claim

Whether a ledgered substrate holds structure, or at which κ. That is Milestone R exp_29, registered
before running with the unbounded engine (κ = ∞) and the engine removed (κ = 0) as controls.
