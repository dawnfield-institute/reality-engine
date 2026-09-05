# v4 Particle Substrate — energy ledger, ledger severance, Landauer erasure

**Status:** in progress (2026-09-05) · **Scope:** the instruments and operators for POC-11 / Milestone R
exp_28, "does a derived dissipation channel let the substrate hold structure?" The physics claim is
registered and scored in dawn-field-theory; this spec covers what must be true of the code for that
claim to be testable. Builds on `v4-particle-substrate.spec.md` (integrator hygiene).

## Overview

With its speed cap gone the substrate is a relaxation oscillator (`.spec/challenges.md` C4.1), and
exp_04 showed the tuned drag `damping = 0.99` to be the equation of motion for 80% of a run. Milestone
R licenses exactly two derived pieces of a sink: a **trigger** (a vertex severs when all its bonds are
simultaneously overstressed — exp_15/16) and a **form** (severance is decoupling, not destruction;
the severed part leaves with its own value and never interacts again — exp_01). It licenses no amount,
so severance here is **whole-particle decoupling**. The engine also holds a derived erasure rule that
was never closed dynamically (Landauer, `fracton.constants.LN2`). Neither can be measured without an
energy budget the substrate did not have.

## Requirements

- **R1 — the energy ledger closes.** `PACLedger` reports, every tick: `kinetic_int`, `potential_int`,
  `total_int = kinetic_int + potential_int`, `e_int = total_int / n_alive`, `n_alive`, and the
  interacting / severed / global split of mass, kinetic energy and momentum; `entropy_total` over the
  interacting set. The Integrator reports the per-tick partition `work_gravity`, `work_pressure`,
  `loss_drag`, `loss_guard`, `impulse_pressure_{xyz}`, with float64 cumulative sums. The residual
  `|ΔKE_int − (work_gravity + work_pressure − loss_drag − loss_guard − loss_landauer − loss_severance_ke)| / max(KE_int, 1)`
  is reported as `closure_residual` and is ≤ 1e-5 on every tick.
- **R2 — the potential is the force's.** The gravity pair potential is tabulated from the same kernel
  the force uses (`U(r) = −∫_r^{3r0} g e^{−r'/r0}/(r'+0.1) dr'`, zero beyond the cutoff), validated
  against the closed form in `scipy.special.exp1` to 1e-6 relative. No second gravity is introduced.
- **R3 — the work partition is exact.** `work_X = Σ m v0·a_X dt + ½ Σ m a_X·a_tot dt²` for each force
  X, which sums to ΔKE of the kick identically because Σ_X a_X = a_tot.
- **R4 — severance is whole-particle decoupling on the stress trigger.** A particle severs when
  `min over neighbours within sev_radius of |S_i − S_j| > sev_tau` with at least one neighbour
  (Milestone R exp_15's rule, per-edge gradient, all edges). `sev_radius` defaults to the lattice
  spacing `box / ceil(n^(1/dims))` — the degree regime in which the derived barrier is live (design
  pass: within r0 the degree is ~80 and only outliers fire). `sev_tau` is a declared free scale
  parameter, swept in the registration; `sev_tau = None` makes the operator inert.
- **R5 — a severed particle interacts with nothing.** Excluded as source and target from gravity,
  pressure and the SEC update; its entropy is frozen; it is not damped, not guarded, and not counted
  in the Integrator's step statistics; it drifts. Its mass, kinetic energy and momentum leave the
  interacting ledger at the tick of severance and are reported (`loss_severance_ke/energy/mass`);
  `mass_int + mass_sev == mass_total` exactly and a severed particle's velocity is bit-identical
  thereafter. `u_out`, the particle's whole interaction energy with the retained set, is charged to
  `total_int` at that tick.
- **R6 — a random mode exists for the selection control.** `sev_mode = "random"` severs a scheduled
  count of uniformly random alive particles per tick (schedule replayed by `sim_time`), through the
  same bookkeeping as the stress mode.
- **R7 — Landauer erasure is a dynamical step.** When a particle's entropy is released (ΔS < 0), it
  loses `min(KE_i, LN2·|ΔS_i|)` of kinetic energy, `LN2` imported named from fracton, `k_B T = 1` in
  substrate units. dt-invariant by construction (ΔS is rate-scaled). `landauer = False` makes it inert.
  Its magnitude inherits `memory_decay` — the one tuned rate this round declares and does not touch.
- **R8 — inert by default.** With `sev_tau = None` and `landauer = False`, `CANONICAL_SINK` is
  bit-identical to `CANONICAL` on positions and velocities; the existing anchors still pass.
- **R9 — no new physics constant, no tuned rate in a derived arm.** `PHI`, `LN2`, `XI_ANALYTIC` are
  imported named; `sev_tau` is swept, `sev_radius` is declared, `damping` is 1.0 in derived arms and
  appears only as the matched-energy control.
- **R10 — tests fail first** (`xfail(strict=True)` until the implementing commit), CPU, fast.

## Acceptance (known answers; POC-11 exp_01 and `tests/v4`)

KA-0 closure residual ≤ 1e-5 every tick · KA-i energy conserved at damping 1 / sec 0 to the Courant
truncation bound `|ΔE| ≤ cfl_max · Σ|work|`, and halving `cfl` reduces |ΔE| by at least ×1.5 (measured ×7.6) · KA-ii at
damping 0.99, sec 0: `Σ loss_drag = KE_0 − KE_T` to 1e-6 and the fitted decay rate equals
`2 ln(0.99)/dt_ref = −0.4020` within 1% · KA-iii `loss_guard = Σ ½ m (v² − cap²)` over clamped
particles, exactly · KA-iv momentum closure `Δp_int = Σ impulse_pressure − p_out` to 1e-5 (with the
third-law pressure, `impulse_pressure ≡ 0`) · severance: the fired set equals an independent
recomputation of the rule; no retained particle with a neighbour satisfies it; degree-0 never fires;
`mass_int + mass_sev` exact; severed velocity bit-identical over 100 ticks; random mode reproduces its
schedule; `ΔE_int` at a severance tick = −(ke_out + u_out) · Landauer: per-particle loss equals
`min(KE, LN2|ΔS|)` on a constructed releasing state, zero on growth, total erased equal at dt and
dt/2 within 1e-4 · anchors: `CANONICAL_SINK` (sinks off) ≡ `CANONICAL`; free drift unchanged.

**Falsification of this spec:** a closure residual above 1e-5 with the guard unbound, or any drift in
`kinetic_sev`, means the instrument is wrong — not the physics.

## What this spec does not claim

Whether the sink lets the substrate hold structure. That is exp_28's question, scored against its
sealed registration in dawn-field-theory with matched-energy and matched-count controls, and a null is
a bearing. The pressure pair law was replaced the same day (`.changelog/20260905_180000_v4_pressure_pair_law.md`);
every pre-seal number is computed on the corrected force.
