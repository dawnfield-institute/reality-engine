# POC-11 (v4): A derived sink, not a tuned one

**Status**: completed 2026-09-05 · **Pillar**: SEC / PAC · **Registered in**: dawn-field-theory Milestone R exp_28 (sealed `bf833113`) · **Result**: 0/4 — the trigger is a core detector

## Result

The derived sink does not let the substrate hold structure, and the way it fails is the finding.
`min_j |S_i − S_j| > τ` fires only at a local extremum of the entropy field; exp_15 ran it on noise,
whose extrema are random, while here the SEC entropy grows with density, so the extrema are the
collapse cores. Severance removed the bound, connected part (`u_out < 0` in every run, fired/retained
KE ratio ≈ 0.5 at first firing) and left the rest hotter per particle (T1 fail, the registered live
direction) and *less* connected than a random subset of the baseline at the same count (T2 fail with
sign: 3/3 seeds, margin −0.023, pooled σ 0.009 at the proxy's informative τ = 20; a null at n = 4000,
τ\* = 10, where half leaves and S = R). Random removal at the onset τ held more; the tuned drag, calibrated to the same retained KE per particle, held as much or
more. Landauer erasure is ~1% of the pressure work. Every closure residual in the grid ≤ 6 × 10⁻⁷.
At n = 4000 (τ\* = 10, three seeds, 48–54% severed) the picture repeats: S 0.051 / 0.046 / 0.051
against R 0.055 / 0.046 / 0.052, with B0 dissolved to 0.028–0.038 on its own set.
C4.1 option 3 closes; option 2 (derive the pressure term from the SEC functional) remains. Journal:
`journals/2026-09-05_the-trigger-selects-the-cores.md`; scoring and outcomes in dawn-field-theory
`milestone-r/journals/2026-09-05_exp28_outcomes.md`.

## Why

With its speed cap gone (POC-07, 2026-09-05) the particle substrate has no energy sink, and it
is a relaxation oscillator: collapse, entropy spike, detonation, dispersal, re-collapse. exp_04
then showed that for 80% of a run the tuned drag `damping = 0.99` *was* the equation of motion
(kinetic decay = 2 ln 0.99 / 0.05 to four figures). A cap and a drag are both ways of turning
the physics off. The question is whether the corpus's own physics supplies the sink.

Milestone R says radiation is **ledger severance**. It licenses exactly two derived pieces: a
**trigger** — a vertex severs when *all* its bonds are simultaneously overstressed (exp_15,
exp_16) — and a **form** — severance is decoupling, not destruction; the severed part leaves
with its own value and never interacts again (exp_01). It licenses no amount. So here a
particle that fires **leaves the interacting ledger whole**: no fraction, no energy law, and
global conservation to machine precision as the gate.

Two things had to exist first. The pressure pair law was self-propulsion (every pair injected
net momentum; POC-07 journal `2026-09-05_the-pressure-was-self-propulsion.md`) and was replaced
by a third-law pressure. And the substrate had no energy accounting at all — no potential
energy, no work partition — so "the sink removed energy" could not be told from "the force
stopped injecting it." Both are `.spec/v4-derived-sink.spec.md`.

## What is built here

| piece | what it is | derived / declared / inherited |
|---|---|---|
| energy ledger | potential from the force's own kernel; exact per-force work partition; `closure_residual` every tick | instrument, gated (KA-0..iv) |
| `LedgerSeverance` | whole-particle decoupling when `min over neighbours of \|S_i − S_j\| > τ` | trigger derived (M-R exp_15/16), form derived (exp_01); τ **swept**; radius **declared** (lattice spacing) |
| `LandauerErasure` | released entropy costs `LN2·\|ΔS\|` of kinetic energy | form derived; magnitude **inherits** `memory_decay` — exploratory arm |
| random mode | same bookkeeping, random selection | the selection control |
| matched drag | damping calibrated so the retained kinetic energy per particle at t_end matches the severance arm's (positive-definite; one pre-declared refinement) | the derivation control |

## Scripts

- `exp_01_ledger_gates.py` — the known-answer gates. Must PASS before anything else runs.
- `exp_02_trigger_calibration_proxy.py` — where and how often the trigger fires on the proxy,
  **with** removal, across τ. Computes no structure metric by construction: calibration, not tuning.
- `exp_03_sink_arms.py` — one run per invocation (`--arm B0|S|D|R|L|SL --tau --seed --size`),
  one JSON per run. Recorded at every unit of simulated time.
- `exp_04_aggregate.py` — the grid JSON with per-run hashes and the commit. Aggregates only;
  **scoring lives in dawn-field-theory** (`milestone-r/scripts/exp_28_dynamical_severance.py`).
- `run_grid.sh proxy | proxy_d <τ*> | full <τ*>` — the grid, phase by phase, into `results/proxy/`
  and `results/full/`. τ\* comes from the dawn-field-theory scorer's proxy pass (pre-declared rule).

## What this POC does not claim

Whether the sink lets the substrate hold structure. That is registered, sealed and scored in
Milestone R exp_28 with two matched controls and two kill sentences, and a null is a bearing.
