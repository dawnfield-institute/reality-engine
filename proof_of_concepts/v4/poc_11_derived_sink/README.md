# POC-11 (v4): A derived sink, not a tuned one

**Status**: active · **Pillar**: SEC / PAC · **Registered in**: dawn-field-theory Milestone R exp_28

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
| matched drag | damping calibrated to the severance arm's measured loss | the derivation control |

## Scripts

- `exp_01_ledger_gates.py` — the known-answer gates. Must PASS before anything else runs.
- `exp_02_trigger_calibration_proxy.py` — where and how often the trigger fires on the proxy,
  **with** removal, across τ. Computes no structure metric by construction: calibration, not tuning.
- `exp_03_sink_arms.py` — one run per invocation (`--arm B0|S|D|R|L|SL --tau --seed --size`),
  one JSON per run. Recorded at every unit of simulated time.
- `exp_04_aggregate.py` — the grid JSON with per-run hashes and the commit. Aggregates only;
  **scoring lives in dawn-field-theory** (`milestone-r/scripts/exp_28_dynamical_severance.py`).

## What this POC does not claim

Whether the sink lets the substrate hold structure. That is registered, sealed and scored in
Milestone R exp_28 with two matched controls and two kill sentences, and a null is a bearing.
