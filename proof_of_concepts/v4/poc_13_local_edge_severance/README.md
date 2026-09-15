# POC-13 (v4): severance where the local ledger crosses the edge

**Status:** the harness for dawn-field-theory Milestone R **exp_33** (R2). Nothing here is scored;
scoring lives in `dawn-field-theory/experiments/sidecars/milestone-r/scripts/exp_33_r2.py` against a
sealed registration. Companion to POC-12 (the ledger) and the edge instrumentation of
`feat/v4-ledger-virial`.

## The trigger

`particles.LedgerSeverance` with `sev_mode = "local_edge"`: a retained particle severs the tick its
cumulative pressure work `work_p_i` first exceeds zero at or after a declared arming time
`sev_t_arm`. No threshold — the zero is the ledger's. It is the local form of the edge that exp_30–32
measured globally (the sign change of the pressure's net work between κ = 1 and 1.25), and the set
it selects below the global edge is, on every seed looked at, the unbound outskirts: a third of the
mean binding, a fifth of the mean local density. The FORM is exp_28's, unchanged: decoupling, not
destruction; the severed particle leaves with its mass, kinetic energy, momentum and its whole
interaction energy with the retained set; no amount. Tests: `tests/v4/test_local_edge_severance.py`.

## Arms (`scripts/exp_01_arms.py`)

B no severance · S local-edge severance · R count-matched random severance (S's event log replayed
by sim_time through the same bookkeeping — the selection control) · D energy-matched drag (damping
chosen so the retained kinetic energy per particle at t_end matches S's, POC-11's rule with one
secant refinement — the dissipation control). All on the ledgered substrate (`pac_kappa = κ`,
damping 1.0 except D), CANONICAL_SINK, differing by config only. `scripts/exp_02_aggregate.py`
builds the grid with per-run summaries.

## What the smoke test showed (proxy, seed 1, κ = 0.5, t_arm = 8 — exploring; the proxy is the
retired regime for pressure-range questions)

The trigger fires a burst at arming (the 12 % with positive work) and then keeps firing as
particles cross zero, to 43 % severed by t = 15; the severed set leaves **bound on aggregate**
(kinetic out 1.7 × 10⁴ against interaction energy out −7.8 × 10⁴) — less bound than the mean is
not unbound; and the retained set ends **super-virial** (2K/(|V_g| − V_p) = 1.26 against 1.01 for B),
because what leaves carries more of the interaction energy than of the kinetic energy. Two of the
drafted R2 tests would fail as written; the full-size design pass below confirmed it.

## What the full-size design pass showed (n = 4000, seed 1, κ = 0.5, t_arm = 8; `results/design_full/`, exploring)

| arm | severed | energy out (per severed particle) | unbound fraction of the severed | retained KE/N | retained 2K/(|V_g| − V_p) | conn_q10 |
|---|---|---|---|---|---|---|
| B | 0 | — | — | 247 | 1.01 | 0.65 |
| S local edge | 49 % (14 % at arming, then a cascade) | −4.0 × 10⁵ (−205; u −273, ke +69) | 0.06 | 113 | 1.34 | 0.62 |
| R random, count-matched | 49 % | −5.3 × 10⁵ (−271; u −402, ke +131) | 0.33 | 73 | 1.49 | 0.65 |

Three findings, each against the drafted R2. **The trigger cascades**: once the arming burst leaves,
the next shell's cumulative pressure work turns positive, and by t = 15 half the set has gone.
**What leaves is bound**: the local-edge set is less bound and slower than a random set of the same
count — the outskirts, as the edge work said — but 94 % of it is bound in the retained frame;
"less bound than the mean" is not "unbound". **The remnant is amputated, not healed**: the retained
set is colder per particle and super-virial; the local-edge selection leaves it closer to balance
than random removal does (1.34 vs 1.49) but far from the unsevered substrate (1.01). Whole-particle
severance of the local-edge set is not radiation on this substrate. The registration is not sealed
on this trigger; the decision (register as a negative, redraw, or change the form so that what is
severed is the pressure's positive work rather than the particle) is Peter's.

## What this POC does not claim

That severance on this trigger is radiation; that the retained set is "held"; anything about the
emitted quantum's carriers. Those are exp_33's questions and its kill sentences.
