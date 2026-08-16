# v4 — What the corpus already establishes

**Read this before designing a v4 POC.** POC-01 through POC-03 were designed without it and
tested the wrong model, in the wrong regime, using a non-standard operationalization, while
re-deriving results that already existed. This document exists so that does not recur.

Sources: `dawn-field-theory/experiments/studies/asymmetric_conservation` (17 experiments),
`.../cellular_automata_pac_attractors`, `.../oscillation_attractor_dynamics`,
`reality-engine/spikes/theory_integration` (13 spikes), Lore nodes `asymmetric-conservation`,
`pac-necessity-proof`, `conservation-attractor-frame`, `cellular_automata_pac_attractors`.

---

## 1. The ledger is P + A + Δ = C, not E + I + M

`asymmetric_conservation`, validated 2026-02-08, **5/5 falsification tests**:

| | |
|---|---|
| **Identity** | `P + A + Δ = C` holds always |
| **Δ** | bounded, `Δ ∈ [−bound, +bound]`; a buffer of *unreconciled* actualization |
| **Reconciliation** | events clear Δ → 0, restoring exact conservation |
| **Frame** | "Conservation is **structural** (enforced at reconciliation boundaries), **not procedural** (enforced at every timestep)" |
| **Reading** | "**Asymmetry is a frame effect, not a violation**" |

Δ holds unreconciled actualization **until parent nodes process child events** — the
parent/child hierarchy is part of the model, not a future extension.

Reference implementation exists and is validated:
`experiments/studies/asymmetric_conservation/core/async_pac.py` — `AsyncPACNode` with
`delta`, `C = P + A + delta`, `local_asymmetry()`, `receive_event()`, `reconcile()`.

## 2. Ξ is the reconciliation threshold

| constant | source | role |
|---|---|---|
| φ, 1/φ | **PAC alone** | self-similar collapse ratio, `α/(1−α) = 1/α → α = 1/φ` |
| **Ξ = 1 + π/55** | **SEC + PAC** | **reconciliation threshold** (π continuous, F₁₀ = 55 discrete) |
| λ* = 0.618432 | SEC alone | prime density threshold |

Derived in `oscillation_attractor_dynamics` exp_24: within-depth coupling
`2√(r(1−r)) − 1` at `r = 1/φ` gives net offset π/55. Not curve-fit.

And from `cellular_automata_pac_attractors` (validated, p = 8.58×10⁻⁸):

> **Ξ ≈ 1.057 is NOT a universal constant — it is the maximum sustainable computational
> asymmetry for closed recursive systems under PAC conservation.**

So Ξ is a **ceiling/threshold**, not a value the system returns to. A test that looks for a
fixed point Ξ attracts toward is looking for the wrong shape.

## 3. The corpus tests attractors by path-independence

Not by perturbation-recovery. The signature used repeatedly is **different starts, one
endpoint**:

- `pac-necessity-proof`: φ is the unique attractor of PAC recursion *regardless of initial
  conditions* — from 1.5, 2.0, √2, φ±0.01 → **0.00% error convergence**.
- `theory_integration` spike 12: 7 init variants, **late-time trajectories correlate >0.997**.
- `conservation-attractor-frame` (Ember III r15): "Destination path-independence…
  different paths, one endpoint — **the attractor signature**."

## 4. Regime: the engine has a long transient

`big_bang` is symmetric init. Convergence ticks by init (spike 12):

| init | converges |
|---|---|
| info-dominated | **2300** |
| **symmetric (`big_bang`)** | **8450** |
| entropy-dominated | **13950** |

**Any measurement under ~8450 ticks with `big_bang` is in the transient.** Initial
conditions dominate there; late-time they do not (>0.997 correlation). POC-01→03 all ran
600–1000 ticks.

## 5. Already established — do not re-derive

- **α and λ are one degree of freedom**, r = +1.000 exact, and **PAC conservation forces
  the correlation structure**. Group 1 (γ, α, λ) vs Group 2 (f, G): r = −0.98.
  (spike 07, from M5 exp_11.)
- **Coupling drift is emergent RG flow, not a bug.** Beta functions init-independent.
  Gravity running 1.3× at high-z matches JWST.
- **η ≈ 0.025 optimal** (spike 04). **Bifurcation map** of coupling space (spike 06).
- **Spatial transport amplification**: theory predicts PAC rate asymmetry ~φ; simulator
  shows 6.07× because cells must physically redistribute across the manifold. The excess
  is the SEC contribution on top of PAC.

## 6. Pipeline proliferation — and which one a result was measured with

**Corrected twice; this is the version that survived checking every site.**

First pass: searched for the vocabulary (`delta`, `reconcile`), found nothing, concluded
the engine implements none of the model. Wrong — it exists as `state.P` plus MAR-gated
`ActualizationOperator`.

Second pass: found `build_default_pipeline()` omitted six operators and concluded the
documented physics does not run, "including the scorecard's 7/13". **Also wrong**, and it
made `CLAUDE.md` worse before being reverted.

What is actually true, after enumerating all 25 `Pipeline([...])` sites outside `archive/`:

| operators | sites |
|---|---|
| 16 | 7 — incl. `src/v3/__main__.py`, `physics_scorecard.py`, `theory_integration/harness.py` |
| 15 | 8 |
| 14 | 5 |
| 13 / 12 / 10 / 9 / 8 / 3 | 1–2 each |

- **The canonical pipeline is 16 operators**, and `CLAUDE.md` documented it correctly.
  `__main__.py` names it explicitly: `ActualizationOperator(), # replaces EulerIntegrator`.
- **The scorecard's 7/13 and the theory_integration spikes used the canonical 16.** They
  are not affected by any of this.
- **The dashboard is the outlier at 12**, under the misleading name
  `build_default_pipeline`. Under it `state.P` is all zeros — no Δ term.
- **POC-01→03 imported the dashboard's pipeline**, so they measured reduced physics
  without MAR actualization or the φ-cascade. That is a defect in those POCs, not in the
  engine.

The real problem is that **no site declared which pipeline it was using or why**, so
results were not comparable and nothing detected the divergence. Now declared in
`src/v3/engine/pipelines.py` (`CANONICAL`, `DASHBOARD_REDUCED`, `REDUCED_OMITS`, `UNUSED`)
and gated by `tests/v3/test_pipeline_completeness.py`.

**Measured effect of the operators the reduced pipeline drops** (32×16, 1500 ticks, noise
off, seed 7 — all run stably; none is broken):

| operator | effect |
|---|---|
| `Actualization` | M_total **+8.9%**; makes `state.P` live — \|P\| → ~12.4 over 475/512 cells and **saturates**; including P **halves** apparent ledger drift (0.181 → 0.082 at t=3000) |
| `SpinStatistics` | M_total −6.4% |
| `PhiCascade` | M_total +0.5% |
| `SECTracking` | none (read-only) — its absence is why `info_fraction` was missing in POC-02 |
| `ChargeDynamics` | none measured — no Q field in `FieldState`; likely inert |

The Δ saturation is the substantive physics result here: a **bounded** buffer, which is
what `Δ ∈ [−bound, +bound]` predicts, observed in the engine for the first time.

## 7. Genuinely open questions (these are the targets)

Taken verbatim from the corpus, not invented:

1. **"Is there a maximum Δ magnitude before conservation fails?"** — `asymmetric-conservation`
2. **"Can Δ dynamics predict when systems will undergo phase transitions?"** — same
3. **F1: nonzero conservation floor ~ γ — PENDING** — `conservation-attractor-frame`,
   described there as "the resident's question"

Question 1 is directly reachable once the engine has Δ, and CAH gives a prior worth
testing against: if Ξ is the maximum sustainable asymmetry, the Δ ceiling should relate
to it.

---

## What POC-01→03 got wrong

Recorded so the failure modes are legible, not to be self-flagellating.

| error | correct |
|---|---|
| measured `E+I+M` as the ledger | model is `P + A + Δ = C` |
| tested **procedural** per-tick conservation | conservation is **structural**, at reconciliation boundaries |
| ran 600–1000 ticks | symmetric init converges at **8450** |
| used perturbation-recovery for "attractor" | corpus standard is **path-independence** |
| reported α/λ single-DOF as a finding | known: spike 07 / M5 exp_11, and *explained* |
| reported 36× seed dominance as an engine finding | known transient; late-time is init-independent |
| swept `mass_gen_coeff` | dead config field, read by zero operators |

The one finding that survives: **the engine does not implement Δ or reconciliation.**
That is section 6, and it is the reason to build rather than to measure.
