# Reality Engine -- Claude Code Context

## Identity
Reality Engine is a computational physics framework where fundamental physics emerges from information dynamics. Three fields (E=energy, I=information, M=mass) plus local rules generate quantum mechanics, gravity, thermodynamics, and relativity without programming them. v3 architecture (2026-03-14), pre-alpha research software based on Dawn Field Theory principles.

## Architecture

```
reality-engine/
├── src/v3/                    # ACTIVE — v3 composable operator pipeline
│   ├── engine/                # Engine, FieldState, EventBus, Config
│   │   ├── engine.py          # Main simulation loop
│   │   ├── state.py           # Immutable FieldState (E, I, M, T, metrics)
│   │   ├── event_bus.py       # Pub/sub event system
│   │   └── config.py          # SimulationConfig (nu, nv, dt, field_scale, ...)
│   ├── operators/ (18)        # Physics operators (see below)
│   ├── analyzers/ (6)         # Conservation, Gravity, Atom, Star, Quantum, Galaxy
│   ├── emergence/ (3)         # Structure, Particle, Herniation detectors
│   ├── substrate/             # MobiusManifold, Constants, Projections
│   └── dashboard/             # FastAPI + WebSocket + Plotly.js
├── scripts/                   # Diagnostics and validation
│   ├── physics_scorecard.py   # 3-tier physics validation (13 metrics)
│   ├── diagnose_gravity.py    # Gravity operator diagnostics
│   └── ...                    # Various analysis scripts
├── spikes/                    # Research experiments by topic
│   ├── theory_integration/    # DFT theory vs simulator (13 spikes, 10/10 pass)
│   ├── coupling_drift/        # Gravity/memory optimization (6 spikes)
│   ├── atomic_emergence/      # Atom classification
│   ├── big_bang/              # Big bang evolution
│   └── ...
├── tests/v3/                  # 143 tests (pytest) — the only suite that runs
├── docs/                      # Theory and guides
└── archive/                   # earlier generations, preserved not deleted
    ├── v1/                    # Jan 2026 layer packages: core/, dynamics/,
    │                          #   conservation/, substrate/, scales/, emergence/,
    │                          #   cosmology/, memory/, analyzers/ + their consumers
    │                          #   (proof_of_concepts/, dashboard/, examples/, tools/,
    │                          #   and the v1-importing spikes/scripts/tests)
    └── v2/                    # Feb 2026, Reality Engine 2.0.0a1 (was src/) + tests/v2
```

**Three generations existed; v3 is the only live one.** v1 was what
`.spec/architecture.spec.md` originally documented — its SUBSTRATE/CONSERVATION/DYNAMICS/
SCALES/EMERGENCE layers were the top-level directory list. v3 imports nothing from either
archived generation (verified: 0 references).

`src/` is kept as the import root purely because v3 addresses itself absolutely — 31
modules and 72 test references use `src.v3.*`. Promoting `v3` to the top level would mean
rewriting all of them for no behavioural gain.

## v3 Operator Pipeline (16 operators)

**This is the canonical pipeline** — used by `src/v3/__main__.py`, `scripts/physics_scorecard.py`,
and `spikes/theory_integration/harness.py`. Declared in `src/v3/engine/pipelines.py` as
`CANONICAL`; use `build_canonical_pipeline()`.

Default order:
1. **RBF** — Recursive Balance Field: dE/dt from laplacian, coupling terms
2. **QBE** — Quantum Balance: dI/dt = -dE/dt (PAC conservation)
3. **Actualization** — MAR-gated integration, ln(phi) split, pi/2 harmonic modulation. Replaces EulerIntegrator. Populates `state.P`, the unactualized potential buffer.
4. **Memory** — Mass generation (bulk + gradient boundary seeding), de-actualization (PAC cycle completion), quantum pressure, diffusion
5. **PhiCascade** — Fibonacci two-step memory for phi-spaced mass levels
6. **Gravity** — Self-gravity via spectral Poisson solver, xi_mod, cascade-depth tiling filter + pi-harmonic modulation
7. **SpinStatistics** — Emergent Pauli exclusion from information cost
8. **ChargeDynamics** — EM-like forces from charge field Q
9. **Fusion** — Nuclear fusion in dense, hot, gravity-compressed regions
10. **Confluence** — Mobius antiperiodic projection f(u+pi,1-v) = -f(u,v)
11. **Temperature** — Local T from |E-I| gradients
12. **ThermalNoise** — Langevin noise
13. **Normalization** — Soft-clamp E/I, cap M, Landauer reinjection
14. **SECTracking** — Read-only SEC energy functional, entropy, info fraction, cascade depth
15. **Adaptive** — Self-tuning damping and dt from energy growth
16. **TimeEmergence** — dt = dt_base / (1 + kappa*max|E-I|)

### Pipeline proliferation — know which one you are running

**25 sites outside `archive/` build their own `Pipeline([...])`, from 8 to 16 operators**
(7 use 16, 8 use 15, 5 use 14, the rest fewer). Results are not comparable across them:
two spikes can disagree because they ran different physics, and nothing says so.

The notable divergence is the **dashboard**, whose pipeline had the misleading name
`build_default_pipeline` and runs **12** operators — omitting Actualization, PhiCascade,
SpinStatistics, SECTracking and ChargeDynamics. Under it `state.P` is **all zeros**, so
the engine has no Delta term. It is now `build_dashboard_pipeline()`, and
`tests/v3/test_pipeline_completeness.py` fails if the difference from canonical changes
without being declared.

Anything measured with the dashboard pipeline is measuring reduced physics. The scorecard
and theory_integration spikes are **not** affected — they use the canonical 16.

## Physics Scorecard

`python scripts/physics_scorecard.py` — 3-tier validation (13 metrics):
- **Tier 1** (coupling attractors): f->gamma_EM, gamma->1/phi, alpha->ln(2), G->1/phi^2, lambda->1-ln(2)
- **Tier 2** (structural): phi^2 mass spacing, PAC conservation, spin 1/2, entropy reduction
- **Tier 3** (aspirational): fine structure 1/137, Koide Q=2/3, mu/e ratio
- **Current score**: 7/13 passing (GPA C), as of 2026-03-17 (theory integration implementation)
- **NOW tick detection**: finds minimum Tier 1 error epoch, reports beta functions and cosmic epoch map

## Key Physics

- **Gravity**: Spectral Poisson solver with cascade-depth tiling filter (DFT exp_36) + pi-harmonic modulation (spike 02). Entropy-coherence modulation xi_mod. Amplitude coupling nabla^2 Phi = sqrt(M).
- **Mass generation**: Bulk (gamma_local * diseq^2) + boundary gradient seeding (gamma_local * |grad(diseq)|^2 / (1+M))
- **De-actualization**: dM_deact = -eta * M * (1 - gamma_local), eta=0.025 (spike 04 optimal). Memory fades where disequilibrium resolves, completing PAC cycle.
- **PAC conservation**: `E + I + M` held to <1e-12 by an explicit correction in
  `NormalizationOperator` that fires on ~99.8% of ticks — it is ENFORCED, not
  observed. Set `enforce_pac=False` to measure the unenforced dynamics. Note the
  full ledger is `E + I + M + P`: `P` is the unactualized potential buffer (the
  Delta term), inert under the default pipeline because Actualization is excluded.
- **SEC metrics**: info_fraction = |I|/(|E|+|I|) (best duty cycle proxy, r=+0.954), log-time cascade depth with running NOW estimate
- **Initialization**: `big_bang` (symmetric E~I), `entropy_dominated` (E>>I, DFT-correct), `info_dominated` (I>>E, fast convergence)
- **DFT constants**: Xi = gamma_EM + ln(phi) = 1.05843, ln^2(2) = 0.4805, phi = golden ratio

## Conventions

- Physics must EMERGE, never be programmed — no hardcoded F=ma, E=mc^2, etc.
- PAC conservation enforced at machine precision (< 1e-12)
- Mobius manifold substrate with anti-periodic boundaries: f(x+pi) = -f(x)
- Tests: `pytest` from repo root — `pytest.ini` targets `tests/v3` (143 tests)
- Installation: `pip install -r requirements.txt`
- Run the engine: `python -m src.v3 --help`
- Scorecard: `python scripts/physics_scorecard.py`
- Conventions are canonical in [`dawn-field-theory/STANDARDS.md`](../dawn-field-theory/STANDARDS.md);
  this file is repo-specific context, not a second standard.

> The old quick demo `examples/field_visualizer.py` is in `archive/v1/` — it imports
> `core.reality_engine` and visualises v1's P/A fields, not v3's E/I/M.

## Related Repos

- `fracton` — Infodynamics SDK (provides PAC/Mobius primitives). **Imported behind
  try/except fallbacks.** With fracton absent the suite is 138/138; with it present, 15 v3
  tests fail — see Known Gaps.
- `dawn-field-theory` — theoretical foundation (exp_28, exp_29, exp_36 feed gravity)
- `dawn-models` — AI architectures using same DFT principles
- `lore` — the knowledge graph (CT106). **kronos is retired — never write through
  `kronos_*` tools.** Search with `lore_search`; sync after structural changes.

## Theory Integration (2026-03-17)

13 spikes in `spikes/theory_integration/` bridging DFT theory corpus to simulator. Final synthesis: **10/10 quantitative predictions confirmed**.

Key findings:
- **PAC (global) vs SEC (local)**: PAC conserves at 2.06e-14; SEC drives local dynamics. Coupling constants are averages of turbulent local landscape — trajectory shapes universal, absolute values not.
- **Init independence**: Late-time trajectories correlate >0.997 regardless of init. Info-dominated converges fastest (tick 2300), entropy slowest (13950).
- **Emergent RG flow**: Coupling drift = renormalization group flow, not a bug. Beta functions init-independent. Gravity running 1.3x at high-z (JWST match).
- **SEC duty cycle**: Log-time mapping + info fraction proxy gives r=+0.954 correlation with theory.
- **"YOU ARE HERE" tick**: 5400/10000 = 54% lifecycle, 4.75% avg error (post-implementation)

Implemented into engine: info fraction metric, eta=0.025, entropy/info init factories, log-time cascade depth, pi-harmonic tiling filter, scorecard NOW tick + beta functions + epoch map.

## Current State

- v3 architecture, 18 operators, 138 tests
- Physics scorecard: 7/13 (C) — theory integration implemented, NOW tick at 5400 (4.75%)
- Theory integration: 10/10 DFT predictions confirmed (spikes 09-13), 6 findings implemented
- Initialization: 3 modes (big_bang, entropy_dominated, info_dominated)
- 6 analyzers operational, PAC conservation validated at machine precision
- Not accepting code contributions yet

## Known Gaps

- ~~fracton integration is incomplete~~ — **fixed 2026-08-11.** Both suites are now
  138/138, with and without fracton, and CI gates both. What it was: the PAC audit in
  `src/v3/operators/normalization.py` called `validate(E+I+M, [E, I, M])` — residual
  `|x − (a+b+c)|` with `x = a+b+c`, tautologically zero, so it could never fail — through
  `validate()`, typed for `torch.Tensor`, which raised `TypeError` whenever fracton was
  actually installed. It now calls `validate_tree` (the scalar entry point) against the
  **pre-correction** sums, which measures how far PAC drifted before the correction pulled
  it back. Auditing the post-correction state would have been tautological too, one step
  removed. Verified it can fail: injecting 5.0 of drift yields residual 5.0 and one
  counted violation, where normal operation yields exactly 0.
- **`Ξ` has three legitimate values.** fracton defines `XI_ANALYTIC` (γ+ln φ = 1.05843),
  `XI_DISCRETE` (1+π/55 = 1.05712) and `XI_PAC`; the 0.12% spread is structural (γ/48,
  exp_26). Use the named constant — never a bare literal — and say which you mean.
- **`.spec/` is behind the code.** `architecture.spec.md` documents v1;
  `modernization-roadmap.spec.md` predates M6–M15 and sources from Era-1/2 experiments.

## Guardrails

- Do NOT hardcode physics laws — all physics must emerge from field dynamics
- Do NOT break PAC conservation invariants (< 1e-12 error)
- Do NOT modify substrate geometry without understanding Mobius topology
- Always run `pytest` after changes (138 tests must stay green)
- spikes/ are research experiments — treat as exploratory, not production code
- `archive/v1/` and `archive/v2/` are lineage — **read, never modify, never import from**.
  Archived work keeps its original shape; that shape is evidence of when it was done.
- Never add a bare physical constant — import it named from fracton
