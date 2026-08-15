# POC-07 (v4): A particle substrate

**Status**: completed · **Pillar**: SEC

## Why

POC-05 established that the field engine cannot produce connectivity. Its overdense set breaks
into ~1300 pieces where a web is one, and that number does not move under any intervention —
loss, clamping, quantum pressure, gradient transport, curl transport, more room, more time.
Roughly forty runs, one answer.

POC-06 then asked what laws it has and found one, `E+I+M`, held at machine precision by an
operator that corrects it every tick. No persistent objects, so nothing for a force law to hold
between.

`gravity_from_maxwell_pac/exp_09` already produces a cosmic web from 5000 particles with
finite-range gravity and SEC entropy pressure. This is the engineering move: **bring the working
mechanism into the good architecture** — operator protocol, immutable state, read-only ledger —
and measure it with the same instruments, on a density field binned at the same resolution, so
the numbers are directly comparable.

The substrate change is about **identity**. A cell has a value; a particle has a trajectory.
Persistence, objecthood and connectivity are properties of things that endure, and a field of
independent values has none of them.

## Result — it works, and the structure is transient

| | percolation | void | cv | filament |
|---|---|---|---|---|
| white noise | 0.003 | 0.05 | 0.75 | — |
| **field engine, best of ~40 runs** | **0.012** | 0.65 | 1.47 | — |
| particles t=100 | 0.012 | 0.875 | 3.76 | 0.125 |
| particles t=300 | 0.069 | 0.925 | 5.66 | 0.075 |
| **particles t≈600 (peak)** | **0.188** | 0.910 | 5.46 | 0.090 |
| particles t=2000 | 0.080 | 0.873 | 3.56 | 0.127 |
| particles t=4000 | 0.050 | 0.870 | 3.53 | **0.130** |
| *exp_09 reference* | — | 0.50 | ~2.0 | *0.12* |

**Peak percolation is 15× the field engine and 60× white noise**, with curved filamentary arcs,
connected chains and large voids visible in `results/web_formation.png`. Every field-engine
render at every setting was uniform speckle. This is different in kind, not degree.

**Then it declines.** Clumps virialize and the connecting bridges drain into them. The web is a
transient of the mildly-nonlinear regime: without cosmic expansion holding it open, gravity
eventually wins and the structure fragments into isolated bound halos. exp_09 ran 600 steps —
almost exactly where the peak sits.

Filament fraction converges to **0.130** against exp_09's measured 0.12, and CV falls to 3.53
against exp_09's ~2.0, so the late state is in the right neighbourhood on both.

**It never crosses the 0.25 percolation gate**, so `is_web` is False throughout. The honest
statement is *web-like structure with a peak epoch*, not a cosmic web.

## Conservation, on a ledger that corrects nothing

`PACLedger` is read-only. Whatever holds, holds.

- `mass_total` — **EMERGENT**, CV 0
- momentum — verified to conserve **exactly** (0.00e+00) in a two-body test with damping off.
  Under the canonical pipeline it does not, because the velocity damping and speed cap are
  dissipative by construction. Design, not defect.

Contrast with the field engine, where the single conserved quantity is held at 2.5e-15 by an
explicit correction applied every tick.

## Two failures worth keeping

**A sign error the physics caught, not a test.** The first run had gravity *repulsive* —
`d[i,j] = pos_i − pos_j` points away from `j`, so an attractive force needs `−unit`. The cloud
expanded to uniform, damping killed the motion, and every metric froze identically from t=100 to
t=600 (void 0.756, cv 1.772) at percolation 0.001, *below white noise*. Three tells at once.
exp_09 carries the same sign with a comment claiming it points toward the other particle — worth
flagging upstream.

**`fit_force_law` fails here, and that retracts part of POC-06.** R² = **0.0005** on a system
whose force law is known by construction, twice, over 14000 samples. In a dense system each
particle's acceleration sums over many neighbours, so a nearest-neighbour projection measures
mostly the others. A two-body fit needs two bodies.

So POC-06's `force(r): ABSENT` on the field engine was an instrument limit reported as a
finding. The conservation arm is unaffected — and is now calibrated by this POC, which is the
thing it needed: it correctly finds mass conserved here and nothing conserved there.

## Running it

```
python proof_of_concepts/v4/poc_07_particle_substrate/scripts/exp_01_does_it_web.py --steps 600
python proof_of_concepts/v4/structure.py        # instrument calibration must PASS first
python proof_of_concepts/v4/law_detector.py     # detector calibration
```

The substrate itself is `proof_of_concepts/v4/particles.py` — `ParticleState`, `ParticleConfig`,
and five operators (`SECUpdate`, `LocalGravity`, `SECPressure`, `Integrator`, `PACLedger`)
following the same protocol as v3's.

## Open

- **Expansion.** The web's decay is the obvious next lever: a Hubble-like scale factor should
  hold it open and is one term.
- **Percolation never reaches 0.25.** Whether that is a real ceiling or a threshold artifact
  wants the overdensity cut swept.
- **A many-body force fitter** — direct summation against a candidate kernel, or the pair
  correlation function. Not attempted here.
- Whether any of this belongs in `src/v3/` as a second substrate is deliberately not decided.
