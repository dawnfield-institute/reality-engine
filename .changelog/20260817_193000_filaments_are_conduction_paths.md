# Filaments are conduction paths — correcting exp_03's null

## The correction

POC-10's `exp_03` reported that **time does not conduct along the cosmic web** (+0.30σ against a
uniform control), and that went into the merged PR #5 as a settled null. It is wrong, and the
fault was the operator rather than the analysis.

`exp_03` coupled particles by a **distance ball** of radius r0. A ball around a point in a dense
node holds far more particles than one on a filament — measured, **~798 neighbours in the web
against 116 in the uniform control** at the same `n` and `box`. "How much matter is nearby" and
"how it is connected" are therefore entangled *in the definition of the operator*, and no
downstream analysis separates them. It could not have detected conduction either way.

## The fix and the result

A **k-nearest-neighbour graph** gives bounded degree — 7.37 in the web against 7.14 in the
control, CV 0.163 vs 0.170 — so a dense region earns no extra conductance for being dense.
Measuring **effective resistance** from a source (a steady-state linear solve: no threshold, no
step count, no stiffness) against local density within fixed Euclidean shells, topology-only
edges, 5 seeds:

| k | web | uniform control | difference |
|---|---|---|---|
| 4 | +0.410 | −0.003 | +5.21σ |
| 6 | +0.422 | −0.022 | +6.22σ |
| 10 | +0.442 | −0.028 | **+9.14σ** |

**The effect size is k-independent** (0.41–0.44); significance rises only because more edges
means less noise. Controls flat at zero. Stronger with length removed than with conductance
∝ 1/length (+0.297, +2.57σ), so not an edge-length artifact.

## It is a parallel-path effect

The geodesic arm — shortest **physical path length** on the same graph — shows nothing: +0.68σ,
+0.02σ, −1.64σ, scattering around zero. Effective resistance and shortest path diverge only
through **redundancy**. A filament does not conduct because it is a short wire; it conducts
because it is a thick bundle of parallel ones.

## Three more broken observables on the way

- **Hop count** — edge length scales with local spacing on a k-NN graph, so crossing dense
  material costs more hops for purely geometric reasons. Gave −3.59σ, the same wrong answer as
  exp_03 by a different mechanism.
- **Explicit time-stepping** — conductance weights make the network stiff, the stable step
  collapses, and it returned **all-NaN**. "Nothing arrives" reads like a physical statement and
  is a solver failure.
- **Length weighting** — conductance ∝ 1/length hands dense regions high conductance because
  their edges are short. Settled by an unweighted run, which made the effect *stronger*.

## Files

- `poc_10_emergent_local_time/scripts/exp_06_conduction_on_the_graph.py` — new
- `journals/2026-08-17_filaments-are-conduction-paths-and-my-null-was-the-operator.md` — new
- The earlier journal keeps its conduction section under a superseded header; corrections layer
  forward. README and meta.yaml corrected.
- `results/exp_06_conduction_k{04,06,10}_topology_5seeds.{json,png}` plus the length-weighted
  comparison run.

## Note

This is the fifth measurement failure in one day running the same direction — understating
structure — and the first that had already been merged as a result. The rule it earns:
**an operator has to be able to answer the question before its answer means anything.** Ask what
the measurement would return if the hypothesis were true, and check the apparatus can produce
that number at all.
