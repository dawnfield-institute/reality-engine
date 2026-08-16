# POC-09 (v4): Three dimensions

**Status**: active · **Pillar**: SEC

## Why

POC-05 (contrast and connectivity are decoupled) and POC-07 (particles percolate far better
than the field engine) both rest on percolation measured in **2D**, and both engines had only
ever been run there. That is not a neutral choice for a connectivity measurement:

- In 2D, filaments that cross **must** intersect, voids must be fully enclosed, and sheets do
  not exist. exp_11 names the target topology as filaments, **sheets**, nodes and voids.
- The site-percolation threshold is **0.593 in 2D against 0.312 in 3D** — the same field
  percolates far more readily in three dimensions.

exp_31 Part A wants 3D independently: the cascade 1/r profile requires `d_spatial = 3`.

## Result — the confound is refuted

**Connectivity failure is not a 2D artifact**, because the corpus's own 3D result fails it too,
at 3D's much more permissive threshold.

### The replication is exact

Transcribing `exp_11_pac_web_3d.py` literally, signs as written, at its own 32³ binning:

| | void | density CV |
|---|---|---|
| exp_11 reported | 0.89 | 2.94 |
| **transcription** | **0.888** | **2.948** |

### The published web does not percolate

| | void | cv | **percolation** | occupancy |
|---|---|---|---|---|
| exp_11 as written | 0.888 | 2.948 | **0.0068** | 0.1124 |
| attractive-gravity variant | 0.938 | 5.564 | **0.4252** | 0.0616 |
| 3D white-noise control | 0.060 | 0.753 | 0.0025 | 0.1085 |

exp_11's cosmic web percolates at **0.0068 against a noise control of 0.0025**, at essentially
identical occupancy. Rendered (`results/sign_convention_3d.png`), it is scattered isolated cells
spread uniformly through the box — a random point process with high contrast.

This is POC-05's finding in the corpus's own particle result: **void fraction and density CV are
one-point statistics.** They measure contrast and say nothing about connectivity. Not a
criticism of exp_11 on its own terms — it measured void, filament fraction, CV and a sampled
clustering coefficient, and measured them correctly. Percolation was not among them.

### And the variant that percolates is collapsed, not webbed

The attractive variant reaches 0.4252 from a *lower* occupancy, which is the right direction.
But its largest component spans only **0.47 of the box** on its shortest axis and coexists with
**231 other components**. One convention gives scattered points, the other gives collapse.
Neither gives a space-filling network.

## Withdraws a POC-07 claim

POC-07 recorded a *"sign error caught by the physics, not by a test"* in exp_09 — `d[i,j] =
pos_i − pos_j` points away from `j`, so the force reads as repulsive. **The reference
reproduces exactly as written**, so the repulsive convention is what produced the corpus's
published numbers. The v4 substrate runs a different system, which is legitimate, but calling it
a correction to exp_09 was wrong.

Both are now available: `CANONICAL` (attractive) and `EXP11` (as written), plus
`SECUpdateRelative` — exp_11's SEC rule differs from exp_09's in having no threshold and no
decay.

## Two measurement artifacts

**Percolation is severely resolution-dependent** for sparse particle data. The same 3D state:

| binning | 24³ | 32³ | 48³ | 64³ |
|---|---|---|---|---|
| percolation | 0.489 | 0.425 | 0.102 | 0.027 |
| void | 0.899 | 0.938 | 0.974 | 0.988 |

An 18× swing from grid choice alone. exp_11 measures at 32³; the first replication attempt used
64³ and read CV 15.4 against a target of 2.94.

**POC-07's "15×" compared a sparse field to a dense one.** At matched cell count (128²) the
field engine has occupancy 0.257 against the particles' 0.084 and still percolates worse
(0.0117 vs 0.0563). The conclusion survives — the particles win from a disadvantaged position —
but the multiplier is ~5×.

`web_metrics` now reports **`occupancy` on every call**. A percolation number without its
occupancy cannot be compared to anything.

## Running it

```
python proof_of_concepts/v4/structure.py      # includes selftest_3d(); must PASS
python proof_of_concepts/v4/poc_09_three_dimensions/scripts/exp_01_replicate_exp11.py
```

3D selftest expectations are **derived, not copied**: ξ = 1−1/e and ξ = 2σ are per-axis and
carry across dimension; `coherent_fraction`'s baseline is the exact discretised mode count
`Π(2c+1)/n` (0.0199 on 48³, not `band³` = 0.0156); and 3D white noise must not percolate because
occupancy 0.11 sits below the 3D threshold of 0.312.

## Open

- Sweep `sec_balance` in **exp_10's own convention**, past 1.3 so the optimum is bracketed —
  see `poc_07/journals/2026-08-16_xi-is-not-the-optimum.md`.
- Whether any parameter regime between "scattered" and "collapsed" gives a space-filling
  network at all.
