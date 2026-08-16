# 2026-08-16: exp_11 replicates exactly — and its web does not percolate

Three findings, in the order they arrived. The middle one is a correction to my own work.

---

## 1. The replication is exact

Transcribing `exp_11_pac_web_3d.py` literally — signs and all — and measuring at its own
32³ binning:

| | void | density CV |
|---|---|---|
| **exp_11 reported** | **0.89** | **2.94** |
| **transcription** | **0.888** | **2.948** |

Three significant figures on both. The rebuild is faithful and the reference is reproducible.

## 2. My "sign fix" in POC-07 was a deviation, not a fix

`deltas[i,j] = pos_i − pos_j` points *away* from `j`. Both exp_09 and exp_11 use
`force_dir = +deltas` for gravity and `−deltas` for entropy pressure, so **their gravity is
repulsive and their pressure is attractive.** I read that as an error, flipped it, and wrote a
commit message saying exp_09 carried a sign bug with a comment claiming otherwise.

The reference reproduces to three figures *as written*. Whatever the labels say, that is the
system that produced the published numbers, and my substrate was not reproducing it. POC-07's
claim that this was a bug caught "by the physics, not by a test" is withdrawn.

Both conventions are now available: `CANONICAL` (attractive, my variant) and `EXP11`
(as-written), alongside `SECUpdateRelative` — exp_11's SEC rule differs from exp_09's too, in
having no threshold and no decay.

## 3. The published web does not percolate

3D, exp_11 config, 32³, with occupancy reported because percolation is meaningless without it:

| | void | cv | **percolation** | occupancy |
|---|---|---|---|---|
| exp_11 as written | 0.888 | 2.948 | **0.0068** | 0.1124 |
| attractive-gravity variant | 0.938 | 5.564 | **0.4252** | 0.0616 |
| 3D white noise control | 0.060 | 0.753 | 0.0025 | 0.1085 |

**exp_11's published cosmic web percolates at 0.0068 against a white-noise control of 0.0025,
at essentially identical occupancy (0.112 vs 0.109).** Rendered, it is scattered isolated cells
distributed uniformly through the box — a random point process with high contrast.

This is POC-05's finding again, in the corpus's own particle result rather than in the field
engine: **void fraction and density CV are one-point statistics.** They measure contrast. They
say nothing about whether anything is connected, and when connectivity is measured directly the
published web is indistinguishable from noise.

That is not a criticism of exp_11 on its own terms — it reported void, filament fraction, CV
and a sampled clustering coefficient, all of which it measured correctly. Percolation simply
was not among them.

## And the variant that percolates is collapsed, not webbed

The attractive-gravity variant reaches percolation 0.425 from a *lower* occupancy, which is the
right direction. But its largest component **spans only 0.47 of the box on its shortest axis**,
holds 0.425 of the overdense set, and coexists with **231 other components**. The render shows
mass concentrated into part of the volume with the outer slices nearly empty.

That is a large partially-collapsed clump, not a space-filling network. Neither sign convention
produces a cosmic web in the connectivity sense: one gives scattered points, the other gives
collapse, and the web regime sits between them.

## Two measurement artifacts found on the way

**Percolation is severely resolution-dependent for sparse particle data.** The same 3D state
binned to 24³ / 32³ / 48³ / 64³ gives percolation 0.489 / 0.425 / 0.102 / 0.027 — an 18× swing
from grid choice alone, with void and CV moving too (0.899 → 0.988). exp_11 measures at 32³;
this rebuild had been using 64³, which is why the first replication attempt read CV 15.4
against a target of 2.94.

**POC-07's "15× the field engine" compared a sparse field to a dense one.** At matched cell
count (128²) the field engine has occupancy 0.257 against the particles' 0.084 and still
percolates worse — 0.0117 vs 0.0563. The conclusion survives and is arguably strengthened,
since the particles win from a disadvantaged position, but the multiplier is ~5×, not 15×.

`web_metrics` now reports `occupancy` on every call. A percolation number without its occupancy
cannot be compared to anything, because percolation is a function of how far the occupied
fraction sits from the site threshold — 0.593 in 2D, 0.312 in 3D.

## Standing

The 2D-confound question that motivated this round is partly answered: connectivity failure is
**not** a 2D artifact, because the corpus's own 3D result fails it too, at 3D's much more
permissive threshold.

Seven instrument faults across this round now, none caught by a statistic at face value. This
one was caught by a replication gate that succeeded — the numbers matched exactly, and the
measurement that had never been taken is the one that mattered.
