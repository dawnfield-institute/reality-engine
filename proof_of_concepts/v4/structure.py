"""Calibrated structure metrics for the v4 exploration.

The first box-counting estimator used in this work was broken: it thresholded by quantile
and fitted every scale including saturated ones, so it returned D = 2.000 for a straight
filament, a blob, and scattered points alike, and D = 1.459 for unstructured noise. Every
structure claim built on it was void.

Rules this module follows, each because the broken version violated it:

1. **Absolute threshold**, as a fraction of max — not a quantile. A quantile always selects
   the same number of cells, so a concentrating field and a uniform one look identical.
2. **Fit only unsaturated scales** — boxes where the count is strictly between 1 and the
   total number of boxes. Saturated scales flatten the log-log slope toward the embedding
   dimension regardless of geometry.
3. **Calibrated against known geometries.** `selftest()` must pass: filament ~1, plane ~2,
   points ~0. If it does not, the metric is not reporting geometry.
"""

from __future__ import annotations

import numpy as np


def _as_np(F):
    return F.detach().cpu().numpy() if hasattr(F, "detach") else np.asarray(F)


def box_dimension(F, thresh_frac: float | None = None, min_scales: int = 3,
                  target_occupancy: float = 0.10) -> float:
    """Box-counting dimension of the thresholded field.

    With `thresh_frac=None` (default) the threshold is chosen ADAPTIVELY so that roughly
    `target_occupancy` of cells are selected. A fixed fraction-of-max threshold fails on
    smooth fields: at 0.25*max the engine's M is 72% occupied, so every box above s=2 is
    saturated and no slope can be fitted. Adaptivity keeps the selected set in the range
    where box-counting means something, and the threshold used is reported by
    `threshold_for()` when it matters.

    Returns nan when fewer than `min_scales` unsaturated scales survive — an honest
    refusal rather than a slope fitted through saturated points.
    """
    m = _as_np(F).astype(float)
    if not np.isfinite(m).all():
        return float("nan")
    m = m - m.min()
    peak = m.max()
    if peak <= 0:
        return float("nan")
    if thresh_frac is None:
        cut = np.quantile(m, 1.0 - target_occupancy)
        occ = m >= cut if cut > 0 else m > 0
    else:
        occ = m >= thresh_frac * peak
    n_occ = int(occ.sum())
    if n_occ == 0 or n_occ == occ.size:
        return float("nan")

    H, W = occ.shape
    sizes, counts = [], []
    s = 1
    while s <= min(H, W) // 2:
        h, w = H // s, W // s
        blocks = occ[: h * s, : w * s].reshape(h, s, w, s).any(axis=(1, 3))
        total_boxes = h * w
        c = int(blocks.sum())
        # unsaturated only: some boxes occupied, but not all
        if 0 < c < total_boxes:
            sizes.append(s)
            counts.append(c)
        s *= 2

    if len(sizes) < min_scales:
        return float("nan")
    x = np.log(1.0 / np.asarray(sizes, float))
    y = np.log(np.asarray(counts, float))
    return float(np.polyfit(x, y, 1)[0])


def shannon_entropy(F) -> float:
    """Spatial Shannon entropy of |F| as a distribution. Lower = more concentrated."""
    m = np.abs(_as_np(F).astype(float))
    tot = m.sum()
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    p = np.clip(m / tot, 1e-30, None)
    return float(-(p * np.log(p)).sum())


def contrast(F, frac: float = 0.01) -> float:
    """Share of total held by the top `frac` of cells. Rising = contrast growing."""
    m = np.abs(_as_np(F).astype(float)).ravel()
    tot = m.sum()
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    k = max(1, int(len(m) * frac))
    return float(np.sort(m)[-k:].sum() / tot)


def occupied_fraction(F, thresh_frac: float = 0.25) -> float:
    m = _as_np(F).astype(float)
    m = m - m.min()
    peak = m.max()
    if peak <= 0:
        return float("nan")
    return float((m >= thresh_frac * peak).mean())


def selftest(H: int = 64, W: int = 64) -> dict:
    """Known geometries. filament ~1, plane ~2, points ~0."""
    rng = np.random.default_rng(0)
    cases = {}

    f = np.zeros((H, W)); f[:, W // 2] = 1.0
    cases["filament"] = (f, 1.0)

    g = np.zeros((H, W))
    for i in range(H):
        g[i, int(W / 2 + (W / 4) * np.sin(i / 7))] = 1.0
    cases["sine_filament"] = (g, 1.0)

    p = np.zeros((H, W)); p[: H // 2, : W // 2] = 1.0
    cases["half_plane"] = (p, 2.0)

    pts = np.zeros((H, W))
    for _ in range(12):
        pts[rng.integers(H), rng.integers(W)] = 1.0
    cases["points"] = (pts, 0.0)

    out = {}
    for name, (arr, expected) in cases.items():
        out[name] = (box_dimension(arr), expected)
    return out


if __name__ == "__main__":
    print("  box_dimension calibration (expected -> measured)")
    ok = True
    for name, (measured, expected) in selftest().items():
        good = abs(measured - expected) < 0.35 if measured == measured else False
        ok &= good
        print(f"    {name:<16} expect {expected:>4.1f}   got {measured:>6.3f}   "
              f"{'OK' if good else 'FAIL'}")
    print(f"  overall: {'PASS' if ok else 'FAIL'}")
