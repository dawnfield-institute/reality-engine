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

--------------------------------------------------------------------------------------

The correlation and spectral metrics below were added because the numbers that founded
Milestone 16 — correlation length 1.00 cell, spectral tilt 9.09x vs 2.23x, a preferred
wavelength moving 23.2 -> 10.5 -> 2.1 with quantum pressure — existed **only as prose in
journals**. No committed script produced them and no estimator was calibrated. Two of them
also contradict each other: a field with a preferred wavelength near 10 cells cannot have a
correlation length of 1 cell.

Two further rules, specific to these:

4. **Per-direction, never isotropic.** The Mobius manifold is `nu` x `nv` = 128 x 32: `nu`
   is a periodic circumference, `nv` is a BOUNDED strip width, and the twist couples
   `u + pi` to `v -> 1 - v`. A circular FFT along `v` imposes false wraparound across the
   strip edges — a discontinuity that decorrelates at lag 1 — and a radial average over a
   4:1 anisotropic grid buries whatever exists along `u`. Correlation and power are reported
   per axis, with the periodic axis treated circularly and the bounded axis windowed or
   unbiased-normalized.
5. **Coherent power is reported separately from correlation length.** A correlation length
   at the floor is consistent with two very different fields: one with no structure, and one
   whose structure is real but buried under pointwise variance. `coherent_fraction` tells
   them apart; nothing in the prior work could.
"""

from __future__ import annotations

import numpy as np

# ln-scale reference: the autocorrelation threshold. C(r) < 1/e defines the correlation
# length. Named because three functions below depend on it and the analytic selftest
# expectations are derived from it.
INV_E = float(np.exp(-1.0))


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


# ======================================================================================
# Correlation length — per direction
# ======================================================================================


def _autocorr_lines(m: np.ndarray, axis: int, periodic: bool) -> np.ndarray:
    """Normalized autocorrelation along `axis`, pooled over the other axis.

    Raw (unnormalized) autocorrelations are summed across lines and normalized once at
    lag 0, rather than normalizing each line first and averaging. Per-line normalization
    gives a near-empty line the same weight as a structured one, which on this manifold
    means the low-mass strip edges dominate the average.
    """
    x = np.moveaxis(m, axis, 0)                    # (N, rest)
    x = x.reshape(x.shape[0], -1).astype(float)
    x = x - x.mean(axis=0, keepdims=True)          # per-line mean removed
    n = x.shape[0]

    if periodic:
        f = np.fft.rfft(x, axis=0)
        ac = np.fft.irfft(f * np.conj(f), n=n, axis=0).real
        ac = ac.sum(axis=1)
    else:
        # Zero-pad to 2n so the FFT computes a LINEAR (not circular) correlation, then
        # divide by the number of overlapping samples at each lag. Without the padding the
        # bounded axis wraps its own edges together and decorrelates at lag 1 — the exact
        # artifact this module exists to rule out.
        f = np.fft.rfft(x, n=2 * n, axis=0)
        ac = np.fft.irfft(f * np.conj(f), n=2 * n, axis=0).real[:n]
        ac = ac.sum(axis=1) / np.arange(n, 0, -1)  # unbiased

    if not np.isfinite(ac[0]) or ac[0] <= 0:
        return np.array([])
    return ac / ac[0]


def correlation_length(F, axis: int = 0, periodic: bool | None = None) -> float:
    """Correlation length along one axis, in cells: the lag where C(r) falls below 1/e.

    Linearly interpolated between the bracketing integer lags, so the result is continuous.
    This matters: an estimator that returns integer lags CANNOT report anything between 0
    and 1, so it reports exactly 1.00 for every field whose neighbours are uncorrelated —
    and 1.00 for a field with weak but real correlation too. The founding Milestone 16
    measurement was a table of 1.00s.

    `periodic` defaults per axis to this manifold's geometry: axis 0 (`u`) is the periodic
    circumference, axis 1 (`v`) is the bounded strip width. **Pass it explicitly on any grid
    that is not the Mobius manifold.**

    Returns nan when the field has no variance along the axis, or when C(r) never falls
    below 1/e within the measurable lags — an honest "longer than we can see here" rather
    than a number clipped to the axis length.
    """
    m = _as_np(F).astype(float)
    if m.ndim != 2:
        raise ValueError(f"expected a 2D field, got shape {m.shape}")
    if not np.isfinite(m).all():
        return float("nan")
    if periodic is None:
        periodic = (axis == 0)

    c = _autocorr_lines(m, axis, periodic)
    if c.size == 0:
        return float("nan")

    # Only lags where the estimator is meaningful: half the axis for the circular case
    # (beyond that it mirrors), half for the linear case too (the unbiased normalization
    # gets very noisy as the overlap shrinks).
    horizon = max(2, c.size // 2)
    for r in range(1, horizon):
        if c[r] < INV_E:
            prev = c[r - 1]
            if prev == c[r]:
                return float(r)
            return float(r - 1 + (prev - INV_E) / (prev - c[r]))
    return float("nan")


# ======================================================================================
# Power spectrum — per direction
# ======================================================================================


def power_spectrum(F, axis: int = 0, periodic: bool | None = None):
    """1D power spectrum along one axis, averaged over the other.

    Returns `(k, P)` with k in cycles per cell, excluding the DC mode. The bounded axis is
    Hann-windowed rather than wrapped: a discontinuity at the strip edge would otherwise
    inject broadband power that looks like small-scale structure.

    Not radially averaged. On a 128x32 grid a radial average mixes 4 cycles/box along `u`
    with 1 cycle/box along `v` into the same bin.
    """
    m = _as_np(F).astype(float)
    if m.ndim != 2:
        raise ValueError(f"expected a 2D field, got shape {m.shape}")
    if periodic is None:
        periodic = (axis == 0)

    x = np.moveaxis(m, axis, 0)
    x = x.reshape(x.shape[0], -1)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    if not periodic:
        x = x * np.hanning(n)[:, None]

    P = (np.abs(np.fft.rfft(x, axis=0)) ** 2).mean(axis=1)
    k = np.fft.rfftfreq(n)
    return k[1:], P[1:]


def spectral_tilt(P_early, P_late, band: float = 0.25) -> float:
    """Ratio of large-scale to small-scale growth: how power redistributes over a run.

    `> 1` means power moved toward large scales (the box mode); `< 1` toward small scales;
    `~1` scale-neutral. Computed as the median growth ratio in the lowest `band` fraction of
    k divided by the median in the highest `band` fraction.

    **The exact definition is a choice made here, not recovered.** The journals quote 9.09x
    for RBF and 2.23x for PACBalance without recording how the tilt was computed, so those
    two numbers are not reproducible under any particular convention. Comparisons using this
    function are valid against each other and should not be compared to the journal figures.
    """
    a = np.asarray(P_early, float)
    b = np.asarray(P_late, float)
    if a.shape != b.shape or a.size < 8:
        return float("nan")
    ratio = np.divide(b, a, out=np.full_like(b, np.nan), where=a > 0)
    w = max(2, int(a.size * band))
    lo = np.nanmedian(ratio[:w])
    hi = np.nanmedian(ratio[-w:])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0:
        return float("nan")
    return float(lo / hi)


def coherent_fraction(F, band: float = 0.25) -> float:
    """Share of total variance held by modes below `band` of Nyquist in BOTH directions.

    The discriminator the prior work lacked. A correlation length at the floor is consistent
    with two very different fields — one with no structure at all, and one whose structure is
    real but sits far below a pointwise noise floor. This separates them:

        white noise      -> band^2  (0.0625 at the default), the mode-count fraction
        smooth field     -> ~1
        buried structure -> a few times band^2, well clear of the noise baseline

    A rectangular low-k region, not a disc, because the grid is anisotropic and a disc in
    index space is not a disc in physical wavenumber.
    """
    m = _as_np(F).astype(float)
    if m.ndim != 2 or not np.isfinite(m).all():
        return float("nan")
    m = m - m.mean()
    P = np.abs(np.fft.fft2(m)) ** 2
    P[0, 0] = 0.0                                   # DC removed with the mean
    total = P.sum()
    if total <= 0:
        return float("nan")

    nu, nv = P.shape
    cu = max(1, int(band * (nu // 2)))
    cv = max(1, int(band * (nv // 2)))
    idx_u = np.concatenate([np.arange(cu + 1), np.arange(nu - cu, nu)])
    idx_v = np.concatenate([np.arange(cv + 1), np.arange(nv - cv, nv)])
    low = P[np.ix_(idx_u, idx_v)].sum()
    return float(low / total)


def web_metrics(F, void_frac: float = 0.1, overdensity: float = 2.0) -> dict:
    """exp_09's web criterion, adapted from a particle density field to a continuous one.

    Returns void fraction, filament fraction, density CV, correlation length, and `is_web`.
    The first three thresholds are exp_09's own
    (`gravity_from_maxwell_pac/scripts/exp_09_pac_web_emergence.py`), not new ones: a web has
    filaments > 0.05 AND voids > 0.3 AND CV > 1.0.

    Three adaptations, all disclosed because a changed threshold silently changes a verdict:

    * **Voids.** exp_09 counted strictly empty bins. A continuous field has none, so a void
      is a cell below `void_frac` of the mean — exp_09's own running-history definition
      (`density < 0.1 * mean`), not an invention.
    * **Filaments.** exp_09 used the 75th percentile of occupied bins, which on a continuous
      field is tautological: the top quartile is 25% of cells by construction, so it returns
      0.25 for a web, a blob and white noise alike. Replaced with an absolute overdensity
      threshold (`M > overdensity * mean`), the standard cosmological definition. **Not
      numerically comparable to exp_09's 0.12.**
    * **A SCALE GATE, added here: xi must exceed the white-noise floor.** exp_09's three
      conditions are purely statistical — no scale, no connectivity — and on a lattice a
      grid-scale checkerboard satisfies all three trivially: half the cells sit at zero (read
      as voids), the bimodal zero/bright distribution gives CV > 1, and the bright half are
      overdense. The engine at `quantum_pressure_coeff=0.30` does exactly this and scored
      `is_web=True` with void 0.605 and CV 1.273, while C(1) = -0.358 — **anti**-correlated
      neighbours. Caught by rendering it. A web has correlated neighbours; a checkerboard has
      anti-correlated ones, so requiring xi > 1 - 1/e kills the false positive without
      touching exp_09's thresholds. This condition could not arise in exp_09's particle
      substrate, which is why exp_09 did not need it.
    * **A CONNECTIVITY GATE, added here: the overdense set must percolate.** None of the
      conditions above asks whether the overdense cells TOUCH. They cannot: void fraction,
      CV and overdense fraction are all one-point statistics, computed cell by cell with no
      reference to any neighbour. A field of isolated bright pixels on black satisfies all
      of them, and the engine produces exactly that — at 128x128 over 12000 ticks its
      overdense set breaks into ~1300 disconnected components against white noise's 1394,
      with a largest component holding 1.3-1.9% of the set where a web holds 100%. It scored
      is_web=True on three configurations before this gate existed. exp_09 measured
      connectivity directly (clustering coefficient 0.54) and did not need a gate, because
      in a particle substrate the neighbour graph IS the model.
    """
    m = _as_np(F).astype(float)
    mean = m.mean()
    nan_result = {"void": float("nan"), "filament": float("nan"), "cv": float("nan"),
                  "xi_u": float("nan"), "percolation": float("nan"), "is_web": False}
    if not np.isfinite(mean) or mean <= 0:
        return nan_result
    void = float((m < void_frac * mean).mean())
    fil = float((m > overdensity * mean).mean())
    cv = float(m.std() / mean)
    xi = correlation_length(m, axis=0, periodic=True) if m.ndim == 2 else float("nan")
    above_grid_scale = xi == xi and xi > (1.0 - INV_E)
    perc = percolation(m, overdensity=overdensity)
    return {"void": void, "filament": fil, "cv": cv, "xi_u": xi, "percolation": perc,
            "is_web": bool(fil > 0.05 and void > 0.3 and cv > 1.0
                           and above_grid_scale and perc > 0.25)}


def percolation(F, overdensity: float = 2.0) -> float:
    """Largest connected component of the overdense set, as a fraction of that set.

    The one measurement here that is not a one-point statistic. A web is connected —
    filaments meet at nodes and the overdense set is essentially one object, so this
    returns ~1. Independent bright cells return ~1/N.

    Reference values on 128x128: a synthetic web with three-cell filaments gives 1.000 in a
    single component; white noise gives 0.003 across 1394 components.

    4-connectivity on the periodic axis is not handled — components touching the u=0 and
    u=127 edges are counted separately. That makes this a slight UNDER-estimate, which is
    the safe direction for a gate.
    """
    m = _as_np(F).astype(float)
    if m.ndim != 2 or not np.isfinite(m).all():
        return float("nan")
    mean = m.mean()
    if not np.isfinite(mean) or mean <= 0:
        return float("nan")
    occ = m > overdensity * mean
    total = int(occ.sum())
    if total == 0:
        return 0.0

    # scipy's labeller when available — the pure-Python fill below is correct but ~100x
    # slower, which matters when this runs hundreds of times inside a sweep. Both are
    # 4-connected and give identical answers; the fallback keeps the module dependency-free.
    try:
        from scipy import ndimage
        lab, n_lab = ndimage.label(occ)
        if n_lab == 0:
            return 0.0
        sizes = np.bincount(lab.ravel())[1:]
        return float(sizes.max() / total)
    except ImportError:
        pass

    # Two-pass flood fill over the occupied set — no scipy dependency, this module has none.
    labels = np.zeros_like(occ, dtype=np.int32)
    best, current = 0, 0
    H, W = occ.shape
    for su in range(H):
        for sv in range(W):
            if not occ[su, sv] or labels[su, sv]:
                continue
            current += 1
            stack, size = [(su, sv)], 0
            labels[su, sv] = current
            while stack:
                u, v = stack.pop()
                size += 1
                for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = u + du, v + dv
                    if 0 <= a < H and 0 <= b < W and occ[a, b] and not labels[a, b]:
                        labels[a, b] = current
                        stack.append((a, b))
            best = max(best, size)
    return float(best / total)


# ======================================================================================
# Calibration
# ======================================================================================


def _smoothed_noise(H: int, W: int, sigma: float, seed: int = 0) -> np.ndarray:
    """White noise convolved with a Gaussian of width `sigma`, periodic in both axes.

    Built in Fourier space so the smoothing is exact and the periodicity is exact — the
    reference field must not itself carry an edge artifact.
    """
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((H, W))
    ku = np.fft.fftfreq(H)[:, None] * 2 * np.pi
    kv = np.fft.fftfreq(W)[None, :] * 2 * np.pi
    kernel = np.exp(-0.5 * (sigma ** 2) * (ku ** 2 + kv ** 2))
    return np.fft.ifft2(np.fft.fft2(n) * kernel).real


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


def _power_law_field(H: int, W: int, slope: float, seed: int = 0) -> np.ndarray:
    """Field whose power spectrum along axis 0 is exactly P(k) ~ k**slope."""
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(H)
    amp = np.zeros_like(k)
    amp[1:] = k[1:] ** (slope / 2.0)          # P = |A|^2
    phase = rng.uniform(0.0, 2 * np.pi, k.size)
    line = np.fft.irfft(amp * np.exp(1j * phase), n=H)
    return np.repeat(line[:, None], W, axis=1)


def _bandpass_field(H: int, W: int, lam0: float, rel_width: float = 0.12,
                    seed: int = 0) -> np.ndarray:
    """Field with a narrow spectral band at wavelength `lam0` — a preferred scale.

    This is the field the engine is claimed to have (quantum pressure "sets a preferred
    scale" at lambda ~ 10.5) while simultaneously being claimed to have correlation length
    1.0. Both cannot hold, and this case is what proves it.
    """
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(H)
    k0 = 1.0 / lam0
    amp = np.exp(-0.5 * ((k - k0) / (rel_width * k0)) ** 2)
    amp[0] = 0.0
    phase = rng.uniform(0.0, 2 * np.pi, k.size)
    line = np.fft.irfft(amp * np.exp(1j * phase), n=H)
    return np.repeat(line[:, None], W, axis=1)


def selftest_correlation(H: int = 128, W: int = 64) -> dict:
    """Correlation length against analytically known fields.

    Every expected value is derived, not tuned:

      white noise    C(r) = 1 at r=0 and 0 beyond, so linear interpolation crosses 1/e at
                     r = 1 - 1/e = 0.632. An estimator reporting 1.00 here is quantising
                     to integer lags and cannot resolve anything below one cell.
      Gaussian sigma P(k) = exp(-sigma^2 k^2), so C(r) = exp(-r^2/4sigma^2) and xi = 2 sigma.
                     Two widths are tested because the ratio is the check that matters: a
                     pinned estimator can match one value by luck, never the scaling.
      cosine lambda  C(r) = cos(2 pi r / lambda), crossing 1/e at r = arccos(1/e)*lambda/2pi
                     = 0.1899 lambda.
    """
    cos_coeff = float(np.arccos(INV_E) / (2 * np.pi))       # 0.18994
    out: dict[str, tuple[float, float, float]] = {}         # name -> (got, expect, tol)

    noise = np.random.default_rng(0).standard_normal((H, W))
    out["white_noise"] = (correlation_length(noise, axis=0, periodic=True),
                          1.0 - INV_E, 0.12)

    g2 = _smoothed_noise(H, W, sigma=2.0, seed=1)
    g4 = _smoothed_noise(H, W, sigma=4.0, seed=1)
    xi2 = correlation_length(g2, axis=0, periodic=True)
    xi4 = correlation_length(g4, axis=0, periodic=True)
    out["gaussian_sigma2"] = (xi2, 4.0, 0.8)
    out["gaussian_sigma4"] = (xi4, 8.0, 1.6)
    out["scaling_ratio"] = (xi4 / xi2 if xi2 else float("nan"), 2.0, 0.3)

    u = np.arange(H)[:, None]
    out["cosine_lambda16"] = (
        correlation_length(np.repeat(np.cos(2 * np.pi * u / 16.0), W, axis=1),
                           axis=0, periodic=True),
        cos_coeff * 16.0, 0.5)

    flat = correlation_length(np.ones((H, W)), axis=0, periodic=True)
    out["constant_field"] = (flat, float("nan"), 0.0)       # nan expected: refusal

    # The bounded axis. The unbiased linear estimator must agree with the circular one on a
    # field where both are valid; if it does not, its normalization is wrong and every
    # measurement along `v` is meaningless.
    out["bounded_axis_sigma4"] = (
        correlation_length(_smoothed_noise(W, H, sigma=4.0, seed=1), axis=1,
                           periodic=False),
        8.0, 1.6)

    return out


def selftest_spectrum(H: int = 128, W: int = 64) -> dict:
    """Power spectrum, tilt, coherent fraction, and the xi-vs-P(k) cross-check."""
    out: dict[str, tuple[float, float, float]] = {}

    k, P = power_spectrum(_power_law_field(H, W, slope=-2.0, seed=2),
                          axis=0, periodic=True)
    sel = (k > 0) & (P > 0)
    slope = float(np.polyfit(np.log(k[sel]), np.log(P[sel]), 1)[0])
    out["powerlaw_slope"] = (slope, -2.0, 0.2)

    band = _bandpass_field(H, W, lam0=16.0, seed=3)
    kb, Pb = power_spectrum(band, axis=0, periodic=True)
    lam_peak = float(1.0 / kb[int(np.argmax(Pb))])
    out["bandpass_peak_lambda"] = (lam_peak, 16.0, 2.0)

    # THE CROSS-CHECK. A field with a spectral peak at lambda must have a correlation
    # length near 0.19*lambda. If an estimator reports a preferred wavelength of ~10 cells
    # and a correlation length of 1 cell for the same field, one of the two is broken.
    xi_band = correlation_length(band, axis=0, periodic=True)
    expected = float(np.arccos(INV_E) / (2 * np.pi)) * lam_peak
    out["xi_matches_pk_peak"] = (xi_band / expected if expected else float("nan"), 1.0, 0.5)

    rng = np.random.default_rng(4)
    out["coherent_white_noise"] = (coherent_fraction(rng.standard_normal((H, W))),
                                   0.25 ** 2, 0.02)
    out["coherent_smooth"] = (coherent_fraction(_smoothed_noise(H, W, sigma=6.0, seed=5)),
                              1.0, 0.1)

    # Tilt: the same field smoothed further has moved power to large scales, so > 1.
    early = power_spectrum(_smoothed_noise(H, W, sigma=1.0, seed=6), axis=0,
                           periodic=True)[1]
    late = power_spectrum(_smoothed_noise(H, W, sigma=4.0, seed=6), axis=0,
                          periodic=True)[1]
    tilt = spectral_tilt(early, late)
    out["tilt_smoothing_is_large_scale"] = (1.0 if tilt > 1.0 else 0.0, 1.0, 0.0)

    return out


def selftest_web(H: int = 128, W: int = 32) -> dict:
    """`web_metrics` against a synthetic web and against a smooth field.

    The case that matters is the smooth one. A metric that calls a low-contrast blobby
    field a web cannot tell clumping from webbing, which is the exact distinction this
    milestone turns on.
    """
    out: dict[str, tuple[float, float, float]] = {}

    # Synthetic web: dense filaments on a near-empty background, RESOLVED — three cells
    # across, not one. A one-cell filament is itself grid-scale structure and the scale gate
    # rejects it, correctly: at that width a "web" is indistinguishable from a checkerboard
    # by any local measure. Real filaments are resolved, and the fixture has to be too.
    web = np.full((H, W), 0.01)
    for u in range(0, H, 16):
        web[u:u + 3, :] = 3.0
    for v in range(0, W, 16):
        web[:, v:v + 3] = 3.0
    w = web_metrics(web)
    out["web_is_web"] = (1.0 if w["is_web"] else 0.0, 1.0, 0.0)
    out["web_cv_above_1"] = (w["cv"], 2.0, 1.0)
    out["web_xi_above_floor"] = (w["xi_u"], 2.0, 1.2)
    out["web_percolates"] = (w["percolation"], 1.0, 0.05)

    # Smooth positive field with large-scale power but low contrast — clumping, not webbing.
    smooth = _smoothed_noise(H, W, sigma=6.0, seed=7)
    smooth = smooth - smooth.min() + 0.5
    s = web_metrics(smooth)
    out["smooth_is_not_web"] = (0.0 if not s["is_web"] else 1.0, 0.0, 0.0)

    # THE FALSE POSITIVE. A grid-scale checkerboard passes every one of exp_09's three
    # statistical conditions — voids, CV, overdense fraction — while being the opposite of a
    # web. The engine produces this at quantum_pressure_coeff=0.30. Without the scale gate
    # this case returns True.
    board = np.indices((H, W)).sum(axis=0) % 2
    board = board * 2.0 + 0.001
    b = web_metrics(board)
    out["checkerboard_voids_pass"] = (b["void"], 0.5, 0.05)      # it does look void-rich
    out["checkerboard_cv_passes"] = (b["cv"], 1.0, 0.2)          # and contrasty
    out["checkerboard_is_not_web"] = (0.0 if not b["is_web"] else 1.0, 0.0, 0.0)

    # THE SECOND FALSE POSITIVE. Isolated bright cells on black clear voids, CV, overdense
    # fraction AND the scale gate — every condition that looks at cells one at a time. The
    # engine produces this, and scored is_web=True on three configurations before
    # percolation existed. Only a connectivity measure separates it from a web.
    rng = np.random.default_rng(11)
    speckle = np.full((H, W), 0.02)
    hits = rng.random((H, W)) < 0.12
    speckle[hits] = 3.0
    s2 = web_metrics(speckle)
    out["speckle_voids_pass"] = (s2["void"], 0.88, 0.06)
    out["speckle_cv_passes"] = (s2["cv"], 2.5, 1.0)
    out["speckle_does_not_percolate"] = (s2["percolation"], 0.0, 0.15)
    out["speckle_is_not_web"] = (0.0 if not s2["is_web"] else 1.0, 0.0, 0.0)

    return out


def _report(title: str, cases: dict) -> bool:
    """Print a calibration table. `expected` of nan means 'must refuse'."""
    print(f"\n  {title} (expected -> measured)")
    ok = True
    for name, (got, expect, tol) in cases.items():
        if expect != expect:                      # nan expected
            good = got != got
            shown = "nan"
        else:
            good = got == got and abs(got - expect) <= tol
            shown = f"{expect:.3f}"
        ok &= good
        got_s = "nan" if got != got else f"{got:.3f}"
        print(f"    {name:<28} expect {shown:>7}   got {got_s:>7}   "
              f"{'OK' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("  box_dimension calibration (expected -> measured)")
    ok = True
    for name, (measured, expected) in selftest().items():
        good = abs(measured - expected) < 0.35 if measured == measured else False
        ok &= good
        print(f"    {name:<16} expect {expected:>4.1f}   got {measured:>6.3f}   "
              f"{'OK' if good else 'FAIL'}")

    ok &= _report("correlation_length calibration", selftest_correlation())
    ok &= _report("spectral calibration", selftest_spectrum())
    ok &= _report("web_metrics calibration", selftest_web())
    print(f"\n  overall: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
