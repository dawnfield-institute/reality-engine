"""
Emergence detection and MED metrics.

MED (Macro Emergence Dynamics): all complex flows converge to symbolic
patterns with depth ≤ 2 and nodes ≤ 3.

This module detects when structure has formed in the fields and
characterises its complexity.
"""

from __future__ import annotations

import torch


def detect_structures(
    field: torch.Tensor,
    threshold: float = 1.5,
) -> dict[str, float]:
    """
    Detect emergent structures in *field* above *threshold* × σ.

    Returns dictionary of structure metrics:
        n_peaks     — number of local maxima above threshold
        max_val     — largest peak value
        mean_val    — mean of peaks
        coverage    — fraction of grid above threshold
    """
    mean = field.mean()
    std = field.std()
    mask = field > (mean + threshold * std)
    peaks = field[mask]

    return {
        "n_peaks": int(mask.sum().item()),
        "max_val": float(peaks.max().item()) if peaks.numel() > 0 else 0.0,
        "mean_val": float(peaks.mean().item()) if peaks.numel() > 0 else 0.0,
        "coverage": float(mask.float().mean().item()),
    }


def complexity_depth(field: torch.Tensor, levels: int = 4) -> int:
    """
    Estimate hierarchical depth of structures via successive coarse-graining.

    Returns the number of levels at which structure (σ > 0.01) persists.
    MED predicts this should saturate at ≤ 2.
    """
    depth = 0
    f = field.clone()
    for _ in range(levels):
        if f.shape[0] < 4 or f.shape[1] < 4:
            break
        if f.std() < 0.01:
            break
        depth += 1
        # Coarse-grain by 2× averaging
        nu, nv = f.shape
        f = f[: nu - nu % 2, : nv - nv % 2]  # trim to even
        f = f.reshape(nu // 2, 2, nv // 2, 2).mean(dim=(1, 3))
    return depth
