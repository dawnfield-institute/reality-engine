"""connectivity_at_occupancy and cic_deposit — the exp_31 instrument (rank threshold, count deposit)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proof_of_concepts" / "v4"))
from structure import cic_deposit, connectivity_at_occupancy, percolation  # noqa: E402


def _web(res=16, noise=1e-3, seed=0):
    """Three orthogonal filaments through a cube on a faint noise floor: one connected object."""
    rng = np.random.RandomState(seed)
    F = noise * rng.rand(res, res, res)
    c = res // 2
    F[:, c, c] = 1.0; F[c, :, c] = 1.0; F[c, c, :] = 1.0
    return F


def test_cic_conserves_count_and_wraps():
    rng = np.random.RandomState(1); pos = rng.uniform(0, 60.0, size=(4000, 3))
    C = cic_deposit(pos, 60.0, 16)
    assert C.shape == (16, 16, 16)
    assert abs(C.sum() - 4000.0) < 1e-8
    # a particle at the box edge deposits across the periodic boundary, not into a clamped cell
    E = cic_deposit(np.array([[59.99, 30.0, 30.0]]), 60.0, 16)
    assert E[15].sum() > 0 and E[0].sum() > 0 and abs(E.sum() - 1.0) < 1e-12


Q_WEB = 46.5 / 16 ** 3   # selects exactly the 46 filament cells (3 x 16 minus two overlaps) and nothing else


def test_web_is_one_component_at_the_spine():
    F = _web()
    assert connectivity_at_occupancy(F, Q_WEB) == pytest.approx(1.0)
    # a strict subset of the filaments (rank ties broken by the noise floor) may or may not stay
    # connected; it must never read as MORE connected than the whole web
    assert connectivity_at_occupancy(F, 0.008) <= 1.0


def test_noise_is_fragmented():
    rng = np.random.RandomState(2); F = rng.rand(16, 16, 16)
    assert connectivity_at_occupancy(F, 0.10) < 0.3


def test_occupancy_is_exact_and_rank_based():
    rng = np.random.RandomState(3); F = rng.rand(16, 16, 16)
    q = 0.10; k = int(np.floor(q * F.size))
    m = F.ravel(); order = np.argsort(-m, kind="stable")[:k]
    mask = np.zeros(F.size); mask[order] = 1.0
    assert connectivity_at_occupancy(F, q) == pytest.approx(percolation(mask.reshape(F.shape), overdensity=0.5))
    assert int(mask.sum()) == k


def test_independent_of_a_global_rescale():
    rng = np.random.RandomState(4); F = rng.rand(16, 16, 16) ** 3
    for q in (0.05, 0.10, 0.20):
        assert connectivity_at_occupancy(F, q) == pytest.approx(connectivity_at_occupancy(3.7 * F, q))


def test_a_fatter_field_does_not_score_higher_at_fixed_occupancy():
    """The failure mode of percolation-at-fixed-overdensity: adding diffuse mass around a web raises its
    occupancy and its percolation. At fixed occupancy the diffuse mass is ranked out and nothing changes."""
    F = _web(); G = F + 0.3 * np.random.RandomState(5).rand(*F.shape) ** 3   # skewed diffuse mass: some cells clear 2x the mean, none reach the filaments
    assert connectivity_at_occupancy(F, Q_WEB) == pytest.approx(connectivity_at_occupancy(G, Q_WEB)) == pytest.approx(1.0)
    # and at a fixed OVERDENSITY the fatter field's occupied set is larger — the confound this observable removes
    assert (G > 2.0 * G.mean()).mean() != (F > 2.0 * F.mean()).mean()


def test_nan_when_nothing_selected():
    assert np.isnan(connectivity_at_occupancy(np.zeros((4, 4, 4)), 0.001))
