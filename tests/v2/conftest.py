"""Shared fixtures for v2 tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def force_cpu():
    """All v2 tests run on CPU for portability and speed."""
    pass  # Tests import with device="cpu" explicitly


@pytest.fixture
def seed():
    """Reproducible seed."""
    torch.manual_seed(42)
    return 42
