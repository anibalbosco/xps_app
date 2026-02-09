"""Shared fixtures for XPS analysis tests."""

import pytest

from xpsanalysis.synthetic import generate_c1s, generate_fe2p


@pytest.fixture
def c1s_spectrum():
    """A reproducible synthetic C 1s spectrum."""
    return generate_c1s(noise_level=0.01, seed=42)


@pytest.fixture
def fe2p_spectrum():
    """A reproducible synthetic Fe 2p spectrum."""
    return generate_fe2p(noise_level=0.01, seed=42)


@pytest.fixture
def c1s_noiseless():
    """A noiseless synthetic C 1s spectrum."""
    return generate_c1s(noise_level=0.0, seed=0)
