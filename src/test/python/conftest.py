"""Shared pytest fixtures and markers for CropAbility."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: Tests requiring CUDA GPU")
    config.addinivalue_line("markers", "slow: Long-running tests")
    config.addinivalue_line(
        "markers",
        "native: Tests requiring cropability.native._core (maturin build)",
    )


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def gpu_available():
    return torch.cuda.is_available()


@pytest.fixture
def small_sequences():
    return [
        "ATCGATCGATCG",
        "GCTAGCTAGCTA",
        "NNNATCGAAANN",
        "ATCGATCGATCG",
    ]


@pytest.fixture
def reference_sequence():
    return "ATCGATCGATCGATCGATCGATCGATCGATCG"


@pytest.fixture(scope="session")
def native_core():
    """Skip the test if the maturin-built extension is missing."""
    pytest.importorskip("cropability.native._core")
    import cropability.native._core as core

    return core
