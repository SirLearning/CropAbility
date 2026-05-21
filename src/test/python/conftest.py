"""Shared pytest fixtures and marker configuration."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: Tests requiring CUDA GPU")
    config.addinivalue_line("markers", "slow: Long-running tests")


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
        "ATCGATCGATCG",  # Same as first row (identity tests)
    ]


@pytest.fixture
def reference_sequence():
    return "ATCGATCGATCGATCGATCGATCGATCGATCG"
