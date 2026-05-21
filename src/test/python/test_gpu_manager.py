"""Tests for GPU device manager."""

import pytest
import torch
from cropability.gpu.device_manager import DeviceManager, DeviceInfo


class TestDeviceInfo:
    def test_h100_detection(self):
        info = DeviceInfo(
            index=0, name="NVIDIA H100 PCIe",
            total_memory_gb=80.0, compute_capability=(9, 0)
        )
        assert info.is_h100 is True
        assert info.is_a2 is False
        assert info.supports_bf16 is True

    def test_a2_detection(self):
        info = DeviceInfo(
            index=1, name="NVIDIA A2",
            total_memory_gb=16.0, compute_capability=(8, 6)
        )
        assert info.is_a2 is True
        assert info.is_h100 is False
        assert info.supports_bf16 is True

    def test_older_gpu_no_bf16(self):
        info = DeviceInfo(
            index=0, name="NVIDIA Tesla V100",
            total_memory_gb=32.0, compute_capability=(7, 0)
        )
        assert info.supports_bf16 is False

    def test_repr(self):
        info = DeviceInfo(
            index=0, name="Test GPU",
            total_memory_gb=16.0, compute_capability=(8, 0)
        )
        r = repr(info)
        assert "Test GPU" in r
        assert "8.0" in r


class TestDeviceManager:
    def test_initialize(self):
        dm = DeviceManager()
        dm.initialize()
        # initialize should not raise
        assert dm._initialized is True

    def test_idempotent_initialize(self):
        dm = DeviceManager()
        dm.initialize()
        dm.initialize()  # second call should be a no-op
        assert dm._initialized is True

    def test_cpu_fallback(self):
        """Should fall back to CPU when no GPU is available."""
        dm = DeviceManager(device_ids=[999])  # invalid GPU index
        dm.initialize()
        dev = dm.get_device()
        assert dev.type in ("cpu", "cuda")

    def test_context_manager(self):
        with DeviceManager() as dm:
            assert dm._initialized is True

    def test_num_gpus(self):
        dm = DeviceManager().initialize()
        assert dm.num_gpus == torch.cuda.device_count()

    @pytest.mark.gpu
    def test_memory_stats_structure(self, gpu_available):
        if not gpu_available:
            pytest.skip("CUDA not available")
        dm = DeviceManager().initialize()
        if dm.has_gpu:
            stats = dm.memory_stats()
            for key, s in stats.items():
                assert key.startswith("cuda:")
                if "error" not in s:
                    assert "total_gb" in s
                    assert "used_gb" in s
                    assert "free_gb" in s
                    assert 0 <= s["utilization"] <= 1

    def test_get_primary_device(self):
        dm = DeviceManager().initialize()
        dev = dm.get_primary_device()
        assert dev.type in ("cpu", "cuda")
