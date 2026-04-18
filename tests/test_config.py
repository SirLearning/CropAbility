"""测试配置管理模块。"""

import os
import pytest
from cropability.utils.config import Config, get_config


class TestConfig:
    def test_default_values(self):
        cfg = Config()
        assert cfg.get("gpu.enabled") is True
        assert isinstance(cfg.get("gpu.device_ids"), list)
        assert cfg.get("compute.batch_size") > 0

    def test_dot_path_access(self):
        cfg = Config()
        assert cfg.get("project.name") == "CropAbility"
        assert cfg.get("nonexistent.key", "default") == "default"

    def test_set_value(self):
        cfg = Config()
        cfg.set("gpu.device_ids", [0])
        assert cfg.get("gpu.device_ids") == [0]

    def test_override_from_dict(self):
        cfg = Config({"gpu": {"enabled": False, "device_ids": [0]}})
        assert cfg.get("gpu.enabled") is False
        assert cfg.get("gpu.device_ids") == [0]
        # 默认值不应被清除
        assert cfg.get("gpu.memory_fraction") == 0.90

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CROPABILITY_GPU__ENABLED", "false")
        monkeypatch.setenv("CROPABILITY_COMPUTE__BATCH_SIZE", "256")
        cfg = Config()
        assert cfg.get("gpu.enabled") is False
        assert cfg.get("compute.batch_size") == 256

    def test_contains(self):
        cfg = Config()
        assert "gpu.enabled" in cfg
        assert "nonexistent.path" not in cfg

    def test_getitem(self):
        cfg = Config()
        assert cfg["project.name"] == "CropAbility"
        with pytest.raises(KeyError):
            _ = cfg["nonexistent"]

    def test_as_dict(self):
        cfg = Config()
        d = cfg.as_dict()
        assert isinstance(d, dict)
        assert "gpu" in d
        assert "compute" in d

    def test_singleton(self, tmp_path, monkeypatch):
        # 重置全局单例
        import cropability.utils.config as config_mod
        config_mod._global_config = None
        monkeypatch.chdir(tmp_path)
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2
